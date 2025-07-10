import os
import random
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments as HfTrainingArguments,
    HfArgumentParser,
    set_seed,
    DataCollatorWithPadding,
    AutoConfig,
)

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Logging Config
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContrastiveTrainer")


# ----------------------------------------
# SimCSE Model Components
# ----------------------------------------
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], \
            f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]

        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result

        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result

        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


# ----------------------------------------
# AllGather for Distributed Training
# ----------------------------------------
class AllGather(torch.autograd.Function):
    """
    Gathers tensors from all processes, supporting gradients.
    Requires distributed process group to be initialized.
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank() if dist.is_initialized() else 0
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if ctx.world_size == 1:
            return tensor.clone()

        ctx.local_batch_size = tensor.shape[0]

        all_tensors = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(all_tensors, tensor)
        return torch.cat(all_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size == 1:
            return grad_output

        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)

        start_idx = ctx.local_batch_size * ctx.rank
        end_idx = start_idx + ctx.local_batch_size

        if end_idx > grad_output.shape[0]:
            logger.warning(
                f"Rank {ctx.rank}: Slice end index {end_idx} > grad_output size {grad_output.shape[0]}. Clipping.")
            end_idx = grad_output.shape[0]

        if start_idx >= grad_output.shape[0]:
            logger.error(
                f"Rank {ctx.rank}: Slice start index {start_idx} >= grad_output size {grad_output.shape[0]}. Returning None gradient.")
            return None

        return grad_output[start_idx: end_idx]


# ----------------------------------------
# Enhanced InfoNCE Loss with SimCSE Features
# ----------------------------------------
class InfoNCE(nn.Module):
    def __init__(self, device, temperature=0.05, hard_negative_weight=0.0):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.sim = Similarity(temp=temperature)

    def forward(self, z1, z2, z3=None):
        """
        InfoNCE loss using in-batch negatives and provided hard negatives.
        z1, z2, z3 : [B, E] (embeddings)
        z1: query embeddings
        z2: positive embeddings
        z3: hard negative embeddings (optional)
        """
        batch_size = z1.shape[0]

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        if z3 is not None:
            z3 = F.normalize(z3, p=2, dim=-1)

        if dist.is_initialized() and z1.requires_grad:
            if z3 is not None:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        if z3 is not None:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

            if self.hard_negative_weight > 0:
                weights = torch.tensor(
                    [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [self.hard_negative_weight] + [
                        0.0] * (z1_z3_cos.size(-1) - i - 1)
                     for i in range(z1_z3_cos.size(-1))]
                ).to(self.device)
                cos_sim = cos_sim + weights

        labels = torch.arange(batch_size).long().to(self.device)
        if dist.is_initialized():
            labels = labels + batch_size * dist.get_rank()

        loss = F.cross_entropy(cos_sim, labels, reduction='mean')
        return loss


# ----------------------------------------
# Dataset Class with MLM Support
# ----------------------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256, do_mlm=False, mlm_probability=0.15):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_mlm = do_mlm
        self.mlm_probability = mlm_probability

        initial_len = len(self.dataframe)
        required_cols = ['sent0', 'sent1', 'hard_neg']
        if not all(col in self.dataframe.columns for col in required_cols):
            raise ValueError(
                f"Missing required columns. Need: {required_cols}, Found: {self.dataframe.columns.tolist()}")

        self.dataframe = self.dataframe.dropna(subset=required_cols)
        if len(self.dataframe) < initial_len:
            logger.warning(f"Dropped {initial_len - len(self.dataframe)} rows with NaN values.")
            initial_len = len(self.dataframe)

        for col in required_cols:
            self.dataframe[col] = self.dataframe[col].astype(str)
            self.dataframe = self.dataframe[self.dataframe[col].str.strip().astype(bool)]

        if len(self.dataframe) < initial_len:
            logger.warning(f"Dropped {initial_len - len(self.dataframe)} rows with empty strings after cleaning.")

        if len(self.dataframe) == 0:
            raise ValueError("Dataset empty after cleaning. Check input CSV.")

        self.dataframe = self.dataframe.reset_index(drop=True)
        logger.info(f"Dataset initialized with {len(self.dataframe)} valid examples.")

    def __len__(self):
        return len(self.dataframe)

    def mask_tokens(self, inputs: List[int]) -> Tuple[List[int], List[int]]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = torch.tensor(inputs)
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.tensor([
            self.tokenizer.get_special_tokens_mask(inputs.tolist(), already_has_special_tokens=True)
        ], dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs.tolist(), labels.tolist()

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            sent0, sent1, hard_neg = str(row["sent0"]), str(row["sent1"]), str(row["hard_neg"])

            enc0 = self.tokenizer(sent0, truncation=True, max_length=self.max_length, padding=False,
                                  return_tensors=None)

            enc1 = self.tokenizer(sent1, truncation=True, max_length=self.max_length, padding=False,
                                  return_tensors=None)

            enc_neg = self.tokenizer(hard_neg, truncation=True, max_length=self.max_length, padding=False,
                                     return_tensors=None)

            if not enc0['input_ids'] or not enc1['input_ids'] or not enc_neg['input_ids']:
                raise ValueError(f"Empty tokenization result for index {idx}")

            item = {
                'sent0_input_ids': enc0['input_ids'],
                'sent0_attention_mask': enc0['attention_mask'],
                'sent1_input_ids': enc1['input_ids'],
                'sent1_attention_mask': enc1['attention_mask'],
                'hard_neg_input_ids': enc_neg['input_ids'],
                'hard_neg_attention_mask': enc_neg['attention_mask'],
            }

            if self.do_mlm:
                mlm_input_ids, mlm_labels = self.mask_tokens(enc0['input_ids'])
                item.update({
                    'mlm_input_ids': mlm_input_ids,
                    'mlm_attention_mask': enc0['attention_mask'],
                    'mlm_labels': mlm_labels,
                })

            return item

        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")
            try:
                logger.error(f"Problematic row data: {self.dataframe.iloc[idx].to_dict()}")
            except:
                pass
            raise RuntimeError(f"Failed to process item at index {idx}") from e


# ----------------------------------------
# Data Collator with MLM Support
# ----------------------------------------
def collate_triplets(features: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """
    Collates list of dicts from ContrastiveDataset into a padded batch.
    """
    default_collator = DataCollatorWithPadding(tokenizer, padding='longest')

    valid_features = [f for f in features if f is not None]
    if not valid_features:
        logger.error("Batch contains no valid features after filtering None items.")
        return {}

    if len(valid_features) < len(features):
        logger.warning(f"Filtered {len(features) - len(valid_features)} None items from batch.")

    try:
        sent0_batch = default_collator(
            [{'input_ids': f['sent0_input_ids'], 'attention_mask': f['sent0_attention_mask']} for f in valid_features])
        sent1_batch = default_collator(
            [{'input_ids': f['sent1_input_ids'], 'attention_mask': f['sent1_attention_mask']} for f in valid_features])
        hard_neg_batch = default_collator(
            [{'input_ids': f['hard_neg_input_ids'], 'attention_mask': f['hard_neg_attention_mask']} for f in
             valid_features])

        batch = {
            'sent0_input_ids': sent0_batch['input_ids'],
            'sent0_attention_mask': sent0_batch['attention_mask'],
            'sent1_input_ids': sent1_batch['input_ids'],
            'sent1_attention_mask': sent1_batch['attention_mask'],
            'hard_neg_input_ids': hard_neg_batch['input_ids'],
            'hard_neg_attention_mask': hard_neg_batch['attention_mask'],
        }

        if 'mlm_input_ids' in valid_features[0]:
            mlm_batch = default_collator([{
                'input_ids': f['mlm_input_ids'],
                'attention_mask': f['mlm_attention_mask']
            } for f in valid_features])

            mlm_labels_batch = default_collator([{
                'input_ids': f['mlm_labels'],
                'attention_mask': f['mlm_attention_mask']
            } for f in valid_features])

            batch.update({
                'mlm_input_ids': mlm_batch['input_ids'],
                'mlm_attention_mask': mlm_batch['attention_mask'],
                'mlm_labels': mlm_labels_batch['input_ids'],
            })

        return batch

    except KeyError as e:
        logger.error(f"KeyError during collation: {e}. Feature dictionary likely malformed.")
        logger.error(f"First feature keys: {list(valid_features[0].keys()) if valid_features else 'N/A'}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during collation: {e}")
        raise e


# ----------------------------------------
# Custom Trainer with SimCSE Features
# ----------------------------------------
class ContrastiveTrainer(Trainer):
    def __init__(self, model_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args or ModelArguments()

        self.info_nce = InfoNCE(
            device=self.args.device,
            temperature=self.model_args.temp,
            hard_negative_weight=self.model_args.hard_negative_weight
        )

        self.pooler = Pooler(self.model_args.pooler_type)
        if self.model_args.pooler_type == "cls":
            self.mlp = MLPLayer(self.model.config)
            self.mlp.to(self.args.device)

        if self.model_args.do_mlm:
            from transformers import BertLMPredictionHead, RobertaLMHead

            if 'bert' in self.model.config.model_type.lower():
                self.lm_head = BertLMPredictionHead(self.model.config)

            elif 'roberta' in self.model.config.model_type.lower():
                self.lm_head = RobertaLMHead(self.model.config)

            else:
                self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)
            self.lm_head.to(self.args.device)

    def encode(self, model, x):
        """Encode input using SimCSE pooling strategy"""
        x = {k: v.to(model.device) for k, v in x.items()}

        need_hidden_states = self.model_args.pooler_type in ['avg_top2', 'avg_first_last']
        outputs = model(**x, output_hidden_states=need_hidden_states)

        pooled_output = self.pooler(x['attention_mask'], outputs)

        if self.model_args.pooler_type == "cls" and hasattr(self, 'mlp'):
            pooled_output = self.mlp(pooled_output)

        return pooled_output

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = self._prepare_inputs(inputs)

        sent0 = {'input_ids': inputs.get('sent0_input_ids'),
                 'attention_mask': inputs.get('sent0_attention_mask')}

        sent1 = {'input_ids': inputs.get('sent1_input_ids'),
                 'attention_mask': inputs.get('sent1_attention_mask')}

        hard_neg = {'input_ids': inputs.get('hard_neg_input_ids'),
                    'attention_mask': inputs.get('hard_neg_attention_mask')}

        if sent0['input_ids'] is None or sent1['input_ids'] is None or hard_neg['input_ids'] is None:
            logger.error("Missing input tensors in compute_loss. Check collation and device placement.")
            raise ValueError("Input tensors missing in compute_loss")

        sent0_embed = self.encode(model, sent0)
        sent1_embed = self.encode(model, sent1)
        hard_neg_embed = self.encode(model, hard_neg)

        loss = self.info_nce(sent0_embed, sent1_embed, hard_neg_embed)

        if self.model_args.do_mlm and 'mlm_input_ids' in inputs:
            mlm_inputs = {
                'input_ids': inputs['mlm_input_ids'],
                'attention_mask': inputs['mlm_attention_mask']
            }
            mlm_inputs = {k: v.to(model.device) for k, v in mlm_inputs.items()}

            mlm_outputs = model(**mlm_inputs, output_hidden_states=False)
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)

            mlm_labels = inputs['mlm_labels'].to(model.device)
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, model.config.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100
            )

            loss = loss + self.model_args.mlm_weight * masked_lm_loss

        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to handle the custom format of inputs."""
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

    def evaluate_senteval(self):
        """
        Evaluate using SentEval (if available)
        """
        try:
            import sys
            PATH_TO_SENTEVAL = './SentEval'
            PATH_TO_DATA = './SentEval/data'
            sys.path.insert(0, PATH_TO_SENTEVAL)
            import senteval
        except ImportError:
            logger.warning("SentEval not available. Skipping SentEval evaluation.")
            return {}

        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch_encoded = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            for k in batch_encoded:
                batch_encoded[k] = batch_encoded[k].to(self.args.device)

            with torch.no_grad():
                pooled_output = self.encode(self.model, batch_encoded)
            return pooled_output.cpu()

        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {
            'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
            'tenacity': 3, 'epoch_size': 2
        }

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']

        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            "eval_sickr_spearman": sickr_spearman,
            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2
        }

        logger.info(f"SentEval Results: {metrics}")
        return metrics


# ----------------------------------------
# Configuration Dataclasses
# ----------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="./SimCSEFinetuning/deberta-contrastive-finetuned",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    hidden_dropout_prob: Optional[float] = field(
        default=0.1, metadata={"help": "Override model's hidden dropout probability."}
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=None, metadata={"help": "Override model's attention dropout probability."}
    )

    temp: float = field(
        default=0.05,
        metadata={"help": "Temperature for softmax."}
    )
    pooler_type: str = field(
        default="cls_before_pooler",
        metadata={"help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."}
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={"help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."}
    )
    do_mlm: bool = field(
        default=False,
        metadata={"help": "Whether to use MLM auxiliary objective."}
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."}
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={"help": "Use MLP only during training"}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to the training CSV file (needs columns: sent0, sent1, hard_neg)"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the evaluation CSV file. If None, use train/eval split."}
    )
    eval_split_ratio: float = field(
        default=0.00001,
        metadata={"help": "Ratio of training data to use for evaluation if eval_data_path is not provided."}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum total input sequence length after tokenization."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    output_dir: str = field(default="./deberta-base-mlm-contrastive-finetuned")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=32)
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    per_device_eval_batch_size: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: int = field(default=30)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=5000)
    save_total_limit: Optional[int] = field(default=2)
    load_best_model_at_end: bool = field(default=False)
    metric_for_best_model: str = field(default="loss")
    greater_is_better: bool = field(default=False)
    fp16: bool = field(default=False, metadata={"help": "Enable fp16 training if CUDA available."})
    bf16: bool = field(default=False, metadata={"help": "Enable bf16 training."})
    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"])
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=True)
    seed: int = field(default=42)
    remove_unused_columns: bool = field(default=False, metadata={"help": "Must be False for custom collator"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run evaluation."})
    eval_senteval: bool = field(default=False, metadata={"help": "Whether to run SentEval evaluation."})


# ----------------------------------------
# Main Function
# ----------------------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    logger.info(f"Loading tokenizer and config from {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    dropout_modified = False

    if model_args.hidden_dropout_prob is not None:
        logger.info(
            f"Overriding hidden_dropout_prob from {config.hidden_dropout_prob} to {model_args.hidden_dropout_prob}")
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        dropout_modified = True

    if model_args.attention_probs_dropout_prob is not None:
        logger.info(
            f"Overriding attention_probs_dropout_prob from {config.attention_probs_dropout_prob} to {model_args.attention_probs_dropout_prob}")
        config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
        dropout_modified = True

    if dropout_modified:
        logger.info("Loading model with MODIFIED dropout configuration.")
    else:
        logger.info("Loading model with default dropout configuration.")

    logger.info(f"Loading model with custom config from {model_args.model_name_or_path}...")
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )

    logger.info("Model and tokenizer loaded.")

    logger.info(f"Loading data from {data_args.train_data_path}...")

    try:
        full_df = pd.read_csv(data_args.train_data_path)
    except FileNotFoundError:
        logger.error(f"Training data file not found at: {data_args.train_data_path}")
        return
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return

    train_df = None
    eval_df = None
    if data_args.eval_data_path:
        logger.info(f"Loading evaluation data from {data_args.eval_data_path}...")
        try:
            eval_df = pd.read_csv(data_args.eval_data_path)
            train_df = full_df
        except FileNotFoundError:
            logger.error(
                f"Evaluation data file not found: {data_args.eval_data_path}. Falling back to splitting training data.")
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}. Falling back to splitting training data.")

    if eval_df is None:
        logger.info(f"Splitting training data for evaluation with ratio: {data_args.eval_split_ratio}")
        if len(full_df) < 2:
            logger.error("Dataset too small to split for evaluation.")
            return
        try:
            train_df, eval_df = train_test_split(
                full_df,
                test_size=data_args.eval_split_ratio,
                random_state=training_args.seed,
                shuffle=True
            )
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return

    try:
        train_dataset = ContrastiveDataset(
            train_df.copy(),
            tokenizer,
            max_length=data_args.max_seq_length,
            do_mlm=model_args.do_mlm,
            mlm_probability=data_args.mlm_probability
        )
        eval_dataset = ContrastiveDataset(
            eval_df.copy(),
            tokenizer,
            max_length=data_args.max_seq_length,
            do_mlm=False
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"Failed to create datasets: {e}")
        return

    logger.info(f"Using {len(train_dataset)} examples for training and {len(eval_dataset)} for evaluation.")

    data_collator = partial(collate_triplets, tokenizer=tokenizer)

    trainer = ContrastiveTrainer(
        model_args=model_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    logger.info("Starting training...")
    try:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir) and any(
                f.startswith("checkpoint") for f in os.listdir(training_args.output_dir)):
            logger.info(
                f"Found potential checkpoints in {training_args.output_dir}, but resume_from_checkpoint not specified. Starting from scratch.")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        logger.info("Training finished successfully.")

        if training_args.do_eval:
            logger.info("Starting evaluation...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        if training_args.eval_senteval:
            logger.info("Starting SentEval evaluation...")
            senteval_metrics = trainer.evaluate_senteval()
            if senteval_metrics:
                trainer.log_metrics("senteval", senteval_metrics)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)

    logger.info(f"Script finished. Check {training_args.output_dir} for results.")


if __name__ == "__main__":
    main()