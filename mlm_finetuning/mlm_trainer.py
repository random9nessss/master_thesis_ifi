import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config.logger import CustomLogger

logger = CustomLogger(name="debertaMaritimeFTuning")

import requests
from huggingface_hub import configure_http_backend
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# GPU Configuration & Memory Management
# -------------------------------------------------------------------
def setup_gpu_and_memory():
    """Enhanced GPU setup with memory optimization"""
    if not torch.cuda.is_available():
        logger.error("No GPU available! Training will be very slow on CPU.")
        return False, None

    device_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {device_count}")

    torch.cuda.empty_cache()
    gc.collect()

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        cached = torch.cuda.memory_reserved(i) / (1024 ** 3)

        logger.info(f"GPU {i}: {device_name}")
        logger.info(f"  Total: {total_memory:.1f} GB")
        logger.info(f"  Allocated: {allocated:.1f} GB")
        logger.info(f"  Cached: {cached:.1f} GB")

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("Flash Attention enabled")
    except:
        logger.info("Flash Attention not available")

    return True, torch.cuda.get_device_name(0)


# -------------------------------------------------------------------
# Backend Configuration
# -------------------------------------------------------------------
def backend_factory() -> requests.Session:
    """Create a Requests session that disables SSL verification."""
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
def seed_everything(seed=42):
    """Set seeds for reproducible training"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"All seeds set to {seed}")


seed_everything(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Configuration for Maritime Domain Adaptation
# -------------------------------------------------------------------
config = {
    'model': 'microsoft/deberta-v3-small',
    'max_length': 512,
    'tokenizer_max_length': 512,

    'batch_size': 16,
    'effective_batch_size': 128,
    'epochs': 6,
    'mlm_probability': 0.15,
    'eval_ratio': 0.1,

    'learning_rate': 2e-5,
    'warmup_ratio': 0.15,
    'weight_decay': 0.01,
    'lr_scheduler_type': 'cosine_with_restarts',
    'num_cycles': 2,

    'fp16': True,
    'dataloader_num_workers': 4,
    'group_by_length': True,
    'length_column_name': 'word_count',

    'use_curriculum_learning': True,
    'curriculum_epochs': 2,

    'use_quality_weighting': True,
    'quality_power': 0.5,

    'logging_steps': 50,
    'eval_steps': 250,
    'save_steps': 500,
    'eval_accumulation_steps': 1,
    'metric_for_best_model': 'eval_loss',
    'load_best_model_at_end': True,
    'early_stopping_patience': 5,

    'evaluate_maritime_terms': True,
    'maritime_eval_steps': 1000,

    'output_dir': '\models\mlm',
    'save_total_limit': 3,
    'save_strategy': 'steps',

    'preprocessed_data_path': 'data/mlm_data.json',
    'use_quality_filtering': True,
    'min_quality_score': 0.0,
    'min_maritime_relevance': 0.01,
}

# -------------------------------------------------------------------
# Dataset with Quality and Maritime Features
# -------------------------------------------------------------------
class MaritimeTextDataset(Dataset):
    """Enhanced dataset that leverages quality scores and maritime relevance"""

    def __init__(self, texts, tokenizer, max_length, quality_scores=None,
                 maritime_scores=None, metadata=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quality_scores = quality_scores or [1.0] * len(texts)
        self.maritime_scores = maritime_scores or [1.0] * len(texts)
        self.metadata = metadata or [{}] * len(texts)
        self.text_lengths = [len(text.split()) for text in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True
        )

        return {
            'input_ids': encodings['input_ids'][0],
            'attention_mask': encodings['attention_mask'][0],
            'quality_score': torch.tensor(self.quality_scores[idx], dtype=torch.float32),
            'maritime_score': torch.tensor(self.maritime_scores[idx], dtype=torch.float32),
            'word_count': torch.tensor(self.text_lengths[idx], dtype=torch.long)
        }

# -------------------------------------------------------------------
# Quality-Weighted Data Collator
# -------------------------------------------------------------------
class MaritimeDataCollator(DataCollatorForLanguageModeling):
    """Enhanced collator with quality-based masking and maritime-aware processing"""

    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15,
                 use_quality_weighting=False, maritime_term_boost=1.5):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.use_quality_weighting = use_quality_weighting
        self.maritime_term_boost = maritime_term_boost

        # Maritime Terms for Enhanced Masking
        self.maritime_terms = [
            'vessel', 'ship', 'cargo', 'port', 'charter', 'freight', 'maritime',
            'shipping', 'navigation', 'tanker', 'carrier', 'bulk', 'container',
            'demurrage', 'laytime', 'berth', 'terminal', 'loading', 'discharge'
        ]
        self.maritime_token_ids = set()
        for term in self.maritime_terms:
            token_ids = tokenizer.encode(term, add_special_tokens=False)
            self.maritime_token_ids.update(token_ids)

    def torch_call(self, examples):
        quality_scores = None
        maritime_scores = None
        word_counts = None

        if 'quality_score' in examples[0]:
            quality_scores = torch.stack([ex['quality_score'] for ex in examples])
            maritime_scores = torch.stack([ex['maritime_score'] for ex in examples])
            word_counts = torch.stack([ex['word_count'] for ex in examples])

            examples = [{k: v for k, v in ex.items()
                         if k in ['input_ids', 'attention_mask']} for ex in examples]

        batch = super().torch_call(examples)

        if quality_scores is not None:
            batch['quality_scores'] = quality_scores
            batch['maritime_scores'] = maritime_scores
            batch['word_counts'] = word_counts

            if self.use_quality_weighting:
                sample_weights = quality_scores * (1 + maritime_scores)
                batch['sample_weights'] = sample_weights

        return batch


# -------------------------------------------------------------------
# Maritime-Aware Model Wrapper
# -------------------------------------------------------------------
class MaritimeMLMModel(nn.Module):
    """Wrapper around DeBERTa with maritime-specific enhancements"""

    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.maritime_config = config

        self.config = base_model.config

        self.maritime_classifier = nn.Linear(
            base_model.config.hidden_size, 1
        ) if config.get('add_maritime_head', False) else None

    def forward(self, input_ids, attention_mask=None, labels=None,
                quality_scores=None, maritime_scores=None, **kwargs):

        base_model_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['word_counts',
                                          'sample_weights',
                                          'num_items_in_batch']}

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **base_model_kwargs
        )

        if labels is not None and quality_scores is not None:
            loss_weights = quality_scores * (1 + maritime_scores * 0.5)

            if hasattr(outputs, 'loss') and outputs.loss is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits, shift_labels)

                losses = losses.view(labels.size(0), -1)
                weight_mask = (shift_labels.view(labels.size(0), -1) != -100).float()

                weighted_losses = losses * weight_mask
                for i, weight in enumerate(loss_weights):
                    weighted_losses[i] *= weight

                outputs.loss = weighted_losses.sum() / weight_mask.sum()

        return outputs

    def save_pretrained(self, *args, **kwargs):
        """Forward save_pretrained to base model"""
        return self.base_model.save_pretrained(*args, **kwargs)

    def from_pretrained(cls, *args, **kwargs):
        """Forward from_pretrained to base model"""
        return cls.base_model.from_pretrained(*args, **kwargs)


# -------------------------------------------------------------------
# Data Preparation with Curriculum Learning
# -------------------------------------------------------------------
def prepare_enhanced_data(config):
    """Load and prepare data with curriculum learning and quality filtering"""
    logger.info("Loading preprocessed maritime data...")

    # Load preprocessed data
    data_path = config['preprocessed_data_path']
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Preprocessed data not found: {data_path}")

    df = pd.read_json(data_path)
    logger.info(f"Loaded {len(df)} preprocessed texts")

    original_count = len(df)

    if config.get('use_quality_filtering', False):
        min_quality = config.get('min_quality_score', 0.0)
        min_maritime = config.get('min_maritime_relevance', 0.01)

        df = df[
            (df['quality_score'] >= min_quality) &
            (df['maritime_relevance'] >= min_maritime)
            ]

        logger.info(f"After quality filtering: {len(df)} texts (removed {original_count - len(df)})")

    texts = df['content'].tolist()
    quality_scores = df['quality_score'].tolist() if 'quality_score' in df.columns else None
    maritime_scores = df['maritime_relevance'].tolist() if 'maritime_relevance' in df.columns else None
    word_counts = df['word_count'].tolist() if 'word_count' in df.columns else None

    metadata = []
    for _, row in df.iterrows():
        meta = {
            'original_index': row.get('original_index', -1),
            'segment_index': row.get('segment_index', 0),
            'word_count': row.get('word_count', len(row['content'].split())),
            'quality_score': row.get('quality_score', 1.0),
            'maritime_relevance': row.get('maritime_relevance', 1.0)
        }
        metadata.append(meta)

    if quality_scores:
        quality_quartiles = pd.qcut(quality_scores, q=4, labels=False, duplicates='drop')
        stratify = quality_quartiles if len(set(quality_quartiles)) > 1 else None
    else:
        stratify = None

    train_texts, val_texts, train_quality, val_quality, train_maritime, val_maritime, train_meta, val_meta = train_test_split(
        texts, quality_scores, maritime_scores, metadata,
        test_size=config['eval_ratio'],
        random_state=42,
        stratify=stratify
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    if train_quality:
        logger.info(f"Training quality range: {np.min(train_quality):.3f} - {np.max(train_quality):.3f}")
        logger.info(f"Training maritime relevance range: {np.min(train_maritime):.3f} - {np.max(train_maritime):.3f}")

    return (train_texts, val_texts, train_quality, val_quality,
            train_maritime, val_maritime, train_meta, val_meta)


# -------------------------------------------------------------------
# Curriculum Learning Implementation
# -------------------------------------------------------------------
def create_curriculum_datasets(train_texts, train_quality, train_maritime, train_meta,
                               tokenizer, config):
    """Create datasets with curriculum learning progression"""
    datasets = {}

    if config.get('use_curriculum_learning', False) and train_quality:
        sorted_indices = sorted(range(len(train_texts)),
                                key=lambda i: train_quality[i], reverse=True)

        curriculum_epochs = config.get('curriculum_epochs', 2)
        total_epochs = config['epochs']

        # Phase 1: Top 30% quality texts
        phase1_size = int(0.3 * len(train_texts))
        phase1_indices = sorted_indices[:phase1_size]

        # Phase 2: Top 70% quality texts
        phase2_size = int(0.7 * len(train_texts))
        phase2_indices = sorted_indices[:phase2_size]

        # Phase 3: All texts
        phase3_indices = list(range(len(train_texts)))

        for phase, indices in [('phase1', phase1_indices),
                               ('phase2', phase2_indices),
                               ('phase3', phase3_indices)]:
            phase_texts = [train_texts[i] for i in indices]
            phase_quality = [train_quality[i] for i in indices] if train_quality else None
            phase_maritime = [train_maritime[i] for i in indices] if train_maritime else None
            phase_meta = [train_meta[i] for i in indices] if train_meta else None

            datasets[phase] = MaritimeTextDataset(
                texts=phase_texts,
                tokenizer=tokenizer,
                max_length=config['max_length'],
                quality_scores=phase_quality,
                maritime_scores=phase_maritime,
                metadata=phase_meta
            )

            logger.info(f"Curriculum {phase}: {len(phase_texts)} texts")

    else:
        datasets['standard'] = MaritimeTextDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=config['max_length'],
            quality_scores=train_quality,
            maritime_scores=train_maritime,
            metadata=train_meta
        )

    return datasets


# -------------------------------------------------------------------
# Maritime Term Evaluation
# -------------------------------------------------------------------
class MaritimeEvaluator:
    """Evaluator for maritime-specific model capabilities"""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

        self.maritime_tests = [
            "The vessel was loaded with bulk [MASK] at the terminal.",
            "The charterer must pay [MASK] when laytime is exceeded.",
            "The bill of [MASK] serves as a receipt for cargo shipment.",
            "FOB stands for Free on [MASK].",
            "The Suez Canal is a major shipping [MASK] between Europe and Asia.",
            "DWT measures a ship's [MASK] carrying capacity.",
            "Container ships are measured in [MASK] units.",
            "The port authority issued a Notice of [MASK] when the ship arrived."
        ]

        self.expected_answers = [
            ["cargo", "freight", "grain", "oil"],
            ["demurrage", "penalties", "charges"],
            ["lading", "laden"],
            ["board", "Board"],
            ["lane", "route", "channel"],
            ["deadweight", "weight", "cargo"],
            ["TEU", "twenty-foot"],
            ["readiness", "Readiness"]
        ]

    def evaluate_maritime_knowledge(self):
        """Evaluate model's maritime domain knowledge"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        results = []

        with torch.no_grad():
            for test_text, expected in zip(self.maritime_tests, self.expected_answers):
                inputs = self.tokenizer(test_text, return_tensors='pt', padding=True)

                mask_token_id = self.tokenizer.mask_token_id
                mask_pos = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]

                if len(mask_pos) == 0:
                    continue

                outputs = self.model(**inputs)
                predictions = outputs.logits[0, mask_pos[0]]

                top_tokens = torch.topk(predictions, 5)
                predicted_tokens = [self.tokenizer.decode(token_id).strip()
                                    for token_id in top_tokens.indices]

                is_correct = any(pred.lower() in [exp.lower() for exp in expected]
                                 for pred in predicted_tokens)

                if is_correct:
                    correct_predictions += 1

                total_predictions += 1

                results.append({
                    'text': test_text,
                    'expected': expected,
                    'predicted': predicted_tokens,
                    'correct': is_correct
                })

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        logger.info(f"Maritime Knowledge Evaluation:")
        logger.info(f"  Correct: {correct_predictions}/{total_predictions}")
        logger.info(f"  Accuracy: {accuracy:.3f}")

        return accuracy, results


# -------------------------------------------------------------------
# Trainer with Curriculum Learning
# -------------------------------------------------------------------
class MaritimeTrainer(Trainer):
    """Enhanced trainer with maritime-specific features"""

    def __init__(self, maritime_evaluator=None, **kwargs):
        super().__init__(**kwargs)
        self.maritime_evaluator = maritime_evaluator
        self.maritime_eval_results = []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with maritime-specific metrics"""
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if (self.maritime_evaluator and
                self.state.global_step % self.args.logging_steps == 0):
            maritime_accuracy, maritime_results = self.maritime_evaluator.evaluate_maritime_knowledge()
            eval_results[f"{metric_key_prefix}_maritime_accuracy"] = maritime_accuracy

            self.maritime_eval_results.append({
                'step': self.state.global_step,
                'accuracy': maritime_accuracy,
                'results': maritime_results
            })

        return eval_results


# -------------------------------------------------------------------
# Main Training Function
# -------------------------------------------------------------------
def train_maritime_model(config):
    """Training function with all maritime optimizations"""

    gpu_available, gpu_name = setup_gpu_and_memory()
    if not gpu_available:
        logger.warning("Training on CPU - this will be very slow!")

    logger.info(f"Loading tokenizer and model: {config['model']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added padding token")

    base_model = AutoModelForMaskedLM.from_pretrained(config['model'])

    model = MaritimeMLMModel(base_model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable parameters")

    logger.info("Preparing enhanced dataset...")
    (train_texts, val_texts, train_quality, val_quality,
     train_maritime, val_maritime, train_meta, val_meta) = prepare_enhanced_data(config)

    val_dataset = MaritimeTextDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        quality_scores=val_quality,
        maritime_scores=val_maritime,
        metadata=val_meta
    )

    train_datasets = create_curriculum_datasets(
        train_texts, train_quality, train_maritime, train_meta, tokenizer, config
    )

    data_collator = MaritimeDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config['mlm_probability'],
        use_quality_weighting=config.get('use_quality_weighting', False)
    )

    gradient_accumulation_steps = max(1, config['effective_batch_size'] // config['batch_size'])
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=gradient_accumulation_steps,

        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        eval_accumulation_steps=config.get('eval_accumulation_steps', 1),

        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),

        fp16=config.get('fp16', torch.cuda.is_available()),
        gradient_checkpointing=config.get('gradient_checkpointing', False),
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        group_by_length=config.get('group_by_length', True),
        length_column_name=config.get('length_column_name', 'word_count'),

        logging_dir=str(output_dir / "logs"),
        logging_steps=config['logging_steps'],
        save_strategy=config.get('save_strategy', 'steps'),
        save_steps=config['save_steps'],
        save_total_limit=config.get('save_total_limit', 3),

        metric_for_best_model=config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=False,
        load_best_model_at_end=config.get('load_best_model_at_end', True),

        dataloader_drop_last=True,
        prediction_loss_only=False,
        remove_unused_columns=False,

        seed=42,
        data_seed=42,
    )

    maritime_evaluator = MaritimeEvaluator(tokenizer, model)

    if config.get('use_curriculum_learning', False) and len(train_datasets) > 1:
        logger.info("Starting curriculum learning training...")

        curriculum_epochs = config.get('curriculum_epochs', 2)
        total_epochs = config['epochs']

        logger.info("Curriculum Phase 1: Training on highest quality texts...")
        trainer = MaritimeTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_datasets['phase1'],
            eval_dataset=val_dataset,
            maritime_evaluator=maritime_evaluator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get('early_stopping_patience', 5))]
        )

        trainer.args.num_train_epochs = curriculum_epochs // 2
        trainer.train()

        logger.info("Curriculum Phase 2: Expanding to more texts...")
        trainer.train_dataset = train_datasets['phase2']
        trainer.args.num_train_epochs = curriculum_epochs // 2
        trainer.train(resume_from_checkpoint=False)

        logger.info("Curriculum Phase 3: Training on all texts...")
        trainer.train_dataset = train_datasets['phase3']
        trainer.args.num_train_epochs = total_epochs - curriculum_epochs
        trainer.train(resume_from_checkpoint=False)

    else:
        logger.info("Starting standard training...")
        train_dataset = train_datasets.get('standard', list(train_datasets.values())[0])

        trainer = MaritimeTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            maritime_evaluator=maritime_evaluator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get('early_stopping_patience', 5))]
        )

        trainer.train()

    logger.info("Performing final evaluation...")
    final_eval = trainer.evaluate()

    final_maritime_accuracy, final_maritime_results = maritime_evaluator.evaluate_maritime_knowledge()

    logger.info("Saving final model...")
    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))

    create_plots(trainer, maritime_evaluator, output_dir)

    results = {
        'final_eval_loss': final_eval.get('eval_loss', 0),
        'final_maritime_accuracy': final_maritime_accuracy,
        'maritime_eval_history': trainer.maritime_eval_results,
        'config': config
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training completed successfully!")
    logger.info(f"Final evaluation loss: {final_eval.get('eval_loss', 0):.4f}")
    logger.info(f"Final maritime accuracy: {final_maritime_accuracy:.3f}")
    logger.info(f"Model saved to: {final_model_path}")

    return str(final_model_path), results


# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------
def create_plots(trainer, maritime_evaluator, output_dir):
    """Create comprehensive training visualizations"""
    try:
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        train_logs = [x for x in trainer.state.log_history if 'loss' in x and 'eval_loss' not in x]
        eval_logs = [x for x in trainer.state.log_history if 'eval_loss' in x]

        if train_logs and eval_logs:
            train_steps = [x['step'] for x in train_logs]
            train_losses = [x['loss'] for x in train_logs]
            eval_steps = [x['step'] for x in eval_logs]
            eval_losses = [x['eval_loss'] for x in eval_logs]

            axes[0, 0].plot(train_steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
            axes[0, 0].plot(eval_steps, eval_losses, label='Validation Loss', linewidth=2, color='red')
            axes[0, 0].set_title('Training Progress', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        lr_logs = [x for x in trainer.state.log_history if 'learning_rate' in x]
        if lr_logs:
            lr_steps = [x['step'] for x in lr_logs]
            learning_rates = [x['learning_rate'] for x in lr_logs]

            axes[0, 1].plot(lr_steps, learning_rates, color='green', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        if maritime_evaluator.maritime_eval_results:
            maritime_steps = [x['step'] for x in maritime_evaluator.maritime_eval_results]
            maritime_accuracies = [x['accuracy'] for x in maritime_evaluator.maritime_eval_results]

            axes[0, 2].plot(maritime_steps, maritime_accuracies, 'o-', color='purple', linewidth=2, markersize=6)
            axes[0, 2].set_title('Maritime Knowledge Accuracy', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Training Steps')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim(0, 1)

        if train_logs:
            recent_losses = train_losses[-50:] if len(train_losses) > 50 else train_losses
            axes[1, 0].hist(recent_losses, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_title('Recent Training Loss Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        if train_logs and eval_logs:
            stats_text = f"""Training Summary:
                "• Total Steps: {max([x['step'] for x in train_logs]):,}
                • Epochs: {trainer.state.epoch:.1f}
                • Final Train Loss: {train_losses[-1]:.4f}
                • Final Eval Loss: {eval_losses[-1]:.4f}
                • Best Eval Loss: {min(eval_losses):.4f}
                • Maritime Accuracy: {maritime_evaluator.maritime_eval_results[-1]['accuracy']:.3f if maritime_evaluator.maritime_eval_results else 'N/A'}
                
                Model Configuration:
                • Model: {config['model']}
                • Max Length: {config['max_length']}
                • Batch Size: {config['batch_size']}
                • Learning Rate: {config['learning_rate']:.0e}
                • MLM Probability: {config['mlm_probability']}
            """

            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[1, 1].set_title('Training Configuration', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')

        if train_logs and len(train_losses) > 10:
            window = max(5, len(train_losses) // 20)
            smoothed_train = pd.Series(train_losses).rolling(window=window, center=True).mean()

            axes[1, 2].plot(train_steps, train_losses, alpha=0.3, color='blue', label='Raw')
            axes[1, 2].plot(train_steps, smoothed_train, color='blue', linewidth=2, label=f'Smoothed (w={window})')
            if eval_logs:
                axes[1, 2].plot(eval_steps, eval_losses, color='red', linewidth=2, label='Validation')

            axes[1, 2].set_title('Loss Smoothing Analysis', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Training Steps')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'enhanced_training_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced training plots saved to {output_dir / 'enhanced_training_analysis.png'}")

        if maritime_evaluator.maritime_eval_results:
            maritime_df = pd.DataFrame(maritime_evaluator.maritime_eval_results)
            maritime_df.to_csv(output_dir / 'maritime_evaluation_history.csv', index=False)
            logger.info(f"Maritime evaluation history saved to {output_dir / 'maritime_evaluation_history.csv'}")

    except Exception as e:
        logger.error(f"Error creating enhanced plots: {e}")


# -------------------------------------------------------------------
# Exec
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Enhanced Maritime Domain Training...")

    try:
        model_path, results = train_maritime_model(config)
        logger.info("Maritime training completed")
        logger.info(f"Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise