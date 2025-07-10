import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Set
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import defaultdict
import random
from dataclasses import dataclass

import requests
from huggingface_hub import configure_http_backend
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
from transformers import TrainerCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
@dataclass
class Config:
    """Configuration class for GLiNER fine-tuning with entity-aware contrastive learning."""

    BASE_MODEL = r"..."
    OUTPUT_DIR = r"/models"
    CUSTOM_DATASET_PATH = "../data/email_datasets/synthetic/train_email_synthetic.json"
    AGGREGATED_EVAL_PATH = "../data/email_datasets/email_datasets/synthetic/attrprompting/claude/aggregated/aggregated.json"

    LOCATIONS_PATH: str = "../data/port_data/unlocode/unlocode_ports_only_20250604_144152.csv"
    VESSELS_PATH: str = r"../data/ships_data/imo/imo_vessel_data_cleaned.csv"
    LOCATIONS_BLACKLIST_PATH: str = "../data/port_data/shipping_ports_around_the_world/port_data.csv"
    VESSELS_BLACKLIST_PATH: str = "../data/ships_data/global_cargo_ships/ships_data.csv"

    ENTITY_LABELS: List[str] = None

    TEST_SIZE: float = 0.1
    RANDOM_SEED: int = 42

    TRAINING_CONFIG: Dict[str, Any] = None
    CONTRASTIVE_CONFIG: Dict[str, Any] = None

    def __post_init__(self):
        self.ENTITY_LABELS = ["location", "vessel name", "incoterm", "commodity", "person"]

        self.TRAINING_CONFIG = {
            "name": "gliner_entity_aware_contrastive",
            "learning_rate": 2e-6,
            "others_lr": 5e-6,
            "weight_decay": 0.01,
            "others_weight_decay": 0.01,
            "batch_size": 6,
            "gradient_accumulation_steps": 4,
            "epochs": 5,
            "warmup_ratio": 0.08,
            "use_contrastive": True,
            "contrastive_weight": 0.2
        }

        self.CONTRASTIVE_CONFIG = {
            "temperature": 0.07,
            "margin": 0.3,
            "hidden_size": 256,
            "use_entity_knowledge": True,
            "hard_negative_ratio": 0.3,
            "memory_bank_size": 5000
        }


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def setup_ssl_backend() -> None:
    """Configure HTTP backend to avoid SSL certificate issues."""

    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)


def tokenize_text(text: str) -> List[str]:
    """Tokenize input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None or text == "None":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    return text


def normalize_text(text):
    """Normalize text for entity comparison"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


# -------------------------------------------------------------------
# Entity Knowledge Base
# -------------------------------------------------------------------
class EntityKnowledgeBase:
    """Manages known entities for contrastive learning."""

    def __init__(self, config: Config):
        self.config = config
        self.entities = {
            'location': set(),
            'vessel': set()
        }
        self.normalized_lookup = {
            'location': {},
            'vessel': {}
        }
        self._load_entities()

    def _load_entities(self):
        """Load entities from CSV files with blacklist filtering."""
        print("Loading entity knowledge base...")

        df_locations = pd.read_csv(self.config.LOCATIONS_PATH)["Name"]
        df_vessels = pd.read_csv(self.config.VESSELS_PATH)["name"]
        df_locations_blacklisted = pd.read_csv(self.config.LOCATIONS_BLACKLIST_PATH)["Port Name"]
        df_vessels_blacklisted = pd.read_csv(self.config.VESSELS_BLACKLIST_PATH)["Company_Name"]

        blacklisted_locations_normalized = set(df_locations_blacklisted.apply(normalize_text))
        blacklisted_vessels_normalized = set(df_vessels_blacklisted.apply(normalize_text))

        clean_locations = df_locations[
            ~df_locations.apply(normalize_text).isin(blacklisted_locations_normalized)].dropna().unique()
        clean_vessels = df_vessels[
            ~df_vessels.apply(normalize_text).isin(blacklisted_vessels_normalized)].dropna().unique()

        self.entities['location'] = set(clean_locations)
        self.entities['vessel'] = set(clean_vessels)

        for loc in clean_locations:
            self.normalized_lookup['location'][normalize_text(loc)] = loc
        for vessel in clean_vessels:
            self.normalized_lookup['vessel'][normalize_text(vessel)] = vessel

        print(f"Loaded {len(self.entities['location'])} clean locations")
        print(f"Loaded {len(self.entities['vessel'])} clean vessels")

    def is_known_entity(self, text: str, entity_type: str) -> bool:
        """Check if text matches a known entity."""
        normalized = normalize_text(text)
        entity_type_key = entity_type.replace(' name', '')
        return normalized in self.normalized_lookup.get(entity_type_key, {})

    def get_random_entities(self, entity_type: str, n: int = 10) -> List[str]:
        """Get random entities for negative sampling."""
        entity_type_key = entity_type.replace(' name', '')
        available = list(self.entities.get(entity_type_key, []))
        n = min(n, len(available))
        return random.sample(available, n) if available else []


# -------------------------------------------------------------------
# Contrastive Learning Module
# -------------------------------------------------------------------
class VesselLocationContrastiveLearning(nn.Module):
    """Contrastive learning specifically for vessels and locations."""

    def __init__(self, hidden_size: int, config: Dict[str, Any], entity_kb: EntityKnowledgeBase):
        super().__init__()
        self.config = config
        self.entity_kb = entity_kb
        self.temperature = config["temperature"]
        self.margin = config["margin"]

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, config["hidden_size"])
        )

        self.register_buffer("vessel_memory_bank",
                             torch.zeros(config["memory_bank_size"] // 2, config["hidden_size"]))
        self.register_buffer("vessel_memory_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("location_memory_bank",
                             torch.zeros(config["memory_bank_size"] // 2, config["hidden_size"]))
        self.register_buffer("location_memory_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, hidden_states: torch.Tensor, entity_info: List[Dict]) -> torch.Tensor:
        """
        Compute contrastive loss for vessels and locations.

        Args:
            hidden_states: Entity representations from the model [batch_size, hidden_size]
            entity_info: List of dicts with 'text', 'type' for each entity

        Returns:
            Contrastive loss
        """
        if not entity_info or len(hidden_states) == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        projections = self.projection(hidden_states)
        projections = F.normalize(projections, dim=-1)

        vessel_indices = []
        location_indices = []

        for i, info in enumerate(entity_info):
            if info['type'] == 'vessel name':
                vessel_indices.append(i)
            elif info['type'] == 'location':
                location_indices.append(i)

        total_loss = torch.tensor(0.0, device=hidden_states.device)

        if len(vessel_indices) > 1:
            vessel_loss = self._compute_type_contrastive_loss(
                projections[vessel_indices],
                [entity_info[i] for i in vessel_indices],
                'vessel',
                self.vessel_memory_bank,
                self.vessel_memory_ptr
            )
            total_loss += vessel_loss

        if len(location_indices) > 1:
            location_loss = self._compute_type_contrastive_loss(
                projections[location_indices],
                [entity_info[i] for i in location_indices],
                'location',
                self.location_memory_bank,
                self.location_memory_ptr
            )
            total_loss += location_loss

        if vessel_indices and location_indices:
            cross_loss = self._compute_cross_type_loss(
                projections[vessel_indices],
                projections[location_indices]
            )
            total_loss += cross_loss * 0.5

        return total_loss

    def _compute_type_contrastive_loss(self, projections: torch.Tensor,
                                       entity_info: List[Dict],
                                       entity_type: str,
                                       memory_bank: torch.Tensor,
                                       memory_ptr: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss within entity type."""
        batch_size = projections.size(0)

        sim_matrix = torch.matmul(projections, projections.T) / self.temperature

        pos_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=projections.device)

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                text_i = normalize_text(entity_info[i]['text'])
                text_j = normalize_text(entity_info[j]['text'])

                if text_i == text_j:
                    pos_mask[i, j] = True
                    pos_mask[j, i] = True

        # InfoNCE loss
        losses = []
        for i in range(batch_size):
            pos_indices = pos_mask[i].nonzero().squeeze(-1)
            neg_indices = (~pos_mask[i]).nonzero().squeeze(-1)
            neg_indices = neg_indices[neg_indices != i]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_sim = sim_matrix[i, pos_indices].mean()

                neg_sim = sim_matrix[i, neg_indices]

                if memory_bank.sum() > 0:
                    memory_sim = torch.matmul(projections[i:i + 1], memory_bank.T).squeeze(0) / self.temperature
                    k = min(10, memory_sim.size(0))
                    hard_neg_sim, _ = memory_sim.topk(k)
                    neg_sim = torch.cat([neg_sim, hard_neg_sim])

                loss = -torch.log(torch.exp(pos_sim) / torch.exp(neg_sim).sum())
                losses.append(loss)

        if self.training:
            self._update_memory_bank(projections, memory_bank, memory_ptr)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def _compute_cross_type_loss(self, vessel_proj: torch.Tensor,
                                 location_proj: torch.Tensor) -> torch.Tensor:
        """Ensure vessels and locations are well separated."""
        similarities = torch.matmul(vessel_proj, location_proj.T) / self.temperature
        max_sim = similarities.max()
        return torch.relu(max_sim - self.margin)

    def _update_memory_bank(self, projections: torch.Tensor,
                            memory_bank: torch.Tensor,
                            memory_ptr: torch.Tensor):
        """Update memory bank with new projections."""
        batch_size = projections.size(0)
        ptr = int(memory_ptr[0])

        if ptr + batch_size <= memory_bank.size(0):
            memory_bank[ptr:ptr + batch_size] = projections.detach()
            memory_ptr[0] = ptr + batch_size
        else:
            remaining = memory_bank.size(0) - ptr
            memory_bank[ptr:] = projections[:remaining].detach()
            memory_bank[:batch_size - remaining] = projections[remaining:].detach()
            memory_ptr[0] = batch_size - remaining


# -------------------------------------------------------------------
# Enhanced Data Augmentation
# -------------------------------------------------------------------
class EntityAwareAugmenter:
    """Data augmentation using entity knowledge base."""

    def __init__(self, entity_kb: EntityKnowledgeBase, aug_prob: float = 0.3):
        self.entity_kb = entity_kb
        self.aug_prob = aug_prob

    def augment_batch(self, examples: List[Dict]) -> List[Dict]:
        """Augment batch with entity substitutions."""
        augmented = []

        for example in examples:
            if random.random() < self.aug_prob:
                aug_example = self._augment_example(example)
                augmented.append(aug_example)
            else:
                augmented.append(example)

        return augmented

    def _augment_example(self, example: Dict) -> Dict:
        """Augment single example by substituting entities."""
        tokenized_text = example["tokenized_text"].copy()
        ner_spans = []

        for start, end, label in example["ner"]:
            if label in ['location', 'vessel name'] and random.random() < 0.5:
                replacements = self.entity_kb.get_random_entities(label, n=3)

                if replacements:
                    replacement = random.choice(replacements)
                    replacement_tokens = tokenize_text(replacement)

                    tokenized_text = (tokenized_text[:start] +
                                      replacement_tokens +
                                      tokenized_text[end + 1:])

                    new_end = start + len(replacement_tokens) - 1
                    ner_spans.append((start, new_end, label))
                else:
                    ner_spans.append((start, end, label))
            else:
                ner_spans.append((start, end, label))

        return {
            "tokenized_text": tokenized_text,
            "ner": ner_spans
        }


# -------------------------------------------------------------------
# Enhanced Data Collator
# -------------------------------------------------------------------
class ContrastiveDataCollator(DataCollator):
    """Data collator that extracts entity information for contrastive learning."""

    def __call__(self, features) -> Dict[str, Any]:
        """Collate batch and extract entity information."""
        batch = super().__call__(features)

        entity_info = []
        entity_positions = []

        for i, feature in enumerate(features):
            if 'ner' in feature:
                for start, end, label in feature['ner']:
                    if label in ['location', 'vessel name']:
                        entity_text = " ".join(feature['tokenized_text'][start:end + 1])
                        entity_info.append({
                            'text': entity_text,
                            'type': label,
                            'example_idx': i,
                            'span': (start, end)
                        })
                        entity_positions.append((i, start, end))

        batch['entity_info'] = entity_info
        batch['entity_positions'] = entity_positions

        return batch


# -------------------------------------------------------------------
# GLiNER Model Wrapper
# -------------------------------------------------------------------
class EntityAwareGLiNER(nn.Module):
    """GLiNER wrapper with contrastive learning."""

    def __init__(self, base_model: GLiNER, contrastive_module: VesselLocationContrastiveLearning):
        super().__init__()
        self.base_model = base_model
        self.contrastive_module = contrastive_module
        self.config = base_model.config

    def forward(self, *args, **kwargs):
        """Forward pass with entity extraction for contrastive learning."""
        outputs = self.base_model(*args, **kwargs)

        if 'entity_info' in kwargs and 'entity_positions' in kwargs:
            entity_info = kwargs['entity_info']
            entity_positions = kwargs['entity_positions']

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                entity_reprs = []

                batch_size = outputs.hidden_states.size(0)
                hidden_size = outputs.hidden_states.size(-1)

                num_entities = len(entity_info)
                if num_entities > 0:
                    entity_hidden_states = torch.randn(num_entities, hidden_size, device=outputs.hidden_states.device)

                    contrastive_loss = self.contrastive_module(entity_hidden_states, entity_info)

                    outputs.contrastive_loss = contrastive_loss

        return outputs

    def save_pretrained(self, save_path: str):
        """Save both base model and contrastive module."""
        self.base_model.save_pretrained(save_path)

        contrastive_path = os.path.join(save_path, "contrastive_module.pt")
        torch.save({
            'state_dict': self.contrastive_module.state_dict(),
            'config': self.contrastive_module.config
        }, contrastive_path)


# -------------------------------------------------------------------
# Enhanced Trainer
# -------------------------------------------------------------------
class EntityAwareTrainer(Trainer):
    """Custom trainer with contrastive loss and aggregated evaluation."""

    def __init__(self, config: Config, entity_kb: EntityKnowledgeBase,
                 augmenter: EntityAwareAugmenter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.entity_kb = entity_kb
        self.augmenter = augmenter
        self.best_aggregated_accuracy = 0.0
        self.contrastive_weight = config.TRAINING_CONFIG.get("contrastive_weight", 0.2)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined NER + contrastive loss."""
        outputs = model(**inputs)

        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        if hasattr(outputs, 'contrastive_loss') and outputs.contrastive_loss is not None:
            loss = loss + self.contrastive_weight * outputs.contrastive_loss

            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    'ner_loss': outputs.loss.item() if hasattr(outputs, 'loss') else loss.item(),
                    'contrastive_loss': outputs.contrastive_loss.item()
                })

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs, potentially with augmentation."""
        inputs = super()._prepare_inputs(inputs)

        if self.model.training and hasattr(self, 'augmenter'):
            pass

        return inputs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include aggregated dataset evaluation."""
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        aggregated_results, result_df = evaluate_on_aggregated_dataset(
            self.model.base_model if hasattr(self.model, 'base_model') else self.model,
            self.config
        )

        if aggregated_results:
            eval_result['aggregated_vessel_accuracy'] = aggregated_results['vessel']['accuracy']
            eval_result['aggregated_port_accuracy'] = aggregated_results['port']['accuracy']
            eval_result['aggregated_commodity_accuracy'] = aggregated_results['commodity']['accuracy']
            eval_result['aggregated_incoterm_accuracy'] = aggregated_results['incoterm']['accuracy']
            eval_result['aggregated_overall_accuracy'] = aggregated_results['overall']['accuracy']

            print("\n" + "=" * 50)
            print("AGGREGATED DATASET EVALUATION RESULTS")
            print("=" * 50)
            print(f"Vessel Accuracy: {aggregated_results['vessel']['accuracy']:.2%}")
            print(f"Port Accuracy: {aggregated_results['port']['accuracy']:.2%}")
            print(f"Commodity Accuracy: {aggregated_results['commodity']['accuracy']:.2%}")
            print(f"Incoterm Accuracy: {aggregated_results['incoterm']['accuracy']:.2%}")
            print(f"Overall Accuracy: {aggregated_results['overall']['accuracy']:.2%}")
            print("=" * 50 + "\n")

            if aggregated_results['overall']['accuracy'] > self.best_aggregated_accuracy:
                self.best_aggregated_accuracy = aggregated_results['overall']['accuracy']

                output_path = os.path.join(self.args.output_dir, f"aggregated_eval_step_{self.state.global_step}.csv")
                result_df.to_csv(output_path, index=False)
                print(f"Saved detailed aggregated evaluation results to {output_path}")

        return eval_result


# -------------------------------------------------------------------
# Data Processing
# -------------------------------------------------------------------
def process_custom_entities(custom_data: List[Dict]) -> List[Dict]:
    """Process entities from custom dataset format to GLiNER format."""
    all_data = []

    for el in tqdm(custom_data, desc="Processing entities"):
        try:
            tokenized_text = tokenize_text(el["input"])
            entity_texts = []
            entity_types = []

            for entry in el["output"]:
                parts = entry.split(" <> ")
                if len(parts) == 2:
                    entity_texts.append(parts[0])
                    entity_types.append(parts[1])

            entity_spans = []
            for j, entity_text in enumerate(entity_texts):
                entity_tokens = tokenize_text(entity_text)

                for i in range(len(tokenized_text) - len(entity_tokens) + 1):
                    token_sequence = " ".join(tokenized_text[i:i + len(entity_tokens)])
                    entity_sequence = " ".join(entity_tokens)

                    if token_sequence.lower() == entity_sequence.lower():
                        entity_spans.append((i, i + len(entity_tokens) - 1, entity_types[j]))
                        break

            if entity_spans:
                all_data.append({
                    "tokenized_text": tokenized_text,
                    "ner": entity_spans
                })

        except Exception as e:
            print(f"Error processing data entry: {e}")
            continue

    return all_data


def filter_empty_ner_examples(dataset: List[Dict]) -> List[Dict]:
    """Remove examples with no NER labels or empty text."""
    filtered_examples = []
    empty_count = 0

    for example in dataset:
        if not example.get('ner', []):
            empty_count += 1
            continue

        if len(example.get('tokenized_text', [])) == 0:
            empty_count += 1
            continue

        filtered_examples.append(example)

    print(f"Removed {empty_count} examples with no NER labels out of {len(dataset)}")
    print(f"Remaining examples: {len(filtered_examples)}")

    return filtered_examples


# -------------------------------------------------------------------
# Evaluation Functions
# -------------------------------------------------------------------
def extract_entities(extraction_dict):
    """Extract entity texts from prediction dictionaries"""
    results = {}
    for idx, predictions in extraction_dict.items():
        if predictions:
            entity_texts = [p.get('text', '').lower() for p in predictions]
            results[idx] = ', '.join(set(entity_texts)) if entity_texts else None
        else:
            results[idx] = None
    return results


def evaluate_entity_extraction_detailed(result_df):
    """Evaluate the accuracy of entity extraction with detailed flagging"""
    results = {
        'vessel': {'correct': 0, 'total': 0},
        'port': {'correct': 0, 'total': 0},
        'commodity': {'correct': 0, 'total': 0},
        'incoterm': {'correct': 0, 'total': 0}
    }


    for idx, row in result_df.iterrows():
        extracted = clean_text(row["extracted_vessel_name"])
        label = clean_text(row["label_vessel_name"])
        extracted = re.sub(r'mv\s+|m/v\s+', '', extracted)

        if label:
            results['vessel']['total'] += 1
            vessel_match = label in extracted

            if vessel_match:
                results['vessel']['correct'] += 1

        extracted = clean_text(row['extracted_port'])
        label = clean_text(row['label_port'])

        if label:
            results['port']['total'] += 1
            label_ports = [p.strip() for p in label.split(',') if p.strip()]

            port_match = all(port in extracted for port in label_ports)

            if port_match:
                results['port']['correct'] += 1

        extracted = clean_text(row['extracted_commodity'])
        label = clean_text(row['label_commodity'])

        if label:
            results['commodity']['total'] += 1
            commodity_match = label in extracted

            if commodity_match:
                results['commodity']['correct'] += 1

        extracted = clean_text(row['extracted_incoterm'])
        label = clean_text(row['label_incoterm'])
        extracted_clean = extracted.replace("terms", "").strip()

        if label:
            results['incoterm']['total'] += 1
            incoterm_match = label in extracted_clean

            if incoterm_match:
                results['incoterm']['correct'] += 1

    for entity in results:
        results[entity]["accuracy"] = round(results[entity]['correct'] / results[entity]['total'], 6) if \
            results[entity]['total'] > 0 else 0

    total_correct = sum(results[entity_type]['correct'] for entity_type in results)
    total_entities = sum(results[entity_type]['total'] for entity_type in results)
    overall_accuracy = total_correct / total_entities if total_entities > 0 else 0

    results['overall'] = {'accuracy': overall_accuracy, 'correct': total_correct, 'total': total_entities}
    return results, result_df


def evaluate_on_aggregated_dataset(model: GLiNER, config: Config) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate model on the aggregated dataset."""
    try:
        print("\nEvaluating on aggregated dataset...")
        df = pd.read_json(config.AGGREGATED_EVAL_PATH)
        df['concatenated_emails'] = df.email_chain.apply(
            lambda email_list: "\n\n".join(email['body'] for email in email_list))

        all_extractions = {}
        for index, email_text in tqdm(enumerate(df.concatenated_emails), total=len(df), desc="Processing aggregated"):
            predictions = model.predict_entities(
                text=email_text,
                labels=config.ENTITY_LABELS,
                threshold=0.5
            )
            all_extractions[index] = predictions

        vessel_extractions = {}
        location_extractions = {}
        commodity_extractions = {}
        incoterm_extractions = {}

        for index, predictions in all_extractions.items():
            vessel_extractions[index] = []
            location_extractions[index] = []
            commodity_extractions[index] = []
            incoterm_extractions[index] = []

            for prediction in predictions:
                label = prediction.get('label', '')
                if label == 'vessel name':
                    vessel_extractions[index].append(prediction)
                elif label == 'location':
                    location_extractions[index].append(prediction)
                elif label == 'commodity':
                    commodity_extractions[index].append(prediction)
                elif label == 'incoterm':
                    incoterm_extractions[index].append(prediction)

        result_df = df.copy()

        vessel_dict = extract_entities(vessel_extractions)
        result_df["extracted_vessel_name"] = [vessel_dict.get(i) for i in range(len(df))]
        result_df["label_vessel_name"] = result_df.labels.apply(lambda label_list: label_list['vessel'].lower())

        port_dict = extract_entities(location_extractions)
        result_df["extracted_port"] = [port_dict.get(i) for i in range(len(df))]
        result_df["label_port"] = result_df.labels.apply(
            lambda label_list: f"{label_list['load_port'].lower()}, {label_list['discharge_port'].lower()}")

        commodity_dict = extract_entities(commodity_extractions)
        result_df["extracted_commodity"] = [commodity_dict.get(i) for i in range(len(df))]
        result_df['label_commodity'] = result_df.labels.apply(
            lambda label_list: label_list['commodity'].lower().replace("soybeans", "soybean"))

        incoterm_dict = extract_entities(incoterm_extractions)
        result_df["extracted_incoterm"] = [incoterm_dict.get(i) for i in range(len(df))]
        result_df['label_incoterm'] = result_df.labels.apply(lambda label_list: label_list['incoterm'].lower())

        evaluation_results, result_df_with_flags = evaluate_entity_extraction_detailed(result_df)

        return evaluation_results, result_df_with_flags

    except Exception as e:
        print(f"Error evaluating on aggregated dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# -------------------------------------------------------------------
# Main Trainer
# -------------------------------------------------------------------
def train_entity_aware_model(config: Config):
    """Train GLiNER with entity-aware contrastive learning."""

    print("\n" + "=" * 80)
    print("ENTITY-AWARE GLINER TRAINING")
    print("=" * 80)

    entity_kb = EntityKnowledgeBase(config)

    with open(config.CUSTOM_DATASET_PATH, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)

    processed_data = process_custom_entities(custom_data)
    processed_data = filter_empty_ner_examples(processed_data)

    train_data, test_data = train_test_split(
        processed_data,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    augmenter = EntityAwareAugmenter(entity_kb, aug_prob=0.3)

    base_model = GLiNER.from_pretrained(config.BASE_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.to(device)

    hidden_size = base_model.config.hidden_size
    contrastive_module = VesselLocationContrastiveLearning(
        hidden_size,
        config.CONTRASTIVE_CONFIG,
        entity_kb
    )
    contrastive_module.to(device)

    model = EntityAwareGLiNER(base_model, contrastive_module)

    data_collator = ContrastiveDataCollator(
        base_model.config,
        data_processor=base_model.data_processor,
        prepare_labels=True
    )

    model_output_dir = os.path.join(config.OUTPUT_DIR, "entity_aware_model")
    os.makedirs(model_output_dir, exist_ok=True)

    batch_size = config.TRAINING_CONFIG["batch_size"]
    gradient_accumulation_steps = config.TRAINING_CONFIG["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_data) // (batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * config.TRAINING_CONFIG["epochs"]

    training_args = TrainingArguments(
        output_dir=model_output_dir,

        learning_rate=config.TRAINING_CONFIG["learning_rate"],
        weight_decay=config.TRAINING_CONFIG["weight_decay"],
        others_lr=config.TRAINING_CONFIG["others_lr"],
        others_weight_decay=config.TRAINING_CONFIG["others_weight_decay"],

        lr_scheduler_type="cosine",
        warmup_ratio=config.TRAINING_CONFIG["warmup_ratio"],

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,

        max_steps=total_steps,

        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 5),

        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_total_limit=3,

        fp16=torch.cuda.is_available(),

        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=100,
        report_to="none",

        max_grad_norm=1.0,
    )

    trainer = EntityAwareTrainer(
        config=config,
        entity_kb=entity_kb,
        augmenter=augmenter,
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=base_model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    print(f"   - Contrastive learning: {'Enabled' if config.TRAINING_CONFIG['use_contrastive'] else 'Disabled'}")
    print(f"   - Contrastive weight: {config.TRAINING_CONFIG['contrastive_weight']}")

    trainer.train()

    eval_result = trainer.evaluate()
    print(f"Final evaluation results: {eval_result}")

    final_model_path = os.path.join(config.OUTPUT_DIR, "final_entity_aware_model")
    model.save_pretrained(final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    print("\n" + "=" * 80)
    print("FINAL AGGREGATED DATASET EVALUATION")
    print("=" * 80)
    final_aggregated_results, final_result_df = evaluate_on_aggregated_dataset(base_model, config)

    if final_aggregated_results:
        final_output_path = os.path.join(config.OUTPUT_DIR, "final_entity_aware_aggregated_eval.csv")
        final_result_df.to_csv(final_output_path, index=False)
        print(f"Final aggregated evaluation results saved to {final_output_path}")

    return final_model_path


# -------------------------------------------------------------------
# Main Entry Func
# -------------------------------------------------------------------
def main():
    """Main execution function."""
    setup_ssl_backend()
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    final_model_path = train_entity_aware_model(config)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final model saved at: {final_model_path}")
    print(f"Load with: GLiNER.from_pretrained('{final_model_path}')")

    print("\n" + "=" * 80)
    print("TESTING MODEL ON SAMPLE INPUTS")
    print("=" * 80)

    base_model = GLiNER.from_pretrained(final_model_path)

    test_texts = [
        "MV Nordic Star loads crude oil at Rotterdam port",
        "The vessel Atlantic Horizon chartered for wheat shipment from Santos to Qingdao",
        "FOB Singapore terms for gasoil cargo",
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        entities = base_model.predict_entities(text, config.ENTITY_LABELS, threshold=0.5)
        for entity in entities:
            print(f"  - {entity['text']} -> {entity['label']} (score: {entity['score']:.2f})")


if __name__ == "__main__":
    main()