import os
import re
import json
import torch
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path

import requests
from huggingface_hub import configure_http_backend
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
from transformers import TrainerCallback, TrainerControl, TrainerState

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class Config:
    """Configuration class for GLiNER fine-tuning with curriculum learning."""

    BASE_MODEL = r"..."
    OUTPUT_DIR = r"/models"
    CUSTOM_DATASET_PATH = "../data/email_datasets/synthetic/train_email_synthetic.json"
    AGGREGATED_EVAL_PATH = "../data/email_datasets/email_datasets/synthetic/attrprompting/claude/aggregated/aggregated.json"

    ENTITY_LABELS = ["location", "vessel name", "incoterm", "commodity"]

    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    TRAINING_CONFIG = {
        "name": "chartering_gliner_curriculum_fair_simcse",
        "learning_rate": 2e-6,
        "others_lr": 5e-6,
        "weight_decay": 0.01,
        "others_weight_decay": 0.01,
        "batch_size": 6,
        "gradient_accumulation_steps": 4,
        "base_epochs": 4,
        "max_epochs": 10,
        "warmup_ratio": 0.08
    }

    CURRICULUM_CONFIG = {
        "strategy": "progressive_fair",
        "num_stages": 4,
        "initial_percentage": 0.3,
        "final_percentage": 1.0,
        "warmup_fraction": 0.6,
        "use_percentile_based": True,
        "ensure_fair_comparison": True,
        "difficulty_metrics": {
            "num_entities": 0.25,
            "text_length": 0.20,
            "entity_density": 0.20,
            "entity_overlap": 0.15,
            "entity_complexity": 0.20
        },
        "early_stopping_patience": 3,
        "early_stopping_metric": "aggregated_overall_accuracy",
        "save_best_aggregated": True
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
    """
    Tokenize input text into a list of tokens.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None or text == "None":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    return text


# -------------------------------------------------------------------
# Aggregated Dataset Evaluation Functions
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

    result_df['vessel_correct'] = False
    result_df['port_correct'] = False
    result_df['commodity_correct'] = False
    result_df['incoterm_correct'] = False

    result_df['vessel_missing'] = ''
    result_df['vessel_extra'] = ''
    result_df['port_missing'] = ''
    result_df['port_extra'] = ''
    result_df['commodity_missing'] = ''
    result_df['commodity_extra'] = ''
    result_df['incoterm_missing'] = ''
    result_df['incoterm_extra'] = ''

    for idx, row in result_df.iterrows():

        # Vessel evaluation
        extracted = clean_text(row["extracted_vessel_name"])
        label = clean_text(row["label_vessel_name"])
        extracted = re.sub(r'mv\s+|m/v\s+', '', extracted)

        if label:
            results['vessel']['total'] += 1
            vessel_match = label in extracted
            result_df.at[idx, 'vessel_correct'] = vessel_match

            if vessel_match:
                results['vessel']['correct'] += 1
            else:
                result_df.at[idx, 'vessel_missing'] = label
                if extracted:
                    result_df.at[idx, 'vessel_extra'] = extracted

        # Port evaluation
        extracted = clean_text(row['extracted_port'])
        label = clean_text(row['label_port'])
        extracted_clean = re.sub(r'laycan|eur\d+', '', extracted)

        if label:
            results['port']['total'] += 1
            label_ports = [p.strip() for p in label.split(',') if p.strip()]
            missing_ports = []

            port_match = True
            for port in label_ports:
                if port not in extracted_clean:
                    port_match = False
                    missing_ports.append(port)

            result_df.at[idx, 'port_correct'] = port_match

            if port_match:
                results['port']['correct'] += 1
            else:
                result_df.at[idx, 'port_missing'] = ', '.join(missing_ports)
                extracted_ports = [p.strip() for p in extracted_clean.split(',') if p.strip()]
                extra_ports = [p for p in extracted_ports if not any(lp in p for lp in label_ports)]
                if extra_ports:
                    result_df.at[idx, 'port_extra'] = ', '.join(extra_ports)

        # Commodity evaluation
        extracted = clean_text(row['extracted_commodity'])
        label = clean_text(row['label_commodity'])

        if label:
            results['commodity']['total'] += 1
            commodity_match = label in extracted
            result_df.at[idx, 'commodity_correct'] = commodity_match

            if commodity_match:
                results['commodity']['correct'] += 1
            else:
                result_df.at[idx, 'commodity_missing'] = label
                if extracted:
                    result_df.at[idx, 'commodity_extra'] = extracted

        # Incoterm evaluation
        extracted = clean_text(row['extracted_incoterm'])
        label = clean_text(row['label_incoterm'])
        extracted_clean = extracted.replace("terms", "").strip()

        if label:
            results['incoterm']['total'] += 1
            incoterm_match = label in extracted_clean
            result_df.at[idx, 'incoterm_correct'] = incoterm_match

            if incoterm_match:
                results['incoterm']['correct'] += 1
            else:
                result_df.at[idx, 'incoterm_missing'] = label
                if extracted_clean:
                    result_df.at[idx, 'incoterm_extra'] = extracted_clean

    for entity in results:
        results[entity]["accuracy"] = round(results[entity]['correct'] / results[entity]['total'], 6) if \
            results[entity]['total'] > 0 else 0

    total_correct = sum(results[entity_type]['correct'] for entity_type in results)
    total_entities = sum(results[entity_type]['total'] for entity_type in results)
    overall_accuracy = total_correct / total_entities if total_entities > 0 else 0

    results['overall'] = {'accuracy': overall_accuracy, 'correct': total_correct, 'total': total_entities}
    return results, result_df


def evaluate_on_aggregated_dataset(model: GLiNER, config: Config) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Evaluate model on the aggregated dataset.

    Args:
        model: The GLiNER model to evaluate
        config: Configuration object

    Returns:
        Tuple of (evaluation_results, result_dataframe)
    """
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
# Curriculum Learning
# -------------------------------------------------------------------
class CurriculumScorer:
    """Scores training examples by difficulty for curriculum learning."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.entity_stats = defaultdict(int)
        self.max_entities = 0
        self.max_length = 0

    def compute_dataset_statistics(self, examples: List[Dict]):
        """Compute statistics across the dataset for better normalization."""
        lengths = []
        entity_counts = []

        for example in examples:
            lengths.append(len(example['tokenized_text']))
            entity_counts.append(len(example.get('ner', [])))

            for _, _, entity_type in example.get('ner', []):
                self.entity_stats[entity_type] += 1

        self.max_length = np.percentile(lengths, 95) if lengths else 100
        self.max_entities = np.percentile(entity_counts, 95) if entity_counts else 10

        print(f"Dataset statistics - Max length (95th percentile): {self.max_length}")
        print(f"Dataset statistics - Max entities (95th percentile): {self.max_entities}")

    def compute_difficulty_score(self, example: Dict) -> float:
        """
        Compute difficulty score for a training example.

        Args:
            example: Training example with 'tokenized_text' and 'ner' fields

        Returns:
            Difficulty score (0-1, higher = more difficult)
        """
        scores = {}

        # Number of Entities
        num_entities = len(example.get('ner', []))
        scores['num_entities'] = min(num_entities / self.max_entities, 1.0)

        # Text Length
        text_length = len(example['tokenized_text'])
        scores['text_length'] = min(text_length / self.max_length, 1.0)

        # Entity Density
        entity_density = num_entities / max(text_length, 1)
        scores['entity_density'] = min(entity_density * 20, 1.0)

        # Entity Overlap
        overlap_score = self._compute_entity_overlap(example['ner'])
        scores['entity_overlap'] = overlap_score

        # Entity Complexity
        complexity_score = self._compute_entity_complexity(
            example['ner'], example['tokenized_text']
        )
        scores['entity_complexity'] = complexity_score

        # Weighted Combination
        total_score = sum(
            scores[metric] * self.weights[metric]
            for metric in self.weights
        )

        return total_score

    def _compute_entity_overlap(self, ner_spans: List[Tuple]) -> float:
        """Compute score based on overlapping or adjacent entities."""
        if len(ner_spans) < 2:
            return 0.0

        sorted_spans = sorted(ner_spans, key=lambda x: x[0])

        overlaps = 0
        adjacents = 0
        close_entities = 0

        for i in range(len(sorted_spans) - 1):
            span1_end = sorted_spans[i][1]
            span2_start = sorted_spans[i + 1][0]

            gap = span2_start - span1_end

            if gap <= 0:
                overlaps += 1
            elif gap <= 2:
                adjacents += 1
            elif gap <= 5:
                close_entities += 1

        total_challenges = overlaps + adjacents * 0.6 + close_entities * 0.3
        score = total_challenges / max(len(ner_spans) - 1, 1)
        return min(score, 1.0)

    def _compute_entity_complexity(self, ner_spans: List[Tuple],
                                   tokenized_text: List[str]) -> float:
        """Compute complexity based on entity characteristics."""
        if not ner_spans:
            return 0.0

        complexity_scores = []

        for start, end, label in ner_spans:
            entity_tokens = tokenized_text[start:end + 1]
            entity_text = " ".join(entity_tokens)

            complexity = 0.0

            # Multi-token entities
            token_count = len(entity_tokens)
            if token_count > 1:
                complexity += 0.2 + 0.1 * min(token_count - 1, 3)

            # Vessel prefixes
            if any(token in ['M/V', 'MV', 'MT', 'MS', 'm/v'] for token in entity_tokens):
                complexity += 0.15

            # Special characters
            if any(token in [',', '-', '/', '(', ')'] for token in entity_tokens):
                complexity += 0.2

            # Numbers
            if any(any(char.isdigit() for char in token) for token in entity_tokens):
                complexity += 0.15

            # Uppercase tokens
            if any(token.isupper() and len(token) > 1 for token in entity_tokens):
                complexity += 0.1

            # Label-specific complexity
            if label == 'location' and token_count > 2:
                complexity += 0.2
            elif label == 'vessel name' and '/' in entity_text:
                complexity += 0.1
            elif label == 'incoterm':
                complexity += 0.1

            complexity_scores.append(min(complexity, 1.0))

        return np.mean(complexity_scores)


class CurriculumDataset(torch.utils.data.Dataset):
    """Dataset wrapper that supports curriculum learning with fair comparison tracking."""

    def __init__(self, data: List[Dict], difficulty_scores: List[float],
                 use_percentile: bool = True):
        self.data = data
        self.difficulty_scores = np.array(difficulty_scores)
        self.use_percentile = use_percentile
        self.current_difficulty_percentile = 100
        self._original_indices = list(range(len(data)))
        self._indices = self._original_indices.copy()

        self.examples_seen = 0
        self.examples_seen_per_index = defaultdict(int)

        self.percentile_thresholds = np.percentile(
            self.difficulty_scores,
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        )

    def set_curriculum_stage(self, stage: float):
        """
        Set the current curriculum stage.

        Args:
            stage: Either difficulty threshold (0-1) or percentile (0-100)
        """
        if self.use_percentile:
            self.current_difficulty_percentile = stage * 100
            threshold = np.percentile(self.difficulty_scores, self.current_difficulty_percentile)
        else:
            threshold = stage

        self._update_indices(threshold)

    def _update_indices(self, threshold: float):
        """Update available indices based on current difficulty threshold."""
        self._indices = [
            i for i in self._original_indices
            if self.difficulty_scores[i] <= threshold
        ]

        min_examples = min(100, len(self.data) // 10)
        if len(self._indices) < min_examples:
            sorted_indices = np.argsort(self.difficulty_scores)
            self._indices = sorted_indices[:min_examples].tolist()

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        if idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range for curriculum dataset with {len(self._indices)} examples")
        actual_idx = self._indices[idx]

        self.examples_seen += 1
        self.examples_seen_per_index[actual_idx] += 1

        return self.data[actual_idx]

    def get_total_examples_seen(self):
        """Get total number of examples processed during training."""
        return self.examples_seen


class CurriculumTrainer(Trainer):
    """Custom trainer that includes evaluation on aggregated dataset and curriculum tracking"""

    def __init__(self, config: Config, curriculum_dataset: CurriculumDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.curriculum_dataset = curriculum_dataset
        self.best_aggregated_accuracy = 0.0
        self.patience_counter = 0
        self.curriculum_stage_results = {}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include aggregated dataset evaluation"""

        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        aggregated_results, result_df = evaluate_on_aggregated_dataset(self.model, self.config)

        if aggregated_results:
            eval_result['aggregated_vessel_accuracy'] = aggregated_results['vessel']['accuracy']
            eval_result['aggregated_port_accuracy'] = aggregated_results['port']['accuracy']
            eval_result['aggregated_commodity_accuracy'] = aggregated_results['commodity']['accuracy']
            eval_result['aggregated_incoterm_accuracy'] = aggregated_results['incoterm']['accuracy']
            eval_result['aggregated_overall_accuracy'] = aggregated_results['overall']['accuracy']

            current_stage = self.curriculum_dataset.current_difficulty_percentile
            self.curriculum_stage_results[current_stage] = {
                'step': self.state.global_step,
                'examples_seen': self.curriculum_dataset.get_total_examples_seen(),
                'aggregated_accuracy': aggregated_results['overall']['accuracy'],
                'entity_accuracies': {
                    'vessel': aggregated_results['vessel']['accuracy'],
                    'port': aggregated_results['port']['accuracy'],
                    'commodity': aggregated_results['commodity']['accuracy'],
                    'incoterm': aggregated_results['incoterm']['accuracy']
                }
            }

            print("\n" + "=" * 50)
            print("AGGREGATED DATASET EVALUATION RESULTS")
            print("=" * 50)
            print(f"Curriculum Stage: {current_stage:.1f}%")
            print(f"Examples Seen: {self.curriculum_dataset.get_total_examples_seen():,}")
            print(f"Vessel Accuracy: {aggregated_results['vessel']['accuracy']:.2%}")
            print(f"Port Accuracy: {aggregated_results['port']['accuracy']:.2%}")
            print(f"Commodity Accuracy: {aggregated_results['commodity']['accuracy']:.2%}")
            print(f"Incoterm Accuracy: {aggregated_results['incoterm']['accuracy']:.2%}")
            print(f"Overall Accuracy: {aggregated_results['overall']['accuracy']:.2%}")
            print("=" * 50 + "\n")

            if aggregated_results['overall']['accuracy'] > self.best_aggregated_accuracy:
                self.best_aggregated_accuracy = aggregated_results['overall']['accuracy']
                self.patience_counter = 0

                if self.config.CURRICULUM_CONFIG["save_best_aggregated"]:
                    best_model_path = os.path.join(self.args.output_dir, "best_aggregated_model")
                    self.model.save_pretrained(best_model_path)
                    print(f"Saved new best model with aggregated accuracy: {self.best_aggregated_accuracy:.2%}")

                output_path = os.path.join(self.args.output_dir, f"aggregated_eval_step_{self.state.global_step}.csv")
                result_df.to_csv(output_path, index=False)
                print(f"Saved detailed aggregated evaluation results to {output_path}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement in aggregated accuracy. Patience: {self.patience_counter}/{self.config.CURRICULUM_CONFIG['early_stopping_patience']}")

        return eval_result

    def _save_curriculum_progress(self):
        """Save curriculum training progress and statistics"""
        progress_path = os.path.join(self.args.output_dir, "curriculum_progress.json")
        progress_data = {
            'curriculum_stage_results': self.curriculum_stage_results,
            'best_aggregated_accuracy': self.best_aggregated_accuracy,
            'total_examples_seen': self.curriculum_dataset.get_total_examples_seen(),
            'final_step': self.state.global_step
        }
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"Saved curriculum progress to {progress_path}")


class EnhancedCurriculumCallback(TrainerCallback):
    """Enhanced callback for curriculum learning with better tracking and early stopping"""

    def __init__(self, train_dataset: CurriculumDataset,
                 total_train_examples: int,
                 config: Config,
                 trainer: CurriculumTrainer):
        self.train_dataset = train_dataset
        self.config = config
        self.trainer = trainer
        self.base_epochs = config.TRAINING_CONFIG["base_epochs"]
        self.total_train_examples = total_train_examples
        self.target_examples = total_train_examples * self.base_epochs
        self.warmup_fraction = config.CURRICULUM_CONFIG["warmup_fraction"]
        self.initial_percentage = config.CURRICULUM_CONFIG["initial_percentage"]
        self.last_dataset_size = 0
        self.current_epoch = 0
        self.examples_at_epoch_start = 0
        self.stage_transitions = []

        print(f"\nEnhanced Curriculum Learning Configuration:")
        print(f"- Target examples to match baseline: {self.target_examples:,}")
        print(f"- Full dataset size: {total_train_examples:,}")
        print(f"- Starting with {self.initial_percentage * 100:.0f}% easiest examples")
        print(f"- Will reach 100% of data at {self.warmup_fraction * 100:.0f}% of training")
        print(f"- Early stopping patience: {config.CURRICULUM_CONFIG['early_stopping_patience']}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize curriculum at the start of training."""
        self.train_dataset.set_curriculum_stage(self.initial_percentage)
        print(f"\nStarting curriculum learning with {len(self.train_dataset)} examples")

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Track epoch transitions."""
        self.current_epoch = state.epoch
        self.examples_at_epoch_start = self.train_dataset.get_total_examples_seen()

    def on_step_begin(self, args, state, control, **kwargs):
        """Update curriculum stage based on examples seen."""
        if state.global_step == 0:
            return control

        examples_seen = self.train_dataset.get_total_examples_seen()
        progress = min(examples_seen / self.target_examples, 1.0)

        curriculum_progress = min(progress / self.warmup_fraction, 1.0)
        difficulty = self.initial_percentage + (1.0 - self.initial_percentage) * curriculum_progress

        old_size = len(self.train_dataset)
        self.train_dataset.set_curriculum_stage(difficulty)
        current_size = len(self.train_dataset)

        if abs(current_size - old_size) > 50:
            self.stage_transitions.append({
                'step': state.global_step,
                'examples_seen': examples_seen,
                'difficulty': difficulty,
                'dataset_size': current_size
            })

        if abs(current_size - self.last_dataset_size) > 100 or (state.global_step % 100 == 0):
            self.last_dataset_size = current_size
            percentage_complete = (examples_seen / self.target_examples) * 100
            print(f"\nStep {state.global_step}: {examples_seen:,}/{self.target_examples:,} examples "
                  f"({percentage_complete:.1f}% of target)")
            print(f"  Using {current_size:,} examples (top {difficulty * 100:.1f}% by difficulty)")

        if (self.config.CURRICULUM_CONFIG["early_stopping_patience"] > 0 and
                hasattr(self.trainer, 'patience_counter') and
                self.trainer.patience_counter >= self.config.CURRICULUM_CONFIG["early_stopping_patience"]):
            print(f"\n⚠ Early stopping triggered! No improvement for {self.trainer.patience_counter} evaluations.")
            control.should_training_stop = True

        if examples_seen >= self.target_examples:
            print(f"\n✓ Reached target of {self.target_examples:,} examples!")
            print(f"  Total examples seen: {examples_seen:,}")
            print(f"  Best aggregated accuracy: {self.trainer.best_aggregated_accuracy:.2%}")
            control.should_training_stop = True

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Save curriculum statistics at the end of training."""
        stats_path = os.path.join(args.output_dir, "curriculum_statistics.json")
        stats = {
            'stage_transitions': self.stage_transitions,
            'total_examples_seen': self.train_dataset.get_total_examples_seen(),
            'examples_seen_distribution': dict(self.train_dataset.examples_seen_per_index),
            'final_dataset_size': len(self.train_dataset),
            'target_examples': self.target_examples
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved curriculum statistics to {stats_path}")


# -------------------------------------------------------------------
# Data Processing Functions
# -------------------------------------------------------------------
def process_custom_entities_with_scoring(custom_data: List[Dict],
                                         scorer: CurriculumScorer) -> Tuple[List[Dict], List[float]]:
    """
    Process entities and compute difficulty scores.

    Args:
        custom_data: List of examples in custom format
        scorer: CurriculumScorer instance

    Returns:
        Tuple of (processed examples, difficulty scores)
    """
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
                example = {
                    "tokenized_text": tokenized_text,
                    "ner": entity_spans
                }
                all_data.append(example)

        except Exception as e:
            print(f"Error processing data entry: {e}")
            continue

    scorer.compute_dataset_statistics(all_data)

    all_scores = []
    for example in tqdm(all_data, desc="Computing difficulty scores"):
        score = scorer.compute_difficulty_score(example)
        all_scores.append(score)

    return all_data, all_scores


def normalize_difficulty_scores(scores: List[float]) -> List[float]:
    """Normalize scores to ensure good distribution."""
    scores_array = np.array(scores)

    ranks = np.argsort(np.argsort(scores_array))
    normalized = ranks / (len(ranks) - 1)

    return normalized.tolist()


# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------
def load_and_prepare_curriculum_data(config: Config) -> Tuple[CurriculumDataset, List[Dict], List[float], int]:
    """
    Load and prepare training and test datasets with curriculum scoring.

    Args:
        config: Configuration object

    Returns:
        Tuple of (curriculum_train_dataset, test_data, train_scores, total_train_examples)
    """
    print(f"Loading dataset from {config.CUSTOM_DATASET_PATH}...")

    with open(config.CUSTOM_DATASET_PATH, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)

    scorer = CurriculumScorer(config.CURRICULUM_CONFIG["difficulty_metrics"])

    print("Processing dataset and computing difficulty scores...")
    processed_data, difficulty_scores = process_custom_entities_with_scoring(
        custom_data, scorer
    )
    print(f"Dataset size after processing: {len(processed_data)}")

    if config.CURRICULUM_CONFIG["use_percentile_based"]:
        normalized_scores = normalize_difficulty_scores(difficulty_scores)
    else:
        normalized_scores = difficulty_scores

    train_indices, test_indices = train_test_split(
        list(range(len(processed_data))),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    train_data = [processed_data[i] for i in train_indices]
    train_scores = [normalized_scores[i] for i in train_indices]
    test_data = [processed_data[i] for i in test_indices]

    curriculum_train = CurriculumDataset(
        train_data,
        train_scores,
        use_percentile=config.CURRICULUM_CONFIG["use_percentile_based"]
    )

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    print("\nDifficulty distribution (by percentile):")
    for p in [10, 25, 50, 75, 90, 100]:
        threshold = np.percentile(train_scores, p)
        count = sum(1 for s in train_scores if s <= threshold)
        print(f"  {p}th percentile: {count} examples ({count / len(train_scores) * 100:.1f}%)")

    return curriculum_train, test_data, train_scores, len(train_data)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_model_with_curriculum(config: Config, train_dataset: CurriculumDataset,
                                eval_dataset: List[Dict], total_train_examples: int) -> str:
    """
    Train the GLiNER model with enhanced curriculum learning.

    Args:
        config: Configuration object
        train_dataset: Curriculum training dataset
        eval_dataset: Evaluation dataset
        total_train_examples: Total number of training examples

    Returns:
        Path to the saved final model
    """
    print("\n" + "=" * 80)
    print("TRAINING GLINER MODEL WITH ENHANCED CURRICULUM LEARNING")
    print(f"Strategy: {config.CURRICULUM_CONFIG['strategy']}")
    print("=" * 80)

    model = GLiNER.from_pretrained(config.BASE_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )

    model_output_dir = os.path.join(config.OUTPUT_DIR, "curriculum_model")
    os.makedirs(model_output_dir, exist_ok=True)

    batch_size = config.TRAINING_CONFIG["batch_size"]
    gradient_accumulation_steps = config.TRAINING_CONFIG["gradient_accumulation_steps"]

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

        num_train_epochs=config.TRAINING_CONFIG["max_epochs"],

        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,

        fp16=torch.cuda.is_available(),

        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=50,
        report_to="none",

        max_grad_norm=1.0,

        dataloader_drop_last=False,
    )

    trainer = CurriculumTrainer(
        config=config,
        curriculum_dataset=train_dataset,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    curriculum_callback = EnhancedCurriculumCallback(
        train_dataset=train_dataset,
        total_train_examples=total_train_examples,
        config=config,
        trainer=trainer
    )

    trainer.add_callback(curriculum_callback)

    print(f"Training model: {config.TRAINING_CONFIG['name']}")
    trainer.train()

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    final_results = trainer.evaluate()
    print(f"Final evaluation results: {final_results}")

    trainer._save_curriculum_progress()

    final_model_path = os.path.join(config.OUTPUT_DIR, "final_curriculum_model")
    model.save_pretrained(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    if config.CURRICULUM_CONFIG["save_best_aggregated"]:
        best_model_path = os.path.join(model_output_dir, "best_aggregated_model")
        if os.path.exists(best_model_path):
            print(f"Best model (by aggregated accuracy) saved at: {best_model_path}")

    return final_model_path


# -------------------------------------------------------------------
# Exec
# -------------------------------------------------------------------
def main():
    """Main execution function."""
    setup_ssl_backend()
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    curriculum_train, test_data, train_scores, total_train_examples = load_and_prepare_curriculum_data(config)

    target_examples = total_train_examples * config.TRAINING_CONFIG["base_epochs"]
    print(f"\nTraining Plan:")
    print(
        f"- Baseline would train on: {total_train_examples:,} examples × {config.TRAINING_CONFIG['base_epochs']} epochs = {target_examples:,} total examples")
    print(f"- Curriculum will train until: {target_examples:,} examples seen")
    print(
        f"- Estimated epochs needed: ~{target_examples / (total_train_examples * 0.65):.1f} (assuming average 65% of data used)")

    final_model_path = train_model_with_curriculum(
        config, curriculum_train, test_data, total_train_examples
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final model saved at: {final_model_path}")
    print(f"Load with: GLiNER.from_pretrained('{final_model_path}')")

    metadata = {
        "strategy": config.CURRICULUM_CONFIG["strategy"],
        "difficulty_metrics": config.CURRICULUM_CONFIG["difficulty_metrics"],
        "use_percentile_based": config.CURRICULUM_CONFIG["use_percentile_based"],
        "target_examples": target_examples,
        "actual_examples_seen": curriculum_train.get_total_examples_seen(),
        "warmup_fraction": config.CURRICULUM_CONFIG["warmup_fraction"],
        "difficulty_distribution": {
            "min": float(min(train_scores)),
            "max": float(max(train_scores)),
            "mean": float(np.mean(train_scores)),
            "std": float(np.std(train_scores))
        },
        "early_stopping_config": {
            "patience": config.CURRICULUM_CONFIG["early_stopping_patience"],
            "metric": config.CURRICULUM_CONFIG["early_stopping_metric"]
        }
    }

    metadata_path = os.path.join(config.OUTPUT_DIR, "enhanced_curriculum_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Curriculum metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()