import random
import os
import ast
import pandas as pd
import json
import re
from pathlib import Path
from tqdm.auto import tqdm
import torch
import wandb
from transformers import TrainerCallback
from seqeval.metrics import f1_score, precision_score, recall_score
from utils import PredictionThresholdScheduler
from gliner import GLiNER


# -------------------------------------------------------------------
# Shipping Dataset Helper Functions
# -------------------------------------------------------------------
def clean_text(text):
    """Clean and normalize text for comparison"""
    if pd.isna(text) or text is None or text == "None":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    return text


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


def evaluate_entity_extraction(result_df):
    """Evaluate the accuracy of entity extraction for shipping dataset"""
    results = {
        'vessel': {'correct': 0, 'total': 0},
        'port': {'correct': 0, 'total': 0},
        'commodity': {'correct': 0, 'total': 0},
        'incoterm': {'correct': 0, 'total': 0}
    }

    for _, row in result_df.iterrows():
        # Vessel information
        extracted = clean_text(row["extracted_vessel_name"])
        label = clean_text(row["label_vessel_name"])
        extracted = re.sub(r'mv\s+|m/v\s+', '', extracted)
        vessel_match = label in extracted if label else False
        if label:
            results['vessel']['total'] += 1
            if vessel_match:
                results['vessel']['correct'] += 1

        # Port Information
        extracted = clean_text(row['extracted_port'])
        label = clean_text(row['label_port'])
        extracted = re.sub(r'laycan|eur\d+', '', extracted)
        port_match = True
        if label:
            label_ports = [p.strip() for p in label.split(',') if p.strip()]
            for port in label_ports:
                if port not in extracted:
                    port_match = False
                    break
        else:
            port_match = False

        if label:
            results['port']['total'] += 1
            if port_match:
                results['port']['correct'] += 1

        # Commodity
        extracted = clean_text(row['extracted_commodity'])
        label = clean_text(row['label_commodity'])
        commodity_match = label in extracted if label else False
        if label:
            results['commodity']['total'] += 1
            if commodity_match:
                results['commodity']['correct'] += 1

        # Incoterm
        extracted = clean_text(row['extracted_incoterm'])
        label = clean_text(row['label_incoterm'])
        extracted = extracted.replace("terms", "").strip()
        incoterm_match = label in extracted if label else False
        if label:
            results['incoterm']['total'] += 1
            if incoterm_match:
                results['incoterm']['correct'] += 1

    for entity in results:
        results[entity]["accuracy"] = round(results[entity]['correct'] / results[entity]['total'], 6) if \
        results[entity]['total'] > 0 else 0

    total_correct = sum(results[entity_type]['correct'] for entity_type in results)
    total_entities = sum(results[entity_type]['total'] for entity_type in results)
    overall_accuracy = total_correct / total_entities if total_entities > 0 else 0

    results['overall'] = {'accuracy': overall_accuracy, 'correct': total_correct, 'total': total_entities}
    return results


# -------------------------------------------------------------------
# Loading Specific Evaluation Dataset
# -------------------------------------------------------------------
def load_specific_eval_dataset(eval_dir, dataset_name):
    """Load a specific dataset by name"""
    eval_path = Path(eval_dir)
    dataset_file = eval_path / dataset_name

    if not dataset_file.exists():
        print(f"Warning: Dataset {dataset_name} not found in {eval_dir}")
        return None, None

    try:
        df = pd.read_json(dataset_file)
        eval_data = []
        all_entity_types = set()

        for _, row in df.iterrows():
            sentence = row['sentence']
            entities_data = row['entities']

            try:
                if isinstance(entities_data, str):
                    entities = ast.literal_eval(entities_data)
                else:
                    entities = entities_data
            except (SyntaxError, ValueError):
                entities = []

            ner_annotations = []
            for entity in entities:
                if 'name' in entity and 'type' in entity and 'pos' in entity:
                    entity_type = entity['type']
                    pos = entity['pos']
                    if isinstance(pos, list) and len(pos) == 2:
                        start, end = pos[0], pos[1]
                        all_entity_types.add(entity_type.lower())
                        ner_annotations.append([start, end, entity_type.lower()])
                elif 'start' in entity and 'end' in entity and 'type' in entity:
                    start = entity['start']
                    end = entity['end']
                    entity_type = entity['type']
                    all_entity_types.add(entity_type.lower())
                    ner_annotations.append([start, end, entity_type.lower()])

            eval_data.append({
                'text': sentence,
                'ner': ner_annotations
            })

        entity_types = sorted(list(all_entity_types))
        print(f"Entity types in {dataset_name}: {entity_types}")
        print(f"Dataset size: {len(eval_data)} examples")

        entity_counts = {}
        for example in eval_data:
            for _, _, etype in example['ner']:
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

        print(f"Total entities: {len(entity_counts)}")
        return eval_data, entity_types

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None, None


def get_available_datasets(eval_dir):
    """Get list of available evaluation datasets"""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return []
    return [f.name for f in eval_path.glob("*.json")]


# -------------------------------------------------------------------
# Model Evaluation
# -------------------------------------------------------------------
def evaluate_model(model,
                   tokenizer,
                   eval_data,
                   entity_types,
                   device,
                   sample_size=5000,
                   evaluation_threshold=0.5):
    """Evaluate model on specific dataset"""
    model.eval()

    if len(eval_data) > sample_size:
        eval_data = random.sample(eval_data, sample_size)
        print(f"Sampled {sample_size} examples for evaluation")

    all_predictions_bio = []
    all_labels_bio = []
    successful_predictions = 0
    total_examples = len(eval_data)

    for example_data in tqdm(eval_data, desc="OOD Evaluation"):
        text = example_data['text']
        gold_entities = example_data['ner']

        try:
            pred_entities = model.predict_entities(
                text=text,
                labels=entity_types,
                threshold=evaluation_threshold
            )

            text_length = len(text)

            current_gold_bio = ['O'] * text_length
            for start, end, entity_type in gold_entities:
                if 0 <= start <= end < text_length:
                    current_gold_bio[start] = f'B-{entity_type}'
                    for k in range(start + 1, end + 1):
                        if k < text_length:
                            current_gold_bio[k] = f'I-{entity_type}'
            all_labels_bio.append(current_gold_bio)

            current_pred_bio = ['O'] * text_length
            for entity in pred_entities:
                start, end = entity.get('start', 0), entity.get('end', 0)
                entity_type = entity.get('type', entity.get('label', "UNK")).lower()
                if 0 <= start <= end < text_length:
                    current_pred_bio[start] = f'B-{entity_type}'
                    for k in range(start + 1, end + 1):
                        if k < text_length:
                            current_pred_bio[k] = f'I-{entity_type}'
            all_predictions_bio.append(current_pred_bio)

            successful_predictions += 1

        except Exception as e:
            all_labels_bio.append(['O'] * len(text) if text else ['O'])
            all_predictions_bio.append(['O'] * len(text) if text else ['O'])

    print(f"Successfully evaluated {successful_predictions}/{total_examples} examples")

    try:
        f1 = f1_score(all_labels_bio, all_predictions_bio, zero_division="0")
        precision = precision_score(all_labels_bio, all_predictions_bio, zero_division="0")
        recall = recall_score(all_labels_bio, all_predictions_bio, zero_division="0")

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "span_f1": f1,
            "span_precision": precision,
            "span_recall": recall
        }

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "span_f1": 0.0, "span_precision": 0.0, "span_recall": 0.0}


# -------------------------------------------------------------------
# NERTrainer Class
# -------------------------------------------------------------------
class NERTrainer:
    def __init__(self,
                 base_trainer,
                 eval_dir,
                 entity_types=None,
                 eval_steps=50,
                 eval_sample_size=5000,
                 early_stopping_patience=10,
                 use_threshold_scheduler=False,
                 threshold_scheduler_config=None,
                 fixed_evaluation_threshold=0.5,
                 evaluation_strategy="combined"):

        self.base_trainer = base_trainer
        self.eval_dir = eval_dir
        self.entity_types = entity_types or ["PER", "ORG", "LOC", "MISC"]
        self.eval_steps = eval_steps
        self.eval_sample_size = eval_sample_size
        self.early_stopping_patience = early_stopping_patience
        self.evaluation_strategy = evaluation_strategy

        self.shipping_dataset_path = "../Masterthesis/syntheticdata/attrprompting/claude/aggregated/aggregated.json"

        self.dataset_best_f1 = {}
        self.dataset_no_improvement = {}
        self.available_datasets = get_available_datasets(eval_dir)
        self.current_dataset_index = 0

        self.global_best_f1 = 0.0
        self.best_f1 = 0.0
        self.steps_since_any_improvement = 0
        self.best_model_path = os.path.join(base_trainer.args.output_dir, "best_model")
        self.should_stop = False

        self.cached_datasets = {}
        self.load_all_datasets()

        self.results_log_path = os.path.join(base_trainer.args.output_dir, "evaluation_results.json")
        self.results_log = []

        self.fixed_evaluation_threshold = fixed_evaluation_threshold
        self.use_threshold_scheduler = use_threshold_scheduler
        if use_threshold_scheduler:
            scheduler_config = threshold_scheduler_config or {}
            self.threshold_scheduler = PredictionThresholdScheduler(**scheduler_config)
            print(
                f"Prediction threshold scheduler enabled: Initial={self.threshold_scheduler.initial_threshold}, Final={self.threshold_scheduler.final_threshold}")
        else:
            self.threshold_scheduler = None
            print(f"Using fixed threshold: {self.fixed_evaluation_threshold}")

        print(f"Evaluation strategy: {evaluation_strategy}")
        print(f"Available datasets: {self.available_datasets}")
        if self.evaluation_strategy == "all" and 'shipping_aggregated' in self.cached_datasets:
            print(f"Shipping dataset included: {self.shipping_dataset_path}")
        print(f"Early stopping: {early_stopping_patience} steps without ANY dataset improvement")

    def load_shipping_dataset(self, dataset_path):
        """Load the shipping/logistics dataset with special handling"""
        try:
            print(f"Loading shipping dataset from {dataset_path}...")
            df = pd.read_json(dataset_path)

            eval_data = []
            entity_labels = ["location", "vessel name", "shipping term", "commodity"]

            for idx, row in df.iterrows():
                if 'email_chain' in row:
                    text = "\n\n".join(email['body'] for email in row['email_chain'])
                elif 'concatenated_emails' in row:
                    text = row['concatenated_emails']
                else:
                    continue

                eval_data.append({
                    'text': text,
                    'ner': [],
                    'labels': row.get('labels', {}),
                    'is_shipping_dataset': True
                })

            print(f"Loaded {len(eval_data)} shipping examples")
            return eval_data, entity_labels

        except Exception as e:
            print(f"Error loading shipping dataset: {e}")
            return None, None

    def load_all_datasets(self):
        """Pre-load all datasets to avoid repeated I/O"""
        print("Loading all evaluation datasets...")

        for dataset_name in self.available_datasets:
            eval_data, entity_types = load_specific_eval_dataset(self.eval_dir, dataset_name)
            if eval_data is not None:
                self.cached_datasets[dataset_name] = (eval_data, entity_types)
                self.dataset_best_f1[dataset_name] = 0.0
                self.dataset_no_improvement[dataset_name] = 0

        if self.evaluation_strategy == "all" and os.path.exists(self.shipping_dataset_path):
            shipping_data, shipping_labels = self.load_shipping_dataset(self.shipping_dataset_path)
            if shipping_data is not None:
                self.cached_datasets['shipping_aggregated'] = (shipping_data, shipping_labels)
                self.dataset_best_f1['shipping_aggregated'] = 0.0
                self.dataset_no_improvement['shipping_aggregated'] = 0

        print(f"Loaded {len(self.cached_datasets)} datasets successfully")

        if self.evaluation_strategy == "combined":
            self._create_combined_dataset()

    def evaluate_shipping_dataset(self, eval_data, entity_types, evaluation_threshold):
        """Special evaluation for shipping dataset"""
        model = self.base_trainer.model
        model.eval()

        if len(eval_data) > self.eval_sample_size:
            eval_data = random.sample(eval_data, self.eval_sample_size)
            print(f"Sampled {self.eval_sample_size} shipping examples for evaluation")

        all_extractions = {}
        vessel_extractions = {}
        location_extractions = {}
        commodity_extractions = {}
        incoterm_extractions = {}

        for idx, example in tqdm(enumerate(eval_data), total=len(eval_data), desc="Shipping evaluation"):
            text = example['text']
            predictions = model.predict_entities(
                text=text,
                labels=entity_types,
                threshold=evaluation_threshold
            )

            all_extractions[idx] = predictions
            vessel_extractions[idx] = []
            location_extractions[idx] = []
            commodity_extractions[idx] = []
            incoterm_extractions[idx] = []

            for prediction in predictions:
                label = prediction.get('label', '')
                if label == 'vessel name':
                    vessel_extractions[idx].append(prediction)
                elif label == 'location':
                    location_extractions[idx].append(prediction)
                elif label == 'commodity':
                    commodity_extractions[idx].append(prediction)
                elif label == 'shipping term':
                    incoterm_extractions[idx].append(prediction)

        result_data = []
        for idx, example in enumerate(eval_data):
            labels = example.get('labels', {})

            vessel_dict = extract_entities({idx: vessel_extractions[idx]})
            port_dict = extract_entities({idx: location_extractions[idx]})
            commodity_dict = extract_entities({idx: commodity_extractions[idx]})
            incoterm_dict = extract_entities({idx: incoterm_extractions[idx]})

            result_data.append({
                'labels': labels,
                'extracted_vessel_name': vessel_dict.get(idx),
                'label_vessel_name': labels.get('vessel', '').lower() if labels else '',
                'extracted_port': port_dict.get(idx),
                'label_port': f"{labels.get('load_port', '').lower()}, {labels.get('discharge_port', '').lower()}" if labels else '',
                'extracted_commodity': commodity_dict.get(idx),
                'label_commodity': labels.get('commodity', '').lower() if labels else '',
                'extracted_incoterm': incoterm_dict.get(idx),
                'label_incoterm': labels.get('incoterm', '').lower() if labels else ''
            })

        result_df = pd.DataFrame(result_data)

        evaluation_results = evaluate_entity_extraction(result_df)

        f1 = evaluation_results['overall']['accuracy']

        total_correct = evaluation_results['overall']['correct']
        total_predicted = sum(len(all_extractions[i]) for i in range(len(eval_data)))
        total_actual = evaluation_results['overall']['total']

        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_actual if total_actual > 0 else 0

        print(f"\nShipping Dataset Detailed Results:")
        print(
            f"  Vessel accuracy: {evaluation_results['vessel']['accuracy']:.2%} ({evaluation_results['vessel']['correct']}/{evaluation_results['vessel']['total']})")
        print(
            f"  Port accuracy: {evaluation_results['port']['accuracy']:.2%} ({evaluation_results['port']['correct']}/{evaluation_results['port']['total']})")
        print(
            f"  Commodity accuracy: {evaluation_results['commodity']['accuracy']:.2%} ({evaluation_results['commodity']['correct']}/{evaluation_results['commodity']['total']})")
        print(
            f"  Incoterm accuracy: {evaluation_results['incoterm']['accuracy']:.2%} ({evaluation_results['incoterm']['correct']}/{evaluation_results['incoterm']['total']})")

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "span_f1": f1,
            "span_precision": precision,
            "span_recall": recall,
            "shipping_details": evaluation_results
        }

    def get_next_dataset(self):
        """Get next dataset based on evaluation strategy"""
        if not self.cached_datasets:
            return None, None, None

        if self.evaluation_strategy == "cycling":
            dataset_names = list(self.cached_datasets.keys())
            dataset_name = dataset_names[self.current_dataset_index % len(dataset_names)]
            self.current_dataset_index += 1

        elif self.evaluation_strategy == "random":
            dataset_name = random.choice(list(self.cached_datasets.keys()))

        elif self.evaluation_strategy == "all":
            return "all", None, None

        elif self.evaluation_strategy == "combined":
            eval_data, entity_types = self.cached_datasets["_combined_"]
            return "_combined_", eval_data, entity_types

        else:
            raise ValueError(f"Unknown evaluation strategy: {self.evaluation_strategy}")

        eval_data, entity_types = self.cached_datasets[dataset_name]
        return dataset_name, eval_data, entity_types

    def _create_combined_dataset(self):
        """Combine all datasets into one large evaluation set"""
        print("Creating combined dataset from all evaluation sets...")
        combined_data = []
        all_entity_types = set()

        for dataset_name, (eval_data, entity_types) in self.cached_datasets.items():
            if dataset_name != 'shipping_aggregated':  # Skip shipping dataset for combined
                combined_data.extend(eval_data)
                all_entity_types.update(entity_types)
                print(f"Added {len(eval_data)} examples from {dataset_name}")

        combined_entity_types = sorted(list(all_entity_types))

        self.cached_datasets["_combined_"] = (combined_data, combined_entity_types)
        self.dataset_best_f1["_combined_"] = 0.0
        self.dataset_no_improvement["_combined_"] = 0

        print(f"Combined dataset size: {len(combined_data)} examples")
        print(f"Combined entity types ({len(combined_entity_types)}): {combined_entity_types}")

        entity_counts = {}
        for example in combined_data:
            for _, _, etype in example['ner']:
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

        print(f"Entity distribution in combined dataset:")
        for etype, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {etype}: {count}")

    def evaluate_single_dataset(self, dataset_name, eval_data, entity_types, evaluation_threshold):
        """Evaluate on a single dataset with wandb logging"""
        current_step = self.base_trainer.state.global_step

        if dataset_name == 'shipping_aggregated' or (
                eval_data and len(eval_data) > 0 and eval_data[0].get('is_shipping_dataset', False)):
            metrics = self.evaluate_shipping_dataset(eval_data, entity_types, evaluation_threshold)
        else:
            metrics = evaluate_model(
                model=self.base_trainer.model,
                tokenizer=self.base_trainer.processing_class,
                eval_data=eval_data,
                entity_types=entity_types,
                device=self.base_trainer.args.device,
                sample_size=self.eval_sample_size,
                evaluation_threshold=evaluation_threshold
            )

        f1_value = metrics.get("span_f1", metrics.get("f1", 0.0))
        previous_best = self.dataset_best_f1[dataset_name]

        print(f"-F1: {f1_value:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"Dataset: {dataset_name} | Current: {f1_value:.4f} | Previous best: {previous_best:.4f}")

        wandb_metrics = {
            f"eval/{dataset_name}/f1": f1_value
        }

        if 'shipping_details' in metrics:
            shipping_details = metrics['shipping_details']
            wandb_metrics.update({
                f"eval/{dataset_name}/vessel_accuracy": shipping_details['vessel']['accuracy'],
                f"eval/{dataset_name}/port_accuracy": shipping_details['port']['accuracy'],
                f"eval/{dataset_name}/commodity_accuracy": shipping_details['commodity']['accuracy'],
                f"eval/{dataset_name}/incoterm_accuracy": shipping_details['incoterm']['accuracy'],
                f"eval/{dataset_name}/overall_accuracy": shipping_details['overall']['accuracy']
            })

        wandb.log(wandb_metrics, step=current_step)

        improved = False
        if f1_value > previous_best:
            improvement = f1_value - previous_best
            print(f"🎉 New best F1 for {dataset_name}: {f1_value:.4f} (+{improvement:.4f})")
            self.dataset_best_f1[dataset_name] = f1_value
            self.dataset_no_improvement[dataset_name] = 0
            improved = True

        else:
            self.dataset_no_improvement[dataset_name] += 1
            print(f"No improvement for {dataset_name}: {self.dataset_no_improvement[dataset_name]} evaluations")

        return f1_value, improved, metrics

    def evaluate(self, eval_dataset=None):
        """Main evaluation method with comprehensive wandb logging"""
        current_step = self.base_trainer.state.global_step

        evaluation_threshold = self.fixed_evaluation_threshold
        if self.threshold_scheduler:
            evaluation_threshold = self.threshold_scheduler.get_threshold(current_step)
            print(f"Current evaluation threshold: {evaluation_threshold:.3f}")

        any_improvement = False
        step_results = {
            "step": current_step,
            "threshold": evaluation_threshold,
            "results": {}
        }

        if self.evaluation_strategy == "all":
            total_f1 = 0.0
            num_datasets = 0
            dataset_f1s = {}

            for dataset_name in self.cached_datasets:
                if dataset_name == "_combined_":
                    continue

                eval_data, entity_types = self.cached_datasets[dataset_name]
                print(f"\nEvaluating on dataset: {dataset_name}")

                f1_value, improved, metrics = self.evaluate_single_dataset(
                    dataset_name, eval_data, entity_types, evaluation_threshold
                )

                step_results["results"][dataset_name] = {
                    "f1": f1_value,
                    "improved": improved,
                    "metrics": metrics
                }

                dataset_f1s[dataset_name] = f1_value

                if improved:
                    any_improvement = True

                total_f1 += f1_value
                num_datasets += 1

            avg_f1 = total_f1 / num_datasets if num_datasets > 0 else 0.0
            print(f"\nAverage F1 across all datasets: {avg_f1:.4f}")
            step_results["average_f1"] = avg_f1

            summary_metrics = {
                "eval/summary/average_f1": avg_f1,
                "eval/summary/global_best_f1": self.global_best_f1,
                "eval/summary/num_datasets": num_datasets,
                "eval/summary/steps_since_improvement": self.steps_since_any_improvement,
                "eval/summary/any_improvement": any_improvement
            }

            for dataset_name, best_f1 in self.dataset_best_f1.items():
                if dataset_name != "_combined_":
                    summary_metrics[f"eval/summary/best_f1/{dataset_name}"] = best_f1

            wandb.log(summary_metrics, step=current_step)

        else:
            dataset_name, eval_data, entity_types = self.get_next_dataset()

            if dataset_name is None:
                print("No datasets available for evaluation")
                return {"f1": 0.0}

            print(f"\nEvaluating on dataset: {dataset_name}")
            f1_value, improved, metrics = self.evaluate_single_dataset(
                dataset_name, eval_data, entity_types, evaluation_threshold
            )

            step_results["results"][dataset_name] = {
                "f1": f1_value,
                "improved": improved,
                "metrics": metrics
            }

            any_improvement = improved

        if any_improvement:
            self.steps_since_any_improvement = 0

            print(f"Saving model due to improvement - saving to {self.best_model_path}")
            self.base_trainer.save_model(self.best_model_path)
            self.base_trainer.processing_class.save_pretrained(self.best_model_path)

            if self.evaluation_strategy == "all":
                current_max_f1 = max(self.dataset_best_f1.values())
            else:
                current_max_f1 = max(f1_value, self.global_best_f1)

            if current_max_f1 > self.global_best_f1:
                old_best = self.global_best_f1
                self.global_best_f1 = current_max_f1

        else:
            self.steps_since_any_improvement += 1
            print(f"No improvement on any dataset: {self.steps_since_any_improvement}/{self.early_stopping_patience}")

        self.results_log.append(step_results)
        self.save_results_log()

        if self.steps_since_any_improvement >= self.early_stopping_patience:
            print(f"Early stopping triggered! No improvement for {self.early_stopping_patience} evaluations")
            self.should_stop = True

        print(f"\n=== OOD Evaluation Scores ===")
        print(f"Best F1 per dataset: {dict(sorted(self.dataset_best_f1.items(), key=lambda x: x[1], reverse=True))}")
        if self.steps_since_any_improvement > 0:
            print(f"Steps since any improvement: {self.steps_since_any_improvement}")

        if self.evaluation_strategy == "all":
            return {"f1": step_results.get("average_f1", 0.0)}
        else:
            return step_results["results"][dataset_name]["metrics"]

    def save_results_log(self):
        """Save evaluation results log"""
        with open(self.results_log_path, 'w') as f:
            json.dump(self.results_log, f, indent=2)

    def train(self):
        """Train with periodic evaluation on external datasets"""

        class EvalCallback(TrainerCallback):
            def __init__(self, ner_trainer):
                self.ner_trainer = ner_trainer

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % self.ner_trainer.eval_steps == 0 and state.global_step > 0:
                    self.ner_trainer.evaluate()

                    if self.ner_trainer.should_stop:
                        control.should_training_stop = True

        self.base_trainer.add_callback(EvalCallback(self))
        result = self.base_trainer.train()

        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            best_model = GLiNER.from_pretrained(self.best_model_path)
            self.base_trainer.model = best_model

        print(f"\n=== Final Training Summary ===")
        print(f"Final best F1 per dataset:")
        for dataset, f1 in sorted(self.dataset_best_f1.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dataset}: {f1:.4f}")

        return result