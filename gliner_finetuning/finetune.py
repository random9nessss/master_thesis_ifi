import os
import re
import json
import torch
import warnings
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path

import requests
from huggingface_hub import configure_http_backend
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class Config:
    """Configuration class for GLiNER fine-tuning."""

    BASE_MODEL = r"..."
    OUTPUT_DIR = r"/models"
    CUSTOM_DATASET_PATH = "../data/email_datasets/synthetic/train_email_synthetic.json"
    AGGREGATED_EVAL_PATH = "../data/email_datasets/email_datasets/synthetic/attrprompting/claude/aggregated/aggregated.json"

    ENTITY_LABELS = ["location", "vessel name", "incoterm", "commodity"]

    TEST_SIZE = 0.1
    RANDOM_SEED = 42

    TRAINING_CONFIG = {
        "name": "gliner_base_finetuned_",
        "learning_rate": 2e-6,
        "others_lr": 5e-6,
        "weight_decay": 0.01,
        "others_weight_decay": 0.01,
        "batch_size": 6,
        "gradient_accumulation_steps": 4,
        "epochs": 4,
        "warmup_ratio": 0.08
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
# Data Processing
# -------------------------------------------------------------------
def process_custom_entities(custom_data: List[Dict]) -> List[Dict]:
    """
    Process entities from custom dataset format to GLiNER format.

    Args:
        custom_data: List of examples in custom format

    Returns:
        List of processed examples with tokenized text and NER spans
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
    """
    Remove examples with no NER labels or empty text.

    Args:
        dataset: List of examples to filter

    Returns:
        Filtered list of examples
    """
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


def evaluate_on_aggregated_dataset(model: GLiNER, config: Config) -> Dict[str, Any]:
    """
    Evaluate model on the aggregated dataset.

    Args:
        model: The GLiNER model to evaluate
        config: Configuration object

    Returns:
        Dictionary with evaluation results
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
# Custom Trainer with Aggregated Evaluation
# -------------------------------------------------------------------
class CustomTrainer(Trainer):
    """Custom trainer that includes evaluation on aggregated dataset"""

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.best_aggregated_accuracy = 0.0

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


# ============================================================================
# Data Loading
# ============================================================================

def load_and_prepare_data(config: Config) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and prepare training and test datasets.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_data, test_data)
    """
    print(f"Loading dataset from {config.CUSTOM_DATASET_PATH}...")

    with open(config.CUSTOM_DATASET_PATH, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)

    print("Processing dataset...")
    processed_data = process_custom_entities(custom_data)
    print(f"Dataset size after processing: {len(processed_data)}")

    processed_data = filter_empty_ner_examples(processed_data)

    train_data, test_data = train_test_split(
        processed_data,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    return train_data, test_data


# -------------------------------------------------------------------
# Data Processing
# -------------------------------------------------------------------
def create_training_arguments(config: Dict, data_size: int, output_dir: str) -> TrainingArguments:
    """
    Create training arguments for the trainer.

    Args:
        config: Training configuration dictionary
        data_size: Size of training dataset
        output_dir: Output directory for model checkpoints

    Returns:
        TrainingArguments object
    """
    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]

    steps_per_epoch = max(1, data_size // (batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * config["epochs"]

    eval_steps = max(1, steps_per_epoch // 5)
    save_steps = steps_per_epoch

    return TrainingArguments(
        output_dir=output_dir,

        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        others_lr=config["others_lr"],
        others_weight_decay=config["others_weight_decay"],

        lr_scheduler_type="cosine",
        warmup_ratio=config["warmup_ratio"],

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,

        max_steps=total_steps,

        eval_strategy="steps",
        eval_steps=eval_steps,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,

        fp16=torch.cuda.is_available(),

        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=100,
        report_to="none",

        max_grad_norm=1,
    )


def train_model(config: Config, train_dataset: List[Dict], eval_dataset: List[Dict]) -> str:
    """
    Train the GLiNER model with the specified configuration.

    Args:
        config: Configuration object
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset

    Returns:
        Path to the saved final model
    """
    print("\n" + "=" * 80)
    print("TRAINING GLINER MODEL")
    print("=" * 80)

    model = GLiNER.from_pretrained(config.BASE_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )

    model_output_dir = os.path.join(config.OUTPUT_DIR, "trained_model")
    os.makedirs(model_output_dir, exist_ok=True)

    training_args = create_training_arguments(
        config.TRAINING_CONFIG,
        len(train_dataset),
        model_output_dir
    )

    trainer = CustomTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    print(f"Training model: {config.TRAINING_CONFIG['name']}")
    trainer.train()

    eval_result = trainer.evaluate()
    print(f"Final evaluation results: {eval_result}")

    final_model_path = os.path.join(config.OUTPUT_DIR, "custom_gliner_model")
    model.save_pretrained(final_model_path)
    print(f"Model saved to: {final_model_path}")

    print("\n" + "=" * 80)
    print("FINAL AGGREGATED DATASET EVALUATION")
    print("=" * 80)
    final_aggregated_results, final_result_df = evaluate_on_aggregated_dataset(model, config)

    if final_aggregated_results:
        final_output_path = os.path.join(config.OUTPUT_DIR, "final_aggregated_eval_results.csv")
        final_result_df.to_csv(final_output_path, index=False)
        print(f"Final aggregated evaluation results saved to {final_output_path}")

    return final_model_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    setup_ssl_backend()
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_data, test_data = load_and_prepare_data(config)

    final_model_path = train_model(config, train_data, test_data)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final model saved at: {final_model_path}")
    print(f"Load with: GLiNER.from_pretrained('{final_model_path}')")


if __name__ == "__main__":
    main()