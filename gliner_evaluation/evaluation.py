import re
import torch
import warnings
import requests
import pandas as pd
import json
from pathlib import Path
from tqdm.auto import tqdm
from gliner import GLiNER
from huggingface_hub import configure_http_backend
from datetime import datetime
import traceback

# -------------------------------------------------------------------
# Warning Silencing and Setup
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# SSL Backend Configuration
# -------------------------------------------------------------------
def backend_factory() -> requests.Session:
    """Configure SSL backend to handle certificate issues"""
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)

# -------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------
MODELS = {

    # -------------------------------------------------------------------
    # BASE
    # -------------------------------------------------------------------
    "gliner_base": "urchade/gliner_base",

    "gliner_finetuned": "../models/GLiNER/finetuned/normal_finetune_gliner_base",

    "gliner_base_entity_aware": "../models/GLiNER/membank/entity_aware_gliner_base",

    "gliner_finetuned_email": "../models/GLiNER/email/gliner_base_email",

    # -------------------------------------------------------------------
    # SBERT
    # -------------------------------------------------------------------
    "gliner_sbert": "../models/GLiNER/base/sbert",

    "gliner_finetued_sbert": "../models/GLiNER/finetuned/normal_finetune_gliner_sbert",

    "gliner_sbert_entity_aware": "../models/GLiNER/membank/entity_aware_gliner_sbert",

    "gliner_sbert_finetuned_email": "../models/GLiNER/email/gliner_sbert_email",

    # -------------------------------------------------------------------
    # BGE
    # -------------------------------------------------------------------
    "gliner_bge": "../models/GLiNER/base/bge",

    "gliner_finetuned_bge": "../models/GLiNER/finetuned/normal_finetune_gliner_bge",

    "gliner_bge_entity_aware": "../models/GLiNER/membank/entity_aware_bge",

    "gliner_finetuned_bge_email": "../models/GLiNER/email/gliner_bge_email",

    # -------------------------------------------------------------------
    # SimCSE
    # -------------------------------------------------------------------
    "gliner_simcse": "../models/GLiNER/base/simcse",

    "gliner_finetuned_simcse": "../models/GLiNER/finetuned/normal_finetune_gliner_simcse",

    "gliner_simcse_entity_aware": "../models/GLiNER/membank/entity_aware_gliner_simcse",

    "gliner_finetuned_simcse_email": "../models/GLiNER/email/gliner_simcse_email",

    # -------------------------------------------------------------------
    # MLM
    # -------------------------------------------------------------------
    "gliner_mlm": "../models/GLiNER/base/mlm",
    "gliner_finetuned_mlm": "../models/GLiNER/finetuned/normal_finetune_gliner_mlm",
    "gliner_mlm_entity_aware": "../models/GLiNER/membank/entity_aware_gliner_mlm",
    "gliner_finetuned_mlm_email": "../models/GLiNER/email/gliner_mlm_email",
}


# -------------------------------------------------------------------
# Dataset Configuration
# -------------------------------------------------------------------
DATASETS = {
    # ------------------------------------------------------------------
    # Attribute Prompting
    # ------------------------------------------------------------------
    "attr_prompting_claude": "../data/email_datasets/synthetic/attrprompting/claude/aggregated/aggregated.json",
    "attr_prompting_deepseek": "../data/email_datasets/synthetic/attrprompting/deepseek/aggregated/aggregated.json",
    "attr_prompting_gemini": "../data/email_datasets/synthetic/attrprompting/gemini/aggregated/aggregated.json",
    "attr_prompting_gpt4": "../data/email_datasets/synthetic/attrprompting/gpt-4-turbo/aggregated/aggregated.json",
    "attr_prompting_mistral": "../data/email_datasets/synthetic/attrprompting/mistral/aggregated/aggregated.json",

    # ------------------------------------------------------------------
    # BARE Llama3B
    # ------------------------------------------------------------------
    "bare_llama_3b_claude": "../data/email_datasets/synthetic/baserefine/refine/llama3b/claude/aggregated/aggregated.json",
    "bare_llama_3b_deepseek": "../data/email_datasets/synthetic/baserefine/refine/llama3b/deepseek/aggregated/aggregated.json",
    "bare_llama_3b_gemini": "../data/email_datasets/synthetic/baserefine/refine/llama3b/gemini/aggregated/aggregated.json",
    "bare_llama_3b_gpt4": "../data/email_datasets/synthetic/baserefine/refine/llama3b/gpt-4-turbo/aggregated/aggregated.json",
    "bare_llama_3b_mistral": "../data/email_datasets/synthetic/baserefine/refine/llama3b/mistral/aggregated/aggregated.json",

    # ------------------------------------------------------------------
    # BARE Llama8B
    # ------------------------------------------------------------------
    "bare_llama_8b_claude": "../data/email_datasets/synthetic/baserefine/refine/llama8b/claude/aggregated/aggregated.json",
    "bare_llama_8b_deepseek": "../data/email_datasets/synthetic/baserefine/refine/llama8b/deepseek/aggregated/aggregated.json",
    "bare_llama_8b_gemini": "../data/email_datasets/synthetic/baserefine/refine/llama8b/gemini/aggregated/aggregated.json",
    "bare_llama_8b_gpt4": "../data/email_datasets/synthetic/baserefine/refine/llama8b/gpt-4-turbo/aggregated/aggregated.json",
    "bare_llama_8b_mistral": "../data/email_datasets/synthetic/baserefine/refine/llama8b/mistral/aggregated/aggregated.json",

    # ------------------------------------------------------------------
    # Few Shot
    # ------------------------------------------------------------------
    "fewshot_claude": "../data/email_datasets/synthetic/fewshot/claude/aggregated/aggregated.json",
    "fewshot_deepseek": "../data/email_datasets/synthetic/fewshot/deepseek/aggregated/aggregated.json",
    "fewshot_gemini": "../data/email_datasets/synthetic/fewshot/gemini/aggregated/aggregated.json",
    "fewshot_gpt4": "../data/email_datasets/synthetic/fewshot/gpt-4-turbo/aggregated/aggregated.json",
    "fewshot_mistral": "../data/email_datasets/synthetic/fewshot/mistral/aggregated/aggregated.json",

    # ------------------------------------------------------------------
    # Zero Shot
    # ------------------------------------------------------------------
    "zeroshot_claude": "../data/email_datasets/synthetic/zeroshot/claude/aggregated/aggregated.json",
    "zeroshot_deepseek": "../data/email_datasets/synthetic/zeroshot/deepseek/aggregated/aggregated.json",
    "zeroshot_gemini": "../data/email_datasets/synthetic/zeroshot/gemini/aggregated/aggregated.json",
    "zeroshot_gpt4": "../data/email_datasets/synthetic/zeroshot/gpt-4-turbo/aggregated/aggregated.json",
    "zeroshot_mistral": "../data/email_datasets/synthetic/zeroshot/mistral/aggregated/aggregated.json",
}

# -------------------------------------------------------------------
# Entity Labels for Extraction
# -------------------------------------------------------------------
ENTITY_LABELS = ["location", "vessel name", "incoterm", "commodity"]


# -------------------------------------------------------------------
# Helper Functions
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
            results[idx] = entity_texts
        else:
            results[idx] = []
    return results


def safe_get_label(label_data, key, default=""):
    """Safely extract a label value from label data, handling missing keys and list values"""
    try:
        if label_data is None or not isinstance(label_data, dict):
            return default

        value = label_data.get(key, default)

        if isinstance(value, list):
            return default

        if isinstance(value, str):
            return value.lower()

        return str(value).lower() if value is not None else default

    except Exception:
        return default


def calculate_metrics(extracted_list, label_list):
    """Calculate precision, recall, and F1 for entity lists using substring matching"""
    if not label_list and not extracted_list:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}

    if not label_list:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(extracted_list), 'fn': 0}

    if not extracted_list:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(label_list)}

    tp = 0
    matched_labels = set()

    for extracted in extracted_list:
        found_match = False
        for i, label in enumerate(label_list):
            if i not in matched_labels and (label in extracted or extracted in label):
                tp += 1
                matched_labels.add(i)
                found_match = True
                break

    fp = len(extracted_list) - tp
    fn = len(label_list) - len(matched_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def evaluate_entity_extraction(result_df):
    """Evaluate precision, recall, and F1 for entity extraction"""
    results = {
        'vessel': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'port': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'commodity': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'incoterm': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0}
    }

    for idx, row in result_df.iterrows():
        extracted = row["extracted_vessel_name"]
        if isinstance(extracted, str):
            extracted = [extracted] if extracted else []
        extracted = [re.sub(r'mv\s+|m/v\s+', '', clean_text(e)) for e in extracted if e]

        label = clean_text(row["label_vessel_name"])
        label = re.sub(r'mv\s+|m/v\s+', '', label)
        label_list = [label] if label else []

        metrics = calculate_metrics(extracted, label_list)
        results['vessel']['precision'].append(metrics['precision'])
        results['vessel']['recall'].append(metrics['recall'])
        results['vessel']['f1'].append(metrics['f1'])
        results['vessel']['tp'] += metrics['tp']
        results['vessel']['fp'] += metrics['fp']
        results['vessel']['fn'] += metrics['fn']

        extracted = row['extracted_port']
        if isinstance(extracted, str):
            extracted = [e.strip() for e in extracted.split(',') if e.strip()]
        extracted = [re.sub(r'laycan|eur\d+', '', clean_text(e)) for e in extracted if e]

        label = clean_text(row['label_port'])
        label_list = [p.strip() for p in label.split(',') if p.strip()]

        metrics = calculate_metrics(extracted, label_list)
        results['port']['precision'].append(metrics['precision'])
        results['port']['recall'].append(metrics['recall'])
        results['port']['f1'].append(metrics['f1'])
        results['port']['tp'] += metrics['tp']
        results['port']['fp'] += metrics['fp']
        results['port']['fn'] += metrics['fn']

        extracted = row['extracted_commodity']
        if isinstance(extracted, str):
            extracted = [extracted] if extracted else []
        extracted = [clean_text(e) for e in extracted if e]

        label = clean_text(row['label_commodity'])
        label_list = [label] if label else []

        metrics = calculate_metrics(extracted, label_list)
        results['commodity']['precision'].append(metrics['precision'])
        results['commodity']['recall'].append(metrics['recall'])
        results['commodity']['f1'].append(metrics['f1'])
        results['commodity']['tp'] += metrics['tp']
        results['commodity']['fp'] += metrics['fp']
        results['commodity']['fn'] += metrics['fn']

        extracted = row['extracted_incoterm']
        if isinstance(extracted, str):
            extracted = [extracted] if extracted else []
        extracted = [clean_text(e).replace("terms", "").strip() for e in extracted if e]

        label = clean_text(row['label_incoterm'])
        label_list = [label] if label else []

        metrics = calculate_metrics(extracted, label_list)
        results['incoterm']['precision'].append(metrics['precision'])
        results['incoterm']['recall'].append(metrics['recall'])
        results['incoterm']['f1'].append(metrics['f1'])
        results['incoterm']['tp'] += metrics['tp']
        results['incoterm']['fp'] += metrics['fp']
        results['incoterm']['fn'] += metrics['fn']

    for entity in results:
        n = len(results[entity]['precision'])
        results[entity]['avg_precision'] = round(sum(results[entity]['precision']) / n, 4) if n > 0 else 0
        results[entity]['avg_recall'] = round(sum(results[entity]['recall']) / n, 4) if n > 0 else 0
        results[entity]['avg_f1'] = round(sum(results[entity]['f1']) / n, 4) if n > 0 else 0

        tp = results[entity]['tp']
        fp = results[entity]['fp']
        fn = results[entity]['fn']

        results[entity]['micro_precision'] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
        results[entity]['micro_recall'] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
        results[entity]['micro_f1'] = round(2 * results[entity]['micro_precision'] * results[entity]['micro_recall'] /
                                            (results[entity]['micro_precision'] + results[entity]['micro_recall']), 4) \
            if (results[entity]['micro_precision'] + results[entity]['micro_recall']) > 0 else 0

    overall_precision = sum(results[entity]['avg_precision'] for entity in results) / len(results)
    overall_recall = sum(results[entity]['avg_recall'] for entity in results) / len(results)
    overall_f1 = sum(results[entity]['avg_f1'] for entity in results) / len(results)

    total_tp = sum(results[entity]['tp'] for entity in results)
    total_fp = sum(results[entity]['fp'] for entity in results)
    total_fn = sum(results[entity]['fn'] for entity in results)

    overall_micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_micro_f1 = 2 * overall_micro_precision * overall_micro_recall / (
                overall_micro_precision + overall_micro_recall) \
        if (overall_micro_precision + overall_micro_recall) > 0 else 0

    results['overall'] = {
        'precision': round(overall_precision, 4),
        'recall': round(overall_recall, 4),
        'f1': round(overall_f1, 4),
        'micro_precision': round(overall_micro_precision, 4),
        'micro_recall': round(overall_micro_recall, 4),
        'micro_f1': round(overall_micro_f1, 4),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

    return results


def evaluate_model_on_dataset(model, dataset_path, device):
    """Evaluate a single model on a single dataset"""
    try:
        df = pd.read_json(dataset_path)
        df['concatenated_emails'] = df.email_chain.apply(
            lambda email_list: "\n\n".join(email['body'] for email in email_list))

        all_extractions = {}
        for index, email_text in enumerate(df.concatenated_emails):
            predictions = model.predict_entities(
                text=email_text,
                labels=ENTITY_LABELS,
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

        label_column = None
        if 'labels' in result_df.columns:
            label_column = 'labels'
        elif 'label' in result_df.columns:
            label_column = 'label'
        else:
            print(f"WARNING: No 'labels' or 'label' column found in dataset. Using empty labels.")

        vessel_dict = extract_entities(vessel_extractions)
        result_df["extracted_vessel_name"] = [vessel_dict.get(i, []) for i in range(len(df))]

        if label_column:
            result_df["label_vessel_name"] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'vessel', "")
            )
        else:
            result_df["label_vessel_name"] = ""

        port_dict = extract_entities(location_extractions)
        result_df["extracted_port"] = [port_dict.get(i, []) for i in range(len(df))]

        if label_column:
            result_df["label_port"] = result_df[label_column].apply(
                lambda
                    label_data: f"{safe_get_label(label_data, 'load_port', '')}, {safe_get_label(label_data, 'discharge_port', '')}"
                if isinstance(label_data, dict) else ""
            )
        else:
            result_df["label_port"] = ""

        commodity_dict = extract_entities(commodity_extractions)
        result_df["extracted_commodity"] = [commodity_dict.get(i, []) for i in range(len(df))]

        if label_column:
            result_df['label_commodity'] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'commodity', "").replace("soybeans", "soybean")
            )
        else:
            result_df['label_commodity'] = ""

        incoterm_dict = extract_entities(incoterm_extractions)
        result_df["extracted_incoterm"] = [incoterm_dict.get(i, []) for i in range(len(df))]

        if label_column:
            result_df['label_incoterm'] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'incoterm', "")
            )
        else:
            result_df['label_incoterm'] = ""

        # Evaluate performance
        evaluation_results = evaluate_entity_extraction(result_df)
        return evaluation_results

    except Exception as e:
        print(f"Error evaluating dataset: {e}")
        traceback.print_exc()
        return None


# -------------------------------------------------------------------
# Main Evaluation Function
# -------------------------------------------------------------------
def run_comprehensive_evaluation():
    """Run evaluation for all model-dataset combinations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_results = []

    total_combinations = len(MODELS) * len(DATASETS)
    current = 0

    print(f"Total model-dataset combinations to evaluate: {total_combinations}")
    print("=" * 80)

    for model_name, model_path in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Loading model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'=' * 60}")

        try:
            model = GLiNER.from_pretrained(model_path)
            model = model.to(device)
            num_params = sum(p.numel() for p in model.model.parameters())
            print(f"Model loaded successfully with {num_params:,} parameters")

            for dataset_name, dataset_path in DATASETS.items():
                current += 1
                print(f"\n[{current}/{total_combinations}] Evaluating {model_name} on {dataset_name}...")

                if not Path(dataset_path).exists():
                    print(f"WARNING: Dataset file not found: {dataset_path}")
                    continue

                results = evaluate_model_on_dataset(model, dataset_path, device)

                if results:
                    result_row = {
                        'model': model_name,
                        'dataset': dataset_name,

                        # Vessel metrics
                        'vessel_precision': results['vessel']['avg_precision'],
                        'vessel_recall': results['vessel']['avg_recall'],
                        'vessel_f1': results['vessel']['avg_f1'],
                        'vessel_micro_precision': results['vessel']['micro_precision'],
                        'vessel_micro_recall': results['vessel']['micro_recall'],
                        'vessel_micro_f1': results['vessel']['micro_f1'],

                        # Port metrics
                        'port_precision': results['port']['avg_precision'],
                        'port_recall': results['port']['avg_recall'],
                        'port_f1': results['port']['avg_f1'],
                        'port_micro_precision': results['port']['micro_precision'],
                        'port_micro_recall': results['port']['micro_recall'],
                        'port_micro_f1': results['port']['micro_f1'],

                        # Commodity metrics
                        'commodity_precision': results['commodity']['avg_precision'],
                        'commodity_recall': results['commodity']['avg_recall'],
                        'commodity_f1': results['commodity']['avg_f1'],
                        'commodity_micro_precision': results['commodity']['micro_precision'],
                        'commodity_micro_recall': results['commodity']['micro_recall'],
                        'commodity_micro_f1': results['commodity']['micro_f1'],

                        # Incoterm metrics
                        'incoterm_precision': results['incoterm']['avg_precision'],
                        'incoterm_recall': results['incoterm']['avg_recall'],
                        'incoterm_f1': results['incoterm']['avg_f1'],
                        'incoterm_micro_precision': results['incoterm']['micro_precision'],
                        'incoterm_micro_recall': results['incoterm']['micro_recall'],
                        'incoterm_micro_f1': results['incoterm']['micro_f1'],

                        # Overall metrics
                        'overall_precision': results['overall']['precision'],
                        'overall_recall': results['overall']['recall'],
                        'overall_f1': results['overall']['f1'],
                        'overall_micro_precision': results['overall']['micro_precision'],
                        'overall_micro_recall': results['overall']['micro_recall'],
                        'overall_micro_f1': results['overall']['micro_f1'],

                        # Counts
                        'total_tp': results['overall']['tp'],
                        'total_fp': results['overall']['fp'],
                        'total_fn': results['overall']['fn']
                    }
                    all_results.append(result_row)

                    # Summary Prints
                    print(
                        f"  Overall Macro F1: {results['overall']['f1']:.2%} (P: {results['overall']['precision']:.2%}, R: {results['overall']['recall']:.2%})")
                    print(
                        f"  Overall Micro F1: {results['overall']['micro_f1']:.2%} (P: {results['overall']['micro_precision']:.2%}, R: {results['overall']['micro_recall']:.2%})")
                    print(f"  Entity-level F1 scores:")
                    print(
                        f"    - Vessel: {results['vessel']['avg_f1']:.2%} (P: {results['vessel']['avg_precision']:.2%}, R: {results['vessel']['avg_recall']:.2%})")
                    print(
                        f"    - Port: {results['port']['avg_f1']:.2%} (P: {results['port']['avg_precision']:.2%}, R: {results['port']['avg_recall']:.2%})")
                    print(
                        f"    - Commodity: {results['commodity']['avg_f1']:.2%} (P: {results['commodity']['avg_precision']:.2%}, R: {results['commodity']['avg_recall']:.2%})")
                    print(
                        f"    - Incoterm: {results['incoterm']['avg_f1']:.2%} (P: {results['incoterm']['avg_precision']:.2%}, R: {results['incoterm']['avg_recall']:.2%})")
                else:
                    print(f"  ERROR: Evaluation failed for this combination")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR loading model {model_name}: {e}")
            traceback.print_exc()
            continue

    # Result Saving to .csv
    if all_results:
        results_df = pd.DataFrame(all_results)

        results_df = results_df.sort_values(['model', 'dataset'])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"gliner_evaluation_f1_results_{timestamp}.csv"

        results_df.to_csv(output_file, index=False)
        print(f"\n{'=' * 80}")
        print(f"Evaluation complete! Results saved to: {output_file}")
        print(f"Total successful evaluations: {len(all_results)}/{total_combinations}")

        print(f"\n{'=' * 40}")
        print("SUMMARY BY MODEL (Macro F1):")
        print(f"{'=' * 40}")
        model_summary = results_df.groupby('model')[['overall_f1', 'overall_precision', 'overall_recall']].agg(
            ['mean', 'std'])
        print(model_summary.round(4))

        print(f"\n{'=' * 40}")
        print("SUMMARY BY MODEL (Micro F1):")
        print(f"{'=' * 40}")
        model_summary_micro = results_df.groupby('model')[
            ['overall_micro_f1', 'overall_micro_precision', 'overall_micro_recall']].agg(['mean', 'std'])
        print(model_summary_micro.round(4))

        print(f"\n{'=' * 40}")
        print("SUMMARY BY DATASET TYPE:")
        print(f"{'=' * 40}")
        results_df['dataset_type'] = results_df['dataset'].apply(lambda x: x.split('_')[0])
        dataset_summary = results_df.groupby('dataset_type')[['overall_f1', 'overall_micro_f1']].agg(['mean', 'std'])
        print(dataset_summary.round(4))

        print(f"\n{'=' * 40}")
        print("TOP 5 MODELS BY MACRO F1:")
        print(f"{'=' * 40}")
        top_models = results_df.groupby('model')['overall_f1'].mean().sort_values(ascending=False).head(5)
        for i, (model, f1) in enumerate(top_models.items()):
            avg_p = results_df[results_df['model'] == model]['overall_precision'].mean()
            avg_r = results_df[results_df['model'] == model]['overall_recall'].mean()
            print(f"{i + 1}. {model}: F1={f1:.4f} (P={avg_p:.4f}, R={avg_r:.4f})")

        print(f"\n{'=' * 40}")
        print("ENTITY-SPECIFIC BEST PERFORMERS:")
        print(f"{'=' * 40}")
        for entity in ['vessel', 'port', 'commodity', 'incoterm']:
            best_model = results_df.groupby('model')[f'{entity}_f1'].mean().idxmax()
            best_score = results_df.groupby('model')[f'{entity}_f1'].mean().max()
            print(f"{entity.capitalize()}: {best_model} (F1={best_score:.4f})")

    else:
        print("\nERROR: No results were collected. Please check the model and dataset paths.")

# -------------------------------------------------------------------
# Exec
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting comprehensive GLiNER model evaluation with F1, Precision, and Recall...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_comprehensive_evaluation()