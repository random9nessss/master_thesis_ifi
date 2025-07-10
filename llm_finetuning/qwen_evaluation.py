import os
import re
import torch
import warnings
import requests
import pandas as pd
import json
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import configure_http_backend
from datetime import datetime
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
    "qwen_base": "../models/Qwen/base",
    "qwen_email": "../models/Qwen/qwen_emails",
    "qwen_finetuned": "../models/Qwen/qwen_no_email"
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
# Prompt Template
# -------------------------------------------------------------------
def get_role_based_prompt(sentence: str) -> str:
    """Generates the direct role-based prompt for extraction."""
    return f"""You are a shipping data extractor. Extract shipping information from the text.

                Text: {sentence}

                Return a JSON object with:
                - vessel (ship name)
                - commodity (cargo type)
                - incoterm (trade term like CIF, FOB)
                - locations (list of places)

                JSON:
            """


# -------------------------------------------------------------------
# Model Loading Functions
# -------------------------------------------------------------------
def load_qwen_model(model_path, device):
    """Load Qwen model - either base or PEFT fine-tuned"""
    print(f"Loading model from: {model_path}")

    is_peft_model = any(Path(model_path).glob("adapter_*.bin")) or Path(model_path, "adapter_config.json").exists()

    if is_peft_model:
        print("Detected PEFT model, loading with adapters...")
        base_model_path = "llm_finetuning/qwen"

        print(f"Loading base model from: {base_model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True
        )

        model = PeftModel.from_pretrained(model, model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

    if not torch.cuda.is_available():
        model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# -------------------------------------------------------------------
# Prediction Functions
# -------------------------------------------------------------------
def extract_json_from_response(response_text):
    """Extract JSON object from model response"""
    try:
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
    except:
        pass

    try:
        result = {
            "vessel": None,
            "commodity": None,
            "incoterm": None,
            "locations": []
        }

        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            # Vessel Matching
            if 'vessel' in line.lower() and ':' in line:
                vessel_match = re.search(r':\s*"?([^",\n]+)"?', line)
                if vessel_match:
                    result['vessel'] = vessel_match.group(1).strip()

            # Commodity Matching
            elif 'commodity' in line.lower() and ':' in line:
                commodity_match = re.search(r':\s*"?([^",\n]+)"?', line)
                if commodity_match:
                    result['commodity'] = commodity_match.group(1).strip()

            # Incoterm Matching
            elif 'incoterm' in line.lower() and ':' in line:
                incoterm_match = re.search(r':\s*"?([^",\n]+)"?', line)
                if incoterm_match:
                    result['incoterm'] = incoterm_match.group(1).strip()

            # Location Matching
            elif 'location' in line.lower() and ':' in line:
                loc_match = re.search(r':\s*\[([^\]]+)\]', line)
                if loc_match:
                    locations_str = loc_match.group(1)
                    locations = [loc.strip().strip('"') for loc in locations_str.split(',')]
                    result['locations'] = locations

        return result
    except:
        return {
            "vessel": None,
            "commodity": None,
            "incoterm": None,
            "locations": []
        }


def generate_prediction(model, tokenizer, text, device, max_length=256):
    """Generate prediction from the model"""
    prompt = get_role_based_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    extracted_data = extract_json_from_response(response)

    return extracted_data, response


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


def safe_get_label(label_data, key, default=""):
    """Safely extract a label value from label data"""
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


def evaluate_entity_extraction(result_df):
    """Evaluate the accuracy of entity extraction"""
    results = {
        'vessel': {'correct': 0, 'total': 0},
        'port': {'correct': 0, 'total': 0},
        'commodity': {'correct': 0, 'total': 0},
        'incoterm': {'correct': 0, 'total': 0}
    }

    for idx, row in result_df.iterrows():

        # Vessel evaluation
        extracted = clean_text(row["extracted_vessel"])
        label = clean_text(row["label_vessel"])
        extracted = re.sub(r'mv\s+|m/v\s+', '', extracted)
        label = re.sub(r'mv\s+|m/v\s+', '', label)

        if label:
            results['vessel']['total'] += 1
            if label in extracted or extracted in label:
                results['vessel']['correct'] += 1

        # Location evaluation
        extracted = clean_text(row['extracted_locations'])
        label = clean_text(row['label_port'])

        if label:
            results['port']['total'] += 1
            label_ports = [p.strip() for p in label.split(',') if p.strip()]

            port_match = False
            for port in label_ports:
                if port and port in extracted:
                    port_match = True
                    break

            if port_match:
                results['port']['correct'] += 1

        # Commodity evaluation
        extracted = clean_text(row['extracted_commodity'])
        label = clean_text(row['label_commodity'])

        if label:
            results['commodity']['total'] += 1
            if label in extracted or extracted in label:
                results['commodity']['correct'] += 1

        # Incoterm evaluation
        extracted = clean_text(row['extracted_incoterm'])
        label = clean_text(row['label_incoterm'])

        if label:
            results['incoterm']['total'] += 1
            if label in extracted or extracted in label:
                results['incoterm']['correct'] += 1

    for entity in results:
        results[entity]["accuracy"] = round(results[entity]['correct'] / results[entity]['total'], 4) if \
        results[entity]['total'] > 0 else 0

    total_correct = sum(results[entity_type]['correct'] for entity_type in results)
    total_entities = sum(results[entity_type]['total'] for entity_type in results)
    overall_accuracy = total_correct / total_entities if total_entities > 0 else 0

    results['overall'] = {'accuracy': round(overall_accuracy, 4), 'correct': total_correct, 'total': total_entities}
    return results


# -------------------------------------------------------------------
# Main Evaluation
# -------------------------------------------------------------------
def evaluate_model_on_dataset(model, tokenizer, dataset_path, device, model_name, dataset_name):
    """Evaluate a single model on a dataset"""
    try:
        df = pd.read_json(dataset_path)

        print(f"Evaluating on {len(df)} samples")

        df['concatenated_emails'] = df.email_chain.apply(
            lambda email_list: "\n\n".join(email['body'] for email in email_list))

        predictions = []
        raw_responses = []

        print("Generating predictions...")
        for idx, email_text in enumerate(tqdm(df.concatenated_emails)):
            extracted_data, raw_response = generate_prediction(model, tokenizer, email_text, device)
            predictions.append(extracted_data)
            raw_responses.append(raw_response)

            if idx == 0:
                print(f"\nExample prediction:")
                print(f"Raw response: {raw_response[:200]}...")
                print(f"Extracted data: {extracted_data}")

        result_df = df.copy()

        result_df['model_name'] = model_name
        result_df['dataset_name'] = dataset_name

        result_df['predictions'] = predictions
        result_df['raw_responses'] = raw_responses

        result_df['extracted_vessel'] = result_df['predictions'].apply(lambda x: x.get('vessel', '') if x else '')
        result_df['extracted_commodity'] = result_df['predictions'].apply(lambda x: x.get('commodity', '') if x else '')
        result_df['extracted_incoterm'] = result_df['predictions'].apply(lambda x: x.get('incoterm', '') if x else '')
        result_df['extracted_locations'] = result_df['predictions'].apply(
            lambda x: ', '.join(x.get('locations', [])) if x and x.get('locations') else ''
        )

        label_column = 'labels' if 'labels' in result_df.columns else 'label'

        if label_column in result_df.columns:
            result_df["label_vessel"] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'vessel', "")
            )
            result_df["label_commodity"] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'commodity', "").replace("soybeans", "soybean")
            )
            result_df["label_incoterm"] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'incoterm', "")
            )
            result_df["label_port"] = result_df[label_column].apply(
                lambda
                    label_data: f"{safe_get_label(label_data, 'load_port', '')}, {safe_get_label(label_data, 'discharge_port', '')}"
                if isinstance(label_data, dict) else ""
            )
        else:
            print("WARNING: No label column found, using empty labels")
            result_df["label_vessel"] = ""
            result_df["label_commodity"] = ""
            result_df["label_incoterm"] = ""
            result_df["label_port"] = ""

        evaluation_results = evaluate_entity_extraction(result_df)

        output_filename = f"{model_name}_{dataset_name}.csv"
        result_df[['model_name', 'dataset_name', 'concatenated_emails', 'extracted_vessel', 'label_vessel',
                   'extracted_commodity', 'label_commodity', 'extracted_incoterm',
                   'label_incoterm', 'extracted_locations', 'label_port']].to_csv(
            output_filename,
            index=False
        )
        print(f"Saved detailed results to: {output_filename}")

        return evaluation_results

    except Exception as e:
        print(f"Error evaluating dataset: {e}")
        traceback.print_exc()
        return None


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def main():
    """Run evaluation for Qwen models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_results = []

    print(f"Total models to evaluate: {len(MODELS)}")
    print("=" * 80)

    for model_name, model_path in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'=' * 60}")

        try:
            model, tokenizer = load_qwen_model(model_path, device)
            print(f"Model loaded successfully")

            for dataset_name, dataset_path in DATASETS.items():
                print(f"\nEvaluating on {dataset_name}...")

                if not Path(dataset_path).exists():
                    print(f"ERROR: Dataset not found at {dataset_path}")
                    continue

                results = evaluate_model_on_dataset(
                    model, tokenizer, dataset_path, device, model_name, dataset_name
                )

                if results:
                    result_row = {
                        'model': model_name,
                        'dataset': dataset_name,
                        'vessel_accuracy': results['vessel']['accuracy'],
                        'vessel_correct': results['vessel']['correct'],
                        'vessel_total': results['vessel']['total'],
                        'port_accuracy': results['port']['accuracy'],
                        'port_correct': results['port']['correct'],
                        'port_total': results['port']['total'],
                        'commodity_accuracy': results['commodity']['accuracy'],
                        'commodity_correct': results['commodity']['correct'],
                        'commodity_total': results['commodity']['total'],
                        'incoterm_accuracy': results['incoterm']['accuracy'],
                        'incoterm_correct': results['incoterm']['correct'],
                        'incoterm_total': results['incoterm']['total'],
                        'overall_accuracy': results['overall']['accuracy'],
                        'overall_correct': results['overall']['correct'],
                        'overall_total': results['overall']['total']
                    }
                    all_results.append(result_row)

                    print(f"\nResults for {model_name} on {dataset_name}:")
                    print(f"  Overall Accuracy: {results['overall']['accuracy']:.2%}")
                    print(
                        f"  - Vessel: {results['vessel']['accuracy']:.2%} ({results['vessel']['correct']}/{results['vessel']['total']})")
                    print(
                        f"  - Port: {results['port']['accuracy']:.2%} ({results['port']['correct']}/{results['port']['total']})")
                    print(
                        f"  - Commodity: {results['commodity']['accuracy']:.2%} ({results['commodity']['correct']}/{results['commodity']['total']})")
                    print(
                        f"  - Incoterm: {results['incoterm']['accuracy']:.2%} ({results['incoterm']['correct']}/{results['incoterm']['total']})")
                else:
                    print(f"ERROR: Evaluation failed")

            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR with model {model_name}: {e}")
            traceback.print_exc()
            continue

    if all_results:
        results_df = pd.DataFrame(all_results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qwen_evaluation_summary_{timestamp}.csv"

        # Save to CSV
        results_df.to_csv(output_file, index=False)
        print(f"\n{'=' * 80}")
        print(f"Evaluation complete! Summary results saved to: {output_file}")

        # Print summary
        print(f"\n{'=' * 40}")
        print("RESULTS SUMMARY:")
        print(f"{'=' * 40}")
        for _, row in results_df.iterrows():
            print(f"{row['model']} on {row['dataset']}: {row['overall_accuracy']:.2%}")
    else:
        print("\nERROR: No results collected")


if __name__ == "__main__":
    print("Starting Qwen model evaluation...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()