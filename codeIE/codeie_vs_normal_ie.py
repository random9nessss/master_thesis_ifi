import os
from dotenv import load_dotenv
load_dotenv("ENV.txt")

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

import json
import time
import torch
import pandas as pd
import numpy as np
from transformers import pipeline
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import os
import gc
import warnings
warnings.filterwarnings('ignore')

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

# -------------------------------------------------------------------
# F1 Score Calculation
# -------------------------------------------------------------------
def calculate_f1_score(tp, fp, fn):
    """Calculate F1 score from true positives, false positives, and false negatives"""
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def evaluate_entity_extraction_f1(extracted: Dict, labels: Dict) -> Dict:
    """Evaluate the accuracy of entity extraction using F1 Macro scoring"""
    results = {
        'vessel': {'tp': 0, 'fp': 0, 'fn': 0},
        'port': {'tp': 0, 'fp': 0, 'fn': 0},
        'commodity': {'tp': 0, 'fp': 0, 'fn': 0},
        'incoterm': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    # Vessel evaluation
    extracted_vessel = clean_text(extracted.get('vessel', ''))
    label_vessel = clean_text(labels.get('vessel', ''))

    extracted_vessel = re.sub(r'mv\s+|m/v\s+', '', extracted_vessel)
    label_vessel = re.sub(r'mv\s+|m/v\s+', '', label_vessel)

    if extracted_vessel and label_vessel:
        if label_vessel in extracted_vessel or extracted_vessel in label_vessel:
            results['vessel']['tp'] += 1
        else:
            results['vessel']['fp'] += 1
            results['vessel']['fn'] += 1
    elif extracted_vessel and not label_vessel:
        results['vessel']['fp'] += 1
    elif not extracted_vessel and label_vessel:
        results['vessel']['fn'] += 1

    # Location/Port evaluation
    extracted_locations = extracted.get('locations', [])
    if isinstance(extracted_locations, list):
        extracted_locations_str = clean_text(', '.join(extracted_locations))
    else:
        extracted_locations_str = clean_text(str(extracted_locations))

    label_ports = []
    if 'load_port' in labels and labels['load_port']:
        label_ports.append(clean_text(labels['load_port']))
    if 'discharge_port' in labels and labels['discharge_port']:
        label_ports.append(clean_text(labels['discharge_port']))

    label_port_str = ', '.join(label_ports)

    if extracted_locations_str and label_port_str:
        port_match = False
        for port in label_ports:
            if port and port in extracted_locations_str:
                port_match = True
                break

        if port_match:
            results['port']['tp'] += 1
        else:
            results['port']['fp'] += 1
            results['port']['fn'] += 1
    elif extracted_locations_str and not label_port_str:
        results['port']['fp'] += 1
    elif not extracted_locations_str and label_port_str:
        results['port']['fn'] += 1

    # Commodity evaluation
    extracted_commodity = clean_text(extracted.get('commodity', ''))
    label_commodity = clean_text(labels.get('commodity', ''))

    label_commodity = label_commodity.replace("soybeans", "soybean")

    if extracted_commodity and label_commodity:
        if label_commodity in extracted_commodity or extracted_commodity in label_commodity:
            results['commodity']['tp'] += 1
        else:
            results['commodity']['fp'] += 1
            results['commodity']['fn'] += 1
    elif extracted_commodity and not label_commodity:
        results['commodity']['fp'] += 1
    elif not extracted_commodity and label_commodity:
        results['commodity']['fn'] += 1

    # Incoterm evaluation
    extracted_incoterm = clean_text(extracted.get('incoterm', ''))
    label_incoterm = clean_text(labels.get('incoterm', ''))

    if extracted_incoterm and label_incoterm:
        if label_incoterm in extracted_incoterm or extracted_incoterm in label_incoterm:
            results['incoterm']['tp'] += 1
        else:
            results['incoterm']['fp'] += 1
            results['incoterm']['fn'] += 1
    elif extracted_incoterm and not label_incoterm:
        results['incoterm']['fp'] += 1
    elif not extracted_incoterm and label_incoterm:
        results['incoterm']['fn'] += 1

    f1_scores = []
    for entity in results:
        tp = results[entity]['tp']
        fp = results[entity]['fp']
        fn = results[entity]['fn']

        precision, recall, f1 = calculate_f1_score(tp, fp, fn)

        results[entity]['precision'] = round(precision, 4)
        results[entity]['recall'] = round(recall, 4)
        results[entity]['f1'] = round(f1, 4)

        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    total_tp = sum(results[entity]['tp'] for entity in results)
    total_fp = sum(results[entity]['fp'] for entity in results)
    total_fn = sum(results[entity]['fn'] for entity in results)

    overall_precision, overall_recall, overall_f1 = calculate_f1_score(total_tp, total_fp, total_fn)

    results['overall'] = {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': round(overall_precision, 4),
        'recall': round(overall_recall, 4),
        'f1_micro': round(overall_f1, 4),
        'f1_macro': round(macro_f1, 4)
    }

    return results

# -------------------------------------------------------------------
# MaritimeEntityExtractor
# -------------------------------------------------------------------
class MaritimeEntityExtractor:
    """Base class for maritime entity extraction with GPU memory management"""

    def __init__(self, model_id: str, model_name: str, model_type: str = "standard"):
        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = None

    def load_model(self):
        """Load the model pipeline"""
        try:
            if "t5" in self.model_id.lower():
                self.pipe = pipeline(
                    "text2text-generation",
                    model=self.model_id,
                    device=self.device,
                    max_new_tokens=200,
                    do_sample=False,
                    trust_remote_code=True
                )
            else:
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device=self.device,
                    max_new_tokens=200,
                    do_sample=False,
                    trust_remote_code=True,
                    pad_token_id=50256
                )
            print(f"Loaded {self.model_name}")

            if torch.cuda.is_available():
                print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            return True
        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            return False

    def cleanup(self):
        """Clean up model and free GPU memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()

            print(f"Cleaned up {self.model_name}")
            if torch.cuda.is_available():
                print(f"  GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    def create_standard_prompt(self, text: str) -> str:
        """Create standard extraction prompt - aligned with Qwen prompt style"""
        text = text[:500]
        return f"""
                You are a shipping data extractor. Extract shipping information from the text.

                Text: {text}
                
                Return a JSON object with:
                - vessel (ship name)
                - commodity (cargo type)
                - incoterm (trade term like CIF, FOB)
                - locations (list of places)
                
                JSON:
            """

    def create_code_prompt(self, text: str) -> str:
        """Create code-style prompt for code models"""
        text = text[:500]
        return f'''
            def extract_maritime_entities(email_text):
                """Extract maritime shipping entities."""
            
                # Example
                email = "MV STAR loading wheat Hamburg FOB"
                entities = []
                entities.append({{"text": "STAR", "type": "vessel"}})
                entities.append({{"text": "Hamburg", "type": "location"}})
                entities.append({{"text": "wheat", "type": "commodity"}})
                entities.append({{"text": "FOB", "type": "incoterm"}})
            
                # Extract from this email
                email = "{text}"
                entities = []
            '''

    def create_instruction_prompt(self, text: str) -> str:
        """Create instruction-focused prompt for instruction-tuned models"""
        text = text[:500]

        if "t5" in self.model_id.lower():
            return f"""
                Extract shipping entities from this email. Return JSON with vessel, commodity, incoterm, and locations.
    
                Email: {text}
                
                JSON:
            """

        return f"""
            You are an expert at extracting maritime shipping information from emails.
    
            Extract the following entity types from this email and return them as a JSON object:
            - vessel: Names of ships or vessels
            - commodity: Types of cargo or goods being shipped
            - incoterm: Trade terms like FOB, CIF, CFR, etc.
            - locations: Ports, terminals, or geographic locations
            
            Email text: "{text}"
            
            Return only a JSON object with the extracted entities in this exact format:
            {{"vessel": "", "commodity": "", "incoterm": "", "locations": []}}
            
            JSON:
        """

    def extract(self, text: str) -> tuple:
        """Extract entities and return (result, generated_text, inference_time)"""
        if self.model_type == "code":
            prompt = self.create_code_prompt(text)
        elif self.model_type == "instruction":
            prompt = self.create_instruction_prompt(text)
        else:
            prompt = self.create_standard_prompt(text)

        start_time = time.time()
        try:
            if "t5" in self.model_id.lower():
                output = self.pipe(prompt, max_length=200, temperature=0.1)[0]['generated_text']
                generated = output
            else:
                output = self.pipe(prompt, max_length=len(prompt)+200, temperature=0.1)[0]['generated_text']
                generated = output[len(prompt):]

            inference_time = time.time() - start_time

            if self.model_type == "code":
                result = self.parse_code_output(generated)
            else:
                result = self.parse_json_output(generated)

            return result, generated, inference_time, None
        except Exception as e:
            return {'vessel': None, 'commodity': None, 'incoterm': None, 'locations': []}, "", 0, str(e)

    def parse_json_output(self, text: str) -> Dict:
        """Parse JSON output - aligned with Qwen format"""
        try:
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
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

            lines = text.split('\n')
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

    def parse_code_output(self, text: str) -> Dict:
        """Parse code output"""
        result = {'vessel': None, 'commodity': None, 'incoterm': None, 'locations': []}

        append_pattern = r'entities\.append\(\s*{([^}]+)}\s*\)'
        matches = re.findall(append_pattern, text)

        for match in matches:
            try:
                text_match = re.search(r'"text"\s*:\s*"([^"]+)"', match)
                type_match = re.search(r'"type"\s*:\s*"([^"]+)"', match)

                if text_match and type_match:
                    entity_text = text_match.group(1)
                    entity_type = type_match.group(1)

                    if entity_type == "vessel":
                        result['vessel'] = entity_text
                    elif entity_type == "commodity":
                        result['commodity'] = entity_text
                    elif entity_type == "incoterm":
                        result['incoterm'] = entity_text
                    elif entity_type == "location" and entity_text:
                        if result['locations'] is None:
                            result['locations'] = []
                        result['locations'].append(entity_text)
            except:
                continue

        return result

# -------------------------------------------------------------------
# Helper function to prepare labels
# -------------------------------------------------------------------
def prepare_labels_from_dataset(sample_labels: Dict) -> Dict:
    """Convert dataset labels to Qwen evaluation format"""
    if not isinstance(sample_labels, dict):
        return {}

    result = {}

    # Handle vessel
    vessel_keys = ['vessel', 'vessel_name', 'ship_name', 'ship', 'vessel_id']
    for key in vessel_keys:
        if key in sample_labels and sample_labels[key]:
            result['vessel'] = str(sample_labels[key])
            break

    # Handle commodity
    if 'commodity' in sample_labels and sample_labels['commodity']:
        result['commodity'] = str(sample_labels['commodity'])

    # Handle incoterm
    incoterm_keys = ['incoterm', 'terms', 'trade_terms', 'delivery_terms']
    for key in incoterm_keys:
        if key in sample_labels and sample_labels[key]:
            result['incoterm'] = str(sample_labels[key])
            break

    # Handle ports
    if 'load_port' in sample_labels and sample_labels['load_port']:
        result['load_port'] = str(sample_labels['load_port'])
    if 'discharge_port' in sample_labels and sample_labels['discharge_port']:
        result['discharge_port'] = str(sample_labels['discharge_port'])

    return result

# -------------------------------------------------------------------
# Model configurations
# -------------------------------------------------------------------
MODEL_PAIRS = {
    "~60M": {
        "code": ("Salesforce/codet5-small", "CodeT5-Small (60M)")
    },
    "~110M": {
        "code": ("codeparrot/codeparrot-small", "CodeParrot-Small (110M)")
    },
    "~350M-Mono": {
        "code": ("Salesforce/codegen-350M-mono", "CodeGen-350M-Mono")
    },
    "~1B-StarEncoder": {
        "code": ("bigcode/starencoder", "StarEncoder (1B)")
    },
    "~1.4B-CodeGPT": {
        "code": ("microsoft/CodeGPT-small-java-adaptedGPT2", "CodeGPT-Java (1.4B)")
    },
    "~2.7B-CodeGen": {
        "code": ("Salesforce/codegen-2B-nl", "CodeGen-2B-NL")
    },
    "~3B": {
        "code": ("Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen2.5-Coder-3B")
    },
    "~1B-CodeGen": {
        "code": ("Salesforce/codegen-1B-multi", "CodeGen-1B")
    },
    "~80M-IT": {
        "instruction": ("google/flan-t5-small", "Flan-T5-Small (80M)")
    },
    "~160M": {
        "code": ("bigcode/tiny_starcoder_py", "TinyStarCoder (164M)"),
        "standard": ("EleutherAI/pythia-160m", "Pythia-160M")
    },
    "~220M": {
        "code": ("Salesforce/codet5-base", "CodeT5-Base (220M)")
    },
    "~350M": {
        "code": ("Salesforce/codegen-350M-multi", "CodeGen-350M"),
        "standard": ("microsoft/phi-1", "Phi-1 (350M)")
    },
    "~500M": {
        "standard": ("Qwen/Qwen2-0.5B-Instruct", "Qwen2-0.5B")
    },
    "~780M-IT": {
        "instruction": ("google/flan-t5-large", "Flan-T5-Large (780M)")
    },
    "~1.1B": {
        "code": ("bigcode/santacoder", "SantaCoder (1.1B)"),
        "standard": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B")
    },
    "~1.3B": {
        "standard": ("microsoft/phi-1_5", "Phi-1.5 (1.3B)")
    },
    "~1.5B": {
        "standard": ("Qwen/Qwen2-1.5B-Instruct", "Qwen2-1.5B")
    },
    "~1.6B": {
        "standard": ("stabilityai/stablelm-2-1_6b", "StableLM-2-1.6B")
    },
    "~2B": {
        "code": ("Salesforce/codegen-2B-multi", "CodeGen-2B")
    },
    "~2.7B": {
        "standard": ("microsoft/phi-2", "Phi-2 (2.7B)"),
        "code": ("Salesforce/codegen-2B-multi", "CodeGen-2B")
    },
    "~3B-Code": {
        "code": ("bigcode/starcoder2-3b", "StarCoder2-3B")
    },
    "~3B-Standard": {
        "standard": ("stabilityai/stablelm-3b-4e1t", "StableLM-3B")
    },
    "~3B-OpenLlama": {
        "standard": ("openlm-research/open_llama_3b_v2", "OpenLlama-3B")
    }
}

# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
def process_model_with_cleanup(model_id: str, model_name: str, model_type: str,
                             test_samples: pd.DataFrame, num_samples: int):
    """Process a model and clean up GPU memory afterwards"""

    extractor = MaritimeEntityExtractor(model_id, model_name, model_type=model_type)

    if not extractor.load_model():
        return None

    model_results = []
    print(f"\nProcessing {num_samples} samples with {model_name}...")

    for idx, (_, sample) in enumerate(test_samples.iterrows()):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{num_samples} samples processed")

        if 'concatenated_emails' in sample:
            text = sample['concatenated_emails']
        elif 'text' in sample:
            text = sample['text']
        elif 'email_chain' in sample:
            chain = sample['email_chain']
            if isinstance(chain, list):
                text = "\n\n".join(email['body'] for email in chain)
            else:
                text = str(chain)
        else:
            continue

        if not text:
            continue

        extracted, generated, inference_time, error = extractor.extract(text)

        labels = {}
        if 'labels' in sample and isinstance(sample['labels'], dict):
            labels = prepare_labels_from_dataset(sample['labels'])
        elif 'label' in sample and isinstance(sample['label'], dict):
            labels = prepare_labels_from_dataset(sample['label'])

        evaluation_results = evaluate_entity_extraction_f1(extracted, labels)

        result = {
            'sample_id': idx,
            'text': text[:1000],
            'model_name': model_name,
            'extracted_vessel': extracted.get('vessel', ''),
            'extracted_commodity': extracted.get('commodity', ''),
            'extracted_incoterm': extracted.get('incoterm', ''),
            'extracted_locations': ', '.join(extracted.get('locations', [])) if isinstance(extracted.get('locations'), list) else '',
            'label_vessel': labels.get('vessel', ''),
            'label_commodity': labels.get('commodity', ''),
            'label_incoterm': labels.get('incoterm', ''),
            'label_port': f"{labels.get('load_port', '')}, {labels.get('discharge_port', '')}" if labels else '',
            'evaluation_results': evaluation_results,
            'generated_output': generated,
            'inference_time': inference_time,
            'error': error
        }

        model_results.append(result)

        if idx < 3:
            print(f"\n  --- Sample {idx} ---")
            print(f"  Extracted: vessel={extracted.get('vessel')}, commodity={extracted.get('commodity')}")
            print(f"  Labels: vessel={labels.get('vessel')}, commodity={labels.get('commodity')}")
            print(f"  F1 Macro: {evaluation_results['overall']['f1_macro']:.3f}")

    extractor.cleanup()

    return model_results

def main():
    """Main execution"""
    print("="*80)
    print("MARITIME ENTITY EXTRACTION - F1 MACRO EVALUATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    if torch.cuda.is_available():
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    else:
        print("\nNo GPU detected, using CPU")

    output_dir = "maritime_extraction_f1_eval"
    os.makedirs(output_dir, exist_ok=True)

    total_models = sum(len(models) for models in MODEL_PAIRS.values())
    print(f"\nTotal models to evaluate: {total_models}")
    print(f"Model categories: {len(MODEL_PAIRS)}")

    # Load data
    print("\nLoading maritime email data...")
    try:
        df = pd.read_json("aggregated.json")

        if 'concatenated_emails' not in df.columns:
            if 'email_chain' in df.columns:
                df['concatenated_emails'] = df['email_chain'].apply(
                    lambda chain: "\n\n".join(email['body'] for email in chain)
                    if isinstance(chain, list) else str(chain)
                )
            elif 'text' in df.columns:
                df['concatenated_emails'] = df['text']

        num_samples = min(50, len(df))
        test_samples = df.sample(n=num_samples, random_state=42)
        print(f"Loaded {num_samples} samples for testing")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    all_results = []
    summary_results = []
    models_processed = 0

    for size_category, models in MODEL_PAIRS.items():
        print(f"\n{'='*60}")
        print(f"Testing {size_category} models")
        print('='*60)

        for model_type, (model_id, model_name) in models.items():
            models_processed += 1
            print(f"\n[{models_processed}/{total_models}] Processing {model_name}...")

            try:
                model_results = process_model_with_cleanup(
                    model_id, model_name, model_type, test_samples, num_samples
                )

                if model_results is None:
                    print(f"Skipping {model_name} due to loading error")
                    continue

                entity_results = {
                    'vessel': {'tp': 0, 'fp': 0, 'fn': 0},
                    'port': {'tp': 0, 'fp': 0, 'fn': 0},
                    'commodity': {'tp': 0, 'fp': 0, 'fn': 0},
                    'incoterm': {'tp': 0, 'fp': 0, 'fn': 0},
                    'overall': {'tp': 0, 'fp': 0, 'fn': 0}
                }

                for result in model_results:
                    eval_res = result['evaluation_results']
                    for entity_type in ['vessel', 'port', 'commodity', 'incoterm', 'overall']:
                        entity_results[entity_type]['tp'] += eval_res[entity_type]['tp']
                        entity_results[entity_type]['fp'] += eval_res[entity_type]['fp']
                        entity_results[entity_type]['fn'] += eval_res[entity_type]['fn']

                f1_scores = []
                for entity_type in ['vessel', 'port', 'commodity', 'incoterm']:
                    tp = entity_results[entity_type]['tp']
                    fp = entity_results[entity_type]['fp']
                    fn = entity_results[entity_type]['fn']

                    precision, recall, f1 = calculate_f1_score(tp, fp, fn)

                    entity_results[entity_type]['precision'] = precision
                    entity_results[entity_type]['recall'] = recall
                    entity_results[entity_type]['f1'] = f1

                    f1_scores.append(f1)

                overall_tp = entity_results['overall']['tp']
                overall_fp = entity_results['overall']['fp']
                overall_fn = entity_results['overall']['fn']

                overall_precision, overall_recall, overall_f1_micro = calculate_f1_score(
                    overall_tp, overall_fp, overall_fn
                )

                f1_macro = np.mean(f1_scores)

                summary = {
                    'model': model_name,
                    'model_id': model_id,
                    'size_category': size_category,
                    'model_type': model_type,
                    'vessel_f1': entity_results['vessel']['f1'],
                    'vessel_precision': entity_results['vessel']['precision'],
                    'vessel_recall': entity_results['vessel']['recall'],
                    'port_f1': entity_results['port']['f1'],
                    'port_precision': entity_results['port']['precision'],
                    'port_recall': entity_results['port']['recall'],
                    'commodity_f1': entity_results['commodity']['f1'],
                    'commodity_precision': entity_results['commodity']['precision'],
                    'commodity_recall': entity_results['commodity']['recall'],
                    'incoterm_f1': entity_results['incoterm']['f1'],
                    'incoterm_precision': entity_results['incoterm']['precision'],
                    'incoterm_recall': entity_results['incoterm']['recall'],
                    'f1_macro': f1_macro,
                    'f1_micro': overall_f1_micro,
                    'overall_precision': overall_precision,
                    'overall_recall': overall_recall,
                    'avg_inference_time': np.mean([r['inference_time'] for r in model_results])
                }

                summary_results.append(summary)
                all_results.extend(model_results)

                # Print results
                print(f"\nResults for {model_name}:")
                print(f"  F1 Macro: {f1_macro:.3f}")
                print(f"  F1 Micro: {overall_f1_micro:.3f}")
                print(f"  Entity F1 Scores:")
                print(f"    - Vessel: {entity_results['vessel']['f1']:.3f} (P: {entity_results['vessel']['precision']:.3f}, R: {entity_results['vessel']['recall']:.3f})")
                print(f"    - Port: {entity_results['port']['f1']:.3f} (P: {entity_results['port']['precision']:.3f}, R: {entity_results['port']['recall']:.3f})")
                print(f"    - Commodity: {entity_results['commodity']['f1']:.3f} (P: {entity_results['commodity']['precision']:.3f}, R: {entity_results['commodity']['recall']:.3f})")
                print(f"    - Incoterm: {entity_results['incoterm']['f1']:.3f} (P: {entity_results['incoterm']['precision']:.3f}, R: {entity_results['incoterm']['recall']:.3f})")
                print(f"  Avg inference time: {summary['avg_inference_time']:.3f}s")

                if models_processed % 5 == 0:
                    intermediate_df = pd.DataFrame(summary_results)
                    intermediate_file = os.path.join(output_dir, f"intermediate_f1_summary_{models_processed}_models.csv")
                    intermediate_df.to_csv(intermediate_file, index=False)
                    print(f"\nIntermediate results saved to: {intermediate_file}")

            except Exception as e:
                print(f"ERROR processing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_file = os.path.join(output_dir, f"f1_evaluation_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nFinal summary saved to: {summary_file}")

        detailed_df = pd.DataFrame(all_results)
        detailed_file = os.path.join(output_dir, f"f1_detailed_results_{timestamp}.csv")

        detailed_columns = [
            'model_name', 'sample_id', 'text',
            'extracted_vessel', 'label_vessel',
            'extracted_commodity', 'label_commodity',
            'extracted_incoterm', 'label_incoterm',
            'extracted_locations', 'label_port',
            'inference_time'
        ]

        detailed_df[detailed_columns].to_csv(detailed_file, index=False)
        print(f"Detailed results saved to: {detailed_file}")

        print("\n" + "="*80)
        print("F1 EVALUATION SUMMARY")
        print("="*80)

        print(f"\nSuccessfully evaluated {len(summary_results)} out of {total_models} models")

        print("\nTop performing models by F1 Macro:")
        summary_df_sorted = summary_df.sort_values('f1_macro', ascending=False)
        for i, (_, row) in enumerate(summary_df_sorted.head(10).iterrows()):
            print(f"{i+1}. {row['model']} ({row['model_type']}): F1 Macro = {row['f1_macro']:.3f}")

        print("\nBest models by type (F1 Macro):")
        for model_type in ['code', 'standard', 'instruction']:
            type_models = summary_df[summary_df['model_type'] == model_type]
            if len(type_models) > 0:
                best = type_models.sort_values('f1_macro', ascending=False).iloc[0]
                print(f"  Best {model_type}: {best['model']} - F1 Macro = {best['f1_macro']:.3f}")

        print("\nAverage F1 scores by entity type across all models:")
        print(f"  Vessel: {summary_df['vessel_f1'].mean():.3f}")
        print(f"  Port: {summary_df['port_f1'].mean():.3f}")
        print(f"  Commodity: {summary_df['commodity_f1'].mean():.3f}")
        print(f"  Incoterm: {summary_df['incoterm_f1'].mean():.3f}")
        print(f"  Overall F1 Macro: {summary_df['f1_macro'].mean():.3f}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()