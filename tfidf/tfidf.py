import re
import warnings
import pandas as pd
import spacy
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import traceback
from pathlib import Path

# -------------------------------------------------------------------
# Warning Silencing and Setup
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Loading Spacy Model
# -------------------------------------------------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------------------------
# Dataset Configuration (from GLiNER script)
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


def evaluate_entity_extraction_f1(result_df):
    """Evaluate precision, recall, and F1 for entity extraction"""
    results = {
        'vessel': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'port': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'commodity': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'incoterm': {'precision': [], 'recall': [], 'f1': [], 'tp': 0, 'fp': 0, 'fn': 0}
    }

    for idx, row in result_df.iterrows():

        # Vessel evaluation
        extracted = row["extracted_vessel_name"]
        if isinstance(extracted, str):
            extracted = [e.strip() for e in extracted.split(',') if e.strip()]
        else:
            extracted = extracted if isinstance(extracted, list) else []
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

        # Port evaluation
        extracted = row['extracted_port']
        if isinstance(extracted, str):
            extracted = [e.strip() for e in extracted.split(',') if e.strip()]
        else:
            extracted = extracted if isinstance(extracted, list) else []
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

        # Commodity evaluation
        extracted = row['extracted_commodity']
        if isinstance(extracted, str):
            extracted = [e.strip() for e in extracted.split(',') if e.strip()]
        else:
            extracted = extracted if isinstance(extracted, list) else []
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

        # Incoterm evaluation
        extracted = row['extracted_incoterm']
        if isinstance(extracted, str):
            extracted = [e.strip() for e in extracted.split(',') if e.strip()]
        else:
            extracted = extracted if isinstance(extracted, list) else []
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


# -------------------------------------------------------------------
# Rule-Based Extraction System
# -------------------------------------------------------------------
class RuleBasedExtractor:
    def __init__(self, corpus=None):
        self.nlp = nlp
        self.tfidf_vectorizer = None
        self.tfidf_features = None
        self.domain_terms = defaultdict(set)

        # Maritime vessel prefixes/suffixes
        self.vessel_prefixes = {
            'mv', 'm/v', 'mt', 'm/t', 'ss', 's/s', 'ms', 'm/s',
            'bulk carrier', 'tanker', 'vessel', 'ship'
        }

        # Port/Location Indicators
        self.port_indicators = {
            'port', 'terminal', 'berth', 'anchorage', 'from', 'to', 'at',
            'load', 'discharge', 'loading', 'discharging', 'destination',
            'origin', 'departure', 'arrival'
        }

        # Commodity Indicators
        self.commodity_indicators = {
            'cargo', 'commodity', 'product', 'grade', 'tons', 'mt', 'tonnes',
            'barrels', 'bbls', 'bushels', 'metric tons', 'shipment', 'parcel', 'lot'
        }

        # Incoterm Indicators
        self.incoterm_indicators = {
            'terms', 'basis', 'delivered', 'ex-', 'free', 'cost',
            'freight', 'insurance', 'incoterms'
        }

        self.context_words = set()

        if corpus:
            self.train_tfidf(corpus)

    def train_tfidf(self, corpus):
        """Train TF-IDF on the corpus to identify domain-specific terms"""
        print("Training TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9/-]*\b'
        )

        self.tfidf_features = self.tfidf_vectorizer.fit_transform(corpus)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        for doc_idx in range(len(corpus)):
            doc_tfidf = self.tfidf_features[doc_idx].toarray()[0]
            top_indices = doc_tfidf.argsort()[-100:][::-1]

            for idx in top_indices:
                if doc_tfidf[idx] > 0.1:
                    term = feature_names[idx].lower()
                    if any(ind in term for ind in self.vessel_prefixes):
                        self.domain_terms['vessel'].add(term)
                    elif any(ind in term for ind in self.commodity_indicators):
                        self.domain_terms['commodity'].add(term)
                    elif len(term) == 3 and term.isupper():
                        self.domain_terms['incoterm'].add(term)

    def extract_vessels(self, text):
        """Extract vessel names using rules and patterns"""
        vessels = set()
        doc = self.nlp(text)
        text_lower = text.lower()

        vessel_pattern = r'\b(?:' + '|'.join(self.vessel_prefixes) + r')\s+([A-Z][A-Za-z0-9\s\-\.]{2,30})\b'
        matches = re.finditer(vessel_pattern, text, re.IGNORECASE)
        for match in matches:
            vessel_name = match.group(1).strip()
            vessel_name = re.sub(r'\s+', ' ', vessel_name)
            vessels.add(vessel_name.lower())

        quoted_pattern = r'["\']([A-Z][A-Za-z0-9\s\-\.]{2,30})["\']'
        matches = re.finditer(quoted_pattern, text)
        for match in matches:
            potential_vessel = match.group(1).strip()
            context_start = max(0, match.start() - 50)
            context = text_lower[context_start:match.start()].lower()
            if any(prefix in context for prefix in self.vessel_prefixes):
                vessels.add(potential_vessel.lower())

        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                context = text_lower[max(0, ent.start_char - 30):ent.end_char + 30]
                if any(ind in context for ind in self.vessel_prefixes):
                    vessels.add(ent.text.lower())

        return list(vessels)

    def extract_ports(self, text):
        """Extract port/location names"""
        ports = set()
        doc = self.nlp(text)
        text_lower = text.lower()

        port_pattern = r'\b(?:' + '|'.join(
            self.port_indicators) + r')\s+(?:of\s+|at\s+|in\s+)?([A-Z][a-zA-Z\s\-]{2,30})\b'
        matches = re.finditer(port_pattern, text, re.IGNORECASE)
        for match in matches:
            port_name = match.group(1).strip()
            ports.add(port_name.lower())

        from_to_pattern = r'\bfrom\s+([A-Z][a-zA-Z\s\-]{2,30})\s+to\s+([A-Z][a-zA-Z\s\-]{2,30})\b'
        matches = re.finditer(from_to_pattern, text, re.IGNORECASE)
        for match in matches:
            ports.add(match.group(1).strip().lower())
            ports.add(match.group(2).strip().lower())

        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:
                context = text_lower[max(0, ent.start_char - 50):ent.end_char + 50]
                if any(ind in context for ind in self.port_indicators):
                    ports.add(ent.text.lower())

        port_of_pattern = r'\bport\s+of\s+([A-Z][a-zA-Z\s\-]{2,30})\b'
        matches = re.finditer(port_of_pattern, text, re.IGNORECASE)
        for match in matches:
            ports.add(match.group(1).strip().lower())

        return list(ports)

    def extract_commodities(self, text):
        """Extract commodity names"""
        commodities = set()
        doc = self.nlp(text)
        text_lower = text.lower()

        quantity_pattern = r'\b(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:mt|metric tons?|tons?|tonnes?|barrels?|bbls?)\s+(?:of\s+)?([a-zA-Z][a-zA-Z\s\-]{2,30})\b'
        matches = re.finditer(quantity_pattern, text, re.IGNORECASE)
        for match in matches:
            commodity = match.group(2).strip()
            if commodity.lower() not in ['the', 'of', 'at', 'in', 'for', 'per']:
                commodities.add(commodity.lower())

        for token in doc:
            if token.text.lower() in self.commodity_indicators:
                for i in range(1, 4):
                    if token.i + i < len(doc):
                        next_tokens = doc[token.i + 1:token.i + i + 1]
                        noun_phrase = ' '.join([t.text for t in next_tokens if t.pos_ in ['NOUN', 'PROPN', 'ADJ']])
                        if noun_phrase and len(noun_phrase) > 2:
                            commodities.add(noun_phrase.lower())

        grade_pattern = r'\b(?:grade|quality|type)\s+([A-Za-z0-9\-]+)\s+([a-zA-Z\s]{2,20})\b'
        matches = re.finditer(grade_pattern, text, re.IGNORECASE)
        for match in matches:
            commodities.add(match.group(2).strip().lower())

        if hasattr(self, 'domain_terms') and 'commodity' in self.domain_terms:
            for term in self.domain_terms['commodity']:
                if term in text_lower:
                    commodities.add(term)

        return list(commodities)

    def extract_incoterms(self, text):
        """Extract incoterms"""
        incoterms = set()
        text_lower = text.lower()

        three_letter_pattern = r'\b([A-Z]{3})\b'
        matches = re.finditer(three_letter_pattern, text)

        for match in matches:
            potential_incoterm = match.group(1)
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text_lower[context_start:context_end]

            if any(ind in context for ind in self.incoterm_indicators):
                incoterms.add(potential_incoterm.lower())

        terms_pattern = r'\b(?:terms|basis|incoterms?)\s*:?\s*([A-Z]{3,4})\b'
        matches = re.finditer(terms_pattern, text, re.IGNORECASE)
        for match in matches:
            incoterms.add(match.group(1).lower())

        price_pattern = r'\b(?:price|pricing|delivery)\s+(?:basis|terms)\s*:?\s*([A-Z]{3,4})\b'
        matches = re.finditer(price_pattern, text, re.IGNORECASE)
        for match in matches:
            incoterms.add(match.group(1).lower())

        location_pattern = r'\b([A-Z]{3})\s+(?:[A-Za-z\s\-]{2,30})\s+(?:port|terminal)\b'
        matches = re.finditer(location_pattern, text)
        for match in matches:
            potential = match.group(1)
            context = text_lower[max(0, match.start() - 30):match.start()]
            if any(ind in context for ind in ['price', 'cost', 'terms', 'basis']):
                incoterms.add(potential.lower())

        return list(incoterms)

    def extract_all_entities(self, text):
        """Extract all entity types from text"""
        return {
            'vessels': self.extract_vessels(text),
            'ports': self.extract_ports(text),
            'commodities': self.extract_commodities(text),
            'incoterms': self.extract_incoterms(text)
        }


# -------------------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------------------
def evaluate_dataset(dataset_name, dataset_path, extractor):
    """Evaluate rule-based extraction on a single dataset"""
    try:
        if not Path(dataset_path).exists():
            print(f"WARNING: Dataset file not found: {dataset_path}")
            return None

        print(f"Loading dataset: {dataset_name}...")
        df = pd.read_json(dataset_path)
        df['concatenated_emails'] = df.email_chain.apply(
            lambda email_list: "\n\n".join(email['body'] for email in email_list))
        print(f"Dataset loaded with {len(df)} examples")

        print("Extracting entities...")
        vessel_extractions = {}
        location_extractions = {}
        commodity_extractions = {}
        incoterm_extractions = {}

        for index, email_text in tqdm(enumerate(df.concatenated_emails), total=len(df), desc="Processing"):
            entities = extractor.extract_all_entities(email_text)

            vessel_extractions[index] = entities['vessels']
            location_extractions[index] = entities['ports']
            commodity_extractions[index] = entities['commodities']
            incoterm_extractions[index] = entities['incoterms']

        result_df = df.copy()

        label_column = None
        if 'labels' in result_df.columns:
            label_column = 'labels'
        elif 'label' in result_df.columns:
            label_column = 'label'
        else:
            print(f"WARNING: No 'labels' or 'label' column found in dataset. Using empty labels.")

        result_df["extracted_vessel_name"] = [', '.join(vessel_extractions.get(i, [])) for i in range(len(df))]
        if label_column:
            result_df["label_vessel_name"] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'vessel', "")
            )
        else:
            result_df["label_vessel_name"] = ""

        result_df["extracted_port"] = [', '.join(location_extractions.get(i, [])) for i in range(len(df))]
        if label_column:
            result_df["label_port"] = result_df[label_column].apply(
                lambda
                    label_data: f"{safe_get_label(label_data, 'load_port', '')}, {safe_get_label(label_data, 'discharge_port', '')}"
                if isinstance(label_data, dict) else ""
            )
        else:
            result_df["label_port"] = ""

        result_df["extracted_commodity"] = [', '.join(commodity_extractions.get(i, [])) for i in range(len(df))]
        if label_column:
            result_df['label_commodity'] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'commodity', "").replace("soybeans", "soybean")
            )
        else:
            result_df['label_commodity'] = ""

        result_df["extracted_incoterm"] = [', '.join(incoterm_extractions.get(i, [])) for i in range(len(df))]
        if label_column:
            result_df['label_incoterm'] = result_df[label_column].apply(
                lambda label_data: safe_get_label(label_data, 'incoterm', "")
            )
        else:
            result_df['label_incoterm'] = ""

        print("Evaluating extraction performance...")
        evaluation_results = evaluate_entity_extraction_f1(result_df)

        return evaluation_results

    except Exception as e:
        print(f"Error evaluating dataset {dataset_name}: {e}")
        traceback.print_exc()
        return None


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def run_comprehensive_evaluation():
    """Run evaluation for all datasets"""
    print("Starting comprehensive rule-based extraction evaluation with F1 scores...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total datasets to evaluate: {len(DATASETS)}")
    print("=" * 80)

    print("Building combined corpus for TF-IDF training...")
    all_texts = []
    for dataset_name, dataset_path in DATASETS.items():
        if Path(dataset_path).exists():
            try:
                df = pd.read_json(dataset_path)
                df['concatenated_emails'] = df.email_chain.apply(
                    lambda email_list: "\n\n".join(email['body'] for email in email_list))
                all_texts.extend(df['concatenated_emails'].tolist())
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")

    print(f"Total documents in corpus: {len(all_texts)}")
    print("Initializing rule-based extractor...")
    extractor = RuleBasedExtractor(corpus=all_texts)

    all_results = []

    current = 0
    for dataset_name, dataset_path in DATASETS.items():
        current += 1
        print(f"\n[{current}/{len(DATASETS)}] Evaluating dataset: {dataset_name}")
        print("-" * 60)

        results = evaluate_dataset(dataset_name, dataset_path, extractor)

        if results:
            result_row = {
                'model': 'rule_based_tfidf',
                'dataset': dataset_name,

                'vessel_precision': results['vessel']['avg_precision'],
                'vessel_recall': results['vessel']['avg_recall'],
                'vessel_f1': results['vessel']['avg_f1'],
                'vessel_micro_precision': results['vessel']['micro_precision'],
                'vessel_micro_recall': results['vessel']['micro_recall'],
                'vessel_micro_f1': results['vessel']['micro_f1'],

                'port_precision': results['port']['avg_precision'],
                'port_recall': results['port']['avg_recall'],
                'port_f1': results['port']['avg_f1'],
                'port_micro_precision': results['port']['micro_precision'],
                'port_micro_recall': results['port']['micro_recall'],
                'port_micro_f1': results['port']['micro_f1'],

                'commodity_precision': results['commodity']['avg_precision'],
                'commodity_recall': results['commodity']['avg_recall'],
                'commodity_f1': results['commodity']['avg_f1'],
                'commodity_micro_precision': results['commodity']['micro_precision'],
                'commodity_micro_recall': results['commodity']['micro_recall'],
                'commodity_micro_f1': results['commodity']['micro_f1'],

                'incoterm_precision': results['incoterm']['avg_precision'],
                'incoterm_recall': results['incoterm']['avg_recall'],
                'incoterm_f1': results['incoterm']['avg_f1'],
                'incoterm_micro_precision': results['incoterm']['micro_precision'],
                'incoterm_micro_recall': results['incoterm']['micro_recall'],
                'incoterm_micro_f1': results['incoterm']['micro_f1'],

                'overall_precision': results['overall']['precision'],
                'overall_recall': results['overall']['recall'],
                'overall_f1': results['overall']['f1'],
                'overall_micro_precision': results['overall']['micro_precision'],
                'overall_micro_recall': results['overall']['micro_recall'],
                'overall_micro_f1': results['overall']['micro_f1'],

                'total_tp': results['overall']['tp'],
                'total_fp': results['overall']['fp'],
                'total_fn': results['overall']['fn']
            }
            all_results.append(result_row)

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
            print(f"  ERROR: Evaluation failed for this dataset")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('dataset')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"rule_based_evaluation_f1_results_{timestamp}.csv"

        results_df.to_csv(output_file, index=False)
        print(f"\n{'=' * 80}")
        print(f"Evaluation complete! Results saved to: {output_file}")
        print(f"Total successful evaluations: {len(all_results)}/{len(DATASETS)}")

        print(f"\n{'=' * 40}")
        print("OVERALL PERFORMANCE SUMMARY:")
        print(f"{'=' * 40}")
        print(
            f"Average Macro F1 across all datasets: {results_df['overall_f1'].mean():.4f} (±{results_df['overall_f1'].std():.4f})")
        print(
            f"Average Micro F1 across all datasets: {results_df['overall_micro_f1'].mean():.4f} (±{results_df['overall_micro_f1'].std():.4f})")

        print(f"\n{'=' * 40}")
        print("PERFORMANCE BY DATASET TYPE:")
        print(f"{'=' * 40}")
        results_df['dataset_type'] = results_df['dataset'].apply(lambda x: x.split('_')[0])
        dataset_summary = results_df.groupby('dataset_type')[['overall_f1', 'overall_micro_f1']].agg(['mean', 'std'])
        print(dataset_summary.round(4))

        print(f"\n{'=' * 40}")
        print("ENTITY-SPECIFIC PERFORMANCE:")
        print(f"{'=' * 40}")
        for entity in ['vessel', 'port', 'commodity', 'incoterm']:
            avg_f1 = results_df[f'{entity}_f1'].mean()
            std_f1 = results_df[f'{entity}_f1'].std()
            print(f"{entity.capitalize()}: F1={avg_f1:.4f} (±{std_f1:.4f})")

        detailed_output_file = f"rule_based_detailed_results_{timestamp}.csv"
        results_df.to_csv(detailed_output_file, index=False)
        print(f"\nDetailed results saved to: {detailed_output_file}")

    else:
        print("\nERROR: No results were collected. Please check the dataset paths.")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_comprehensive_evaluation()