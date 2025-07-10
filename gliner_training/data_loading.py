# data_loading

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter

import torch
from torch.utils.data import Dataset


# =============================================================================
# Dataset Class
# =============================================================================

class CustomGLiNERDataset(Dataset):
    """
    Custom dataset wrapper for GLiNER training data.

    A simple wrapper around list data that provides the Dataset interface
    required by PyTorch DataLoader.

    Args:
        data (List[Dict]): List of training examples
    """

    def __init__(self, data: List[Dict[str, Any]]):
        """Initialize dataset with training data."""
        self.data = data

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the example at the given index."""
        return self.data[idx]


# =============================================================================
# Data Normalization
# =============================================================================

def normalize_entity_labels(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize all entity labels to lowercase in the dataset.

    Args:
        data: List of training examples

    Returns:
        List of examples with lowercase entity labels
    """
    normalized_data = []

    for example in data:
        normalized_example = example.copy()

        if 'ner' in example:
            normalized_ner = []
            for annotation in example['ner']:
                if len(annotation) >= 3:
                    normalized_annotation = annotation[:2] + [annotation[2].lower()]
                    if len(annotation) > 3:
                        normalized_annotation.extend(annotation[3:])
                    normalized_ner.append(normalized_annotation)
                else:
                    normalized_ner.append(annotation)
            normalized_example['ner'] = normalized_ner

        elif 'entities' in example:
            normalized_entities = []
            for entity in example['entities']:
                normalized_entity = entity.copy()
                if 'type' in entity:
                    normalized_entity['type'] = entity['type'].lower()
                normalized_entities.append(normalized_entity)
            normalized_example['entities'] = normalized_entities

        normalized_data.append(normalized_example)

    return normalized_data


# =============================================================================
# Data Loading Functions
# =============================================================================

def _load_single_file(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from a single JSON file."""
    print(f"Loading from file: {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list format, got {type(data)}")

    return data


def _needs_format_conversion(examples: List[Dict]) -> bool:
    """Check if examples need conversion to GLiNER format."""
    if not examples:
        return False

    sample_size = min(10, len(examples))
    for example in examples[:sample_size]:
        if 'sentence' in example and 'entities' in example:
            return True
    return False


def convert_to_gliner_format(examples: List[Dict]) -> List[Dict]:
    """
    Convert training examples from sentence/entities format to GLiNER format.

    Args:
        examples: List of examples in sentence/entities format

    Returns:
        List[Dict]: Examples in GLiNER format with 'text' and 'ner' fields

    Input format:
        {
            "sentence": "John works at Google.",
            "entities": [
                {"pos": [0, 4], "type": "PERSON"},
                {"pos": [14, 20], "type": "ORG"}
            ]
        }

    Output format:
        {
            "text": "John works at Google.",
            "ner": [[0, 4, "PERSON"], [14, 20, "ORG"]]
        }
    """
    converted_examples = []
    conversion_errors = 0

    for i, example in enumerate(examples):
        if 'ner' in example and 'text' in example:
            converted_examples.append(example)
            continue

        if 'sentence' not in example or 'entities' not in example:
            conversion_errors += 1
            continue

        text = example['sentence']
        ner = []

        for entity in example['entities']:
            try:
                if 'pos' in entity and 'type' in entity:
                    start, end = entity['pos']
                    entity_type = entity['type']
                    ner.append([start, end, entity_type])
                elif all(key in entity for key in ['start', 'end', 'type']):
                    ner.append([entity['start'], entity['end'], entity['type']])
                else:
                    print(f"Invalid entity format at example {i}: {entity}")
            except Exception as e:
                print(f"Error processing entity at example {i}: {e}")

        converted_examples.append({
            'text': text,
            'ner': ner
        })

    if conversion_errors > 0:
        print(f"{conversion_errors} examples skipped due to format errors")

    print(f"Converted {len(converted_examples)} examples to GLiNER format")
    return converted_examples


def load_training_data(config: object) -> List[Dict[str, Any]]:
    """
    Load training data from configuration-specified sources.

    Args:
        config: Configuration object containing data paths

    Returns:
        List[Dict]: List of training examples in GLiNER format

    Raises:
        FileNotFoundError: If specified data files don't exist
        ValueError: If data format is invalid

    Supports loading from:
    - Single JSON file (config.train_data)
    - Directory of JSON files (config.train_data_dir)
    - Automatic format conversion if needed
    """
    train_data = []

    if (hasattr(config, 'train_data_dir') and
            config.train_data_dir and
            config.train_data_dir != "none"):

        data_dir = Path(config.train_data_dir)

        if data_dir.is_dir():
            json_files = list(data_dir.glob("*.json"))

            if json_files:
                print(f"Loading from directory: {data_dir}")
                print(f"Found {len(json_files)} JSON files")

                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                train_data.extend(file_data)
                                print(f"  - {json_file.name}: {len(file_data)} examples")
                    except Exception as e:
                        print(f"Error loading {json_file.name}: {e}")
            else:
                print(f"No JSON files found in {data_dir}, trying single file")
                if hasattr(config, 'train_data') and config.train_data:
                    train_data = _load_single_file(config.train_data)
        else:
            if hasattr(config, 'train_data') and config.train_data:
                train_data = _load_single_file(config.train_data)
    else:
        if hasattr(config, 'train_data') and config.train_data:
            train_data = _load_single_file(config.train_data)

    if not train_data:
        raise ValueError("No training data could be loaded from the specified sources")

    if _needs_format_conversion(train_data):
        print("Converting to GLiNER format...")
        train_data = convert_to_gliner_format(train_data)

    train_data = normalize_entity_labels(train_data)
    random.shuffle(train_data)

    print(f"Loaded {len(train_data)} training examples")
    return train_data


# =============================================================================
# Entity Type Processing
# =============================================================================

def extract_entity_type_names(train_data: List[Dict]) -> List[str]:
    """
    Extract unique entity type names from training data.

    Args:
        train_data: List of training examples

    Returns:
        List[str]: Sorted list of unique entity type names

    Extracts entity types from both 'ner' and 'entities' formats
    and returns them in consistent sorted order for reproducibility.
    """
    entity_types = set()

    for example in train_data:
        if "ner" in example:
            for annotation in example["ner"]:
                if len(annotation) >= 3:
                    entity_types.add(annotation[2])
        elif "entities" in example:
            for entity in example["entities"]:
                if "type" in entity:
                    entity_types.add(entity["type"])

    entity_names = sorted(list(entity_types))
    return entity_names


def extract_entity_types(train_data: List[Dict]) -> List[str]:
    """
    Extract entity types for compatibility (same as extract_entity_type_names).

    Args:
        train_data: List of training examples

    Returns:
        List[str]: List of entity type names, defaults to common NER types if none found
    """
    entity_types = extract_entity_type_names(train_data)
    return entity_types if entity_types else ["PER", "ORG", "LOC", "MISC"]


def prepare_entity_type_mapping(train_data: List[Dict], min_examples: int = 50) -> Dict[int, List[int]]:
    """
    Create mapping from entity type IDs to example indices for contrastive sampling.

    Args:
        train_data: List of training examples
        min_examples: Minimum number of examples required per entity type

    Returns:
        Dict[int, List[int]]: Mapping from entity type ID to list of example indices
                              Only includes types with at least min_examples

    This mapping is used by ContrastiveBatchSampler to ensure batches contain
    multiple examples of the same entity types for effective contrastive learning.
    """
    unique_types = set()
    for example in train_data:
        if "ner" in example:
            for annotation in example["ner"]:
                if len(annotation) >= 3:
                    unique_types.add(annotation[2])

    type_to_id = {type_name: i for i, type_name in enumerate(sorted(unique_types))}
    entity_type_to_example_indices = {}

    for example_idx, example in enumerate(train_data):
        example_entity_types = set()

        if "ner" in example:
            for annotation in example["ner"]:
                if len(annotation) >= 3:
                    example_entity_types.add(annotation[2])

        for entity_type_str in example_entity_types:
            type_id = type_to_id[entity_type_str]
            if type_id not in entity_type_to_example_indices:
                entity_type_to_example_indices[type_id] = []
            entity_type_to_example_indices[type_id].append(example_idx)

    filtered_mapping = {
        type_id: example_indices
        for type_id, example_indices in entity_type_to_example_indices.items()
        if len(example_indices) >= min_examples
    }

    print(f"Entity type mapping: {len(filtered_mapping)} types with ≥{min_examples} examples")
    for type_id, indices in list(filtered_mapping.items())[:5]:
        type_name = [name for name, id_ in type_to_id.items() if id_ == type_id][0]
        print(f"  {type_name}: {len(indices)} examples")

    return filtered_mapping


# =============================================================================
# Data Filtering Functions
# =============================================================================

def get_entity_type_statistics(train_data: List[Dict]) -> Dict[str, int]:
    """
    Get statistics about entity type frequencies in the dataset.

    Args:
        train_data: List of training examples

    Returns:
        Dict[str, int]: Mapping from entity type to count
    """
    type_counts = Counter()

    for example in train_data:
        if 'ner' in example:
            for ner in example['ner']:
                if len(ner) >= 3:
                    type_counts[ner[2]] += 1

    return dict(type_counts)


def filter_common_types(train_data: List[Dict], min_examples: int = 100) -> List[Dict]:
    """
    Filter training data to only include entity types that appear frequently.

    Args:
        train_data: List of training examples
        min_examples: Minimum number of examples required for an entity type

    Returns:
        List of filtered training examples
    """
    print(f"Filtering entity types (min {min_examples} examples)...")

    type_counts = get_entity_type_statistics(train_data)
    common_types = {t for t, c in type_counts.items() if c >= min_examples}

    print(f"Entity types before filtering: {len(type_counts)}")
    print(f"Entity types after filtering (≥{min_examples} examples): {len(common_types)}")

    top_types = Counter(type_counts).most_common(10)
    print("\nTop 10 most common entity types:")
    for entity_type, count in top_types:
        print(f"  {entity_type}: {count} examples")

    filtered_data = []
    for example in train_data:
        if 'ner' in example:
            filtered_ner = []
            for ner in example['ner']:
                if len(ner) >= 3 and ner[2] in common_types:
                    filtered_ner.append(ner)

            if filtered_ner:
                filtered_example = example.copy()
                filtered_example['ner'] = filtered_ner
                filtered_data.append(filtered_example)

    print(f"\nTraining examples before filtering: {len(train_data)}")
    print(f"Training examples after filtering: {len(filtered_data)}")

    return filtered_data


def score_example_difficulty(example: Dict, entity_type_frequencies: Dict[str, float]) -> float:
    """
    Score example difficulty for curriculum learning.
    Lower scores = easier examples.

    Factors:
    - Text length (shorter = easier)
    - Number of entities (fewer = easier)
    - Entity type rarity (common types = easier)
    - Entity span length (shorter spans = easier)
    - Entity density (sparse = easier)
    """
    text = example.get('text', example.get('tokenized_text', ''))
    if isinstance(text, list):
        text = ' '.join(text)

    text_length = len(text.split())
    entities = example.get('ner', [])

    length_score = min(text_length / 256, 1.0)

    entity_count_score = min(len(entities) / 10, 1.0)

    rarity_scores = []
    for ent in entities:
        if len(ent) >= 3:
            entity_type = ent[2]
            freq = entity_type_frequencies.get(entity_type, 0)
            rarity = 1.0 - min(freq / max(entity_type_frequencies.values()), 1.0)
            rarity_scores.append(rarity)

    avg_rarity = sum(rarity_scores) / len(rarity_scores) if rarity_scores else 0.0

    density = len(entities) / max(text_length, 1)
    density_score = min(density * 10, 1.0)

    difficulty = (
            0.2 * length_score +
            0.3 * entity_count_score +
            0.3 * avg_rarity +
            0.2 * density_score
    )

    return difficulty


class CurriculumDataset(Dataset):
    """Dataset wrapper that implements curriculum learning"""

    def __init__(self, data: List[Dict], start_ratio: float = 0.3):
        self.full_data = data
        self.start_ratio = start_ratio
        self.current_ratio = start_ratio

        entity_frequencies = self._compute_entity_frequencies()
        self.difficulties = [
            score_example_difficulty(ex, entity_frequencies)
            for ex in self.full_data
        ]

        self.sorted_indices = sorted(
            range(len(self.full_data)),
            key=lambda i: self.difficulties[i]
        )

        self._update_active_data()

    def _compute_entity_frequencies(self) -> Dict[str, float]:
        """Compute normalized entity type frequencies"""
        type_counts = {}
        for example in self.full_data:
            for ent in example.get('ner', []):
                if len(ent) >= 3:
                    entity_type = ent[2]
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        total = sum(type_counts.values())
        return {k: v / total for k, v in type_counts.items()}

    def _update_active_data(self):
        """Update the active subset based on current ratio"""
        n_examples = int(len(self.full_data) * self.current_ratio)
        n_examples = max(1000, min(n_examples, len(self.full_data)))

        active_indices = self.sorted_indices[:n_examples]
        self.active_data = [self.full_data[i] for i in active_indices]

        indices = list(range(len(self.active_data)))
        random.shuffle(indices)
        self.active_data = [self.active_data[i] for i in indices]

    def set_curriculum_ratio(self, ratio: float):
        """Update curriculum progress (0.0 to 1.0)"""
        self.current_ratio = max(self.start_ratio, min(ratio, 1.0))
        self._update_active_data()

    def __len__(self):
        return len(self.active_data)

    def __getitem__(self, idx):
        return self.active_data[idx]