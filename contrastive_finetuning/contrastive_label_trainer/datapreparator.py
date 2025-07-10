import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from logger import CustomLogger


class DataPreparator:

    def __init__(self, train_data_dir: str):
        self._train_data_dir = Path(train_data_dir)
        self._logger = CustomLogger(name="DataPreparator")

    # -------------------------------------------------------------------
    # Training Data Loading
    # -------------------------------------------------------------------
    def load_training_data(self) -> List[Dict[str, Any]]:
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
        json_files = list(self._train_data_dir.glob("*.json"))

        if json_files:
            self._logger.info(f"Loading from directory: {self._train_data_dir}")
            self._logger.info(f"Found {len(json_files)} JSON files")

            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            train_data.extend(file_data)
                            self._logger.ok(f"{json_file.name}: {len(file_data)} examples")
                except Exception as e:
                    self._logger.error(f"Error loading {json_file.name}: {e}")

            # Format Conversion to GLiNER
            if self._needs_format_conversion(train_data):
                self._logger.info(f"Converting to GLiNER format")
                train_data = self._convert_to_gliner_format(train_data)

            # Label normalization
            train_data = self._normalize_entity_labels(train_data)
            random.shuffle(train_data)

            self._logger.ok(f"Loaded {len(train_data)} training examples")
            return train_data

        else:
            self._logger.error(f"No training data could be loaded from {self._train_data_dir}")
            return []

    # -------------------------------------------------------------------
    # GLiNER Format Conversion
    # -------------------------------------------------------------------
    def _needs_format_conversion(self, examples: List[Dict]) -> bool:
        """Check if examples need conversion to GLiNER format."""
        if not examples:
            return False

        sample_size = min(10, len(examples))
        for example in examples[:sample_size]:
            if 'sentence' in example and 'entities' in example:
                return True
        return False

    # -------------------------------------------------------------------
    # GLiNER Format Conversion
    # -------------------------------------------------------------------
    def _convert_to_gliner_format(self, examples: List[Dict]) -> List[Dict]:
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
                        self._logger.warning(f"Invalid entity format at example {i}: {entity}")
                except Exception as e:
                    self._logger.error(f"Error parsing example {i}: {e}")

            converted_examples.append({
                'text': text,
                'ner': ner
            })

        if conversion_errors > 0:
            self._logger.warning(f"Conversion errors: {conversion_errors} examples skipped due to format errors.")

        self._logger.info(f"Converted {len(converted_examples)} examples to GLiNER format")
        return converted_examples

    # -------------------------------------------------------------------
    # Label Normalization
    # -------------------------------------------------------------------
    def _normalize_entity_labels(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    # -------------------------------------------------------------------
    # Filtering Entities
    # -------------------------------------------------------------------
    def filter_common_types(self, train_data: List[Dict], min_examples: int = 500) -> List[Dict]:
        """Filter training data to only include frequently occurring entity types"""

        self._logger.info(f"Filtering entity types (min {min_examples} examples)...")

        type_counts = Counter()
        for example in train_data:
            for _, _, entity_type in example.get('ner', []):
                type_counts[entity_type] += 1

        common_types = {t for t, c in type_counts.items() if c >= min_examples}

        self._logger.info(f"Entity types before filtering: {len(type_counts)}")
        self._logger.info(f"Entity types after filtering (≥{min_examples} examples): {len(common_types)}")

        top_types = type_counts.most_common(10)
        self._logger.info("Top 10 most common entity types:")
        for entity_type, count in top_types:
            self._logger.info(f"  {entity_type}: {count} examples")

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

        self._logger.info(f"Training examples before filtering: {len(train_data)}")
        self._logger.info(f"Training examples after filtering: {len(filtered_data)}")

        return filtered_data