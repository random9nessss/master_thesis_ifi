from tqdm import tqdm
from logger import CustomLogger
from dataclasses import dataclass
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Dict


# -------------------------------------------------------------------
# Entity Dataset - Focused on Entity Text Only
# -------------------------------------------------------------------
@dataclass
class EntityExample:
    """Represents a single entity with its type"""
    entity_text: str
    entity_type: str
    entity_type_id: int
    example_id: int = 0


class EntityDataset(Dataset):
    """Dataset for entity texts with their types - NO CONTEXT"""

    def __init__(
            self,
            gliner_data: List[Dict],
            tokenizer,
            entity_type_to_id: Dict[str, int],
            max_length: int = 64,
            min_entity_length: int = 2,
            max_entity_length: int = 50
    ):
        self.tokenizer = tokenizer
        self.entity_type_to_id = entity_type_to_id
        self.max_length = max_length
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length

        self.entities = []
        self.entities_by_type = defaultdict(list)
        self._extract_entities_from_gliner_data(gliner_data)
        self._logger = CustomLogger(name="EntityDataset")
        self._logger.info(f"Total entities extracted: {len(self.entities)}")

    def _extract_entities_from_gliner_data(self, gliner_data: List[Dict]):
        """Extract entity texts from GLiNER format data"""

        type_counters = defaultdict(int)
        example_id = 0
        unique_entities = defaultdict(set)

        for example in tqdm(gliner_data, desc="Extracting entities"):
            text = example.get('tokenized_text', '')
            ner_tags = example.get('ner', [])

            if not text or not ner_tags:
                continue

            for ner_tag in ner_tags:
                if len(ner_tag) < 3:
                    continue

                start_char, end_char, entity_type = ner_tag[0], ner_tag[1], ner_tag[2]

                if entity_type not in self.entity_type_to_id:
                    continue

                entity_text = " ".join(text[start_char:end_char+1]).strip().replace(", ", ",").replace(". ", ".")

                if len(entity_text) < self.min_entity_length or len(entity_text) > self.max_entity_length:
                    continue

                if entity_text in unique_entities[entity_type]:
                    continue

                unique_entities[entity_type].add(entity_text)

                entity_example = EntityExample(
                    entity_text=entity_text,
                    entity_type=entity_type,
                    entity_type_id=self.entity_type_to_id[entity_type],
                    example_id=example_id
                )

                self.entities.append(entity_example)
                self.entities_by_type[entity_type].append(example_id)
                type_counters[entity_type] += 1
                example_id += 1

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        entity = self.entities[idx]

        encoding = self.tokenizer(
            entity.entity_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'entity_type_id': entity.entity_type_id,
            'entity_type': entity.entity_type,
            'entity_text': entity.entity_text,
            'example_id': entity.example_id
        }