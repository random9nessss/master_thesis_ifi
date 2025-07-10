import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler
from logger import CustomLogger
from typing import List, Tuple, Set, Dict


# -------------------------------------------------------------------
# Contrastive Batch Sampler
# -------------------------------------------------------------------
class StratifiedContrastiveBatchSampler(Sampler):
    """
    Optimized batch sampler for batch size 128 with sophisticated composition.

    Creates batches with optimal ratios:
    - Strong anchor representation (many examples)
    - Gradient of similarities (similar → dissimilar)
    - ~10% positive pairs for effective contrastive learning
    """

    def __init__(
            self,
            dataset: 'EntityDataset',
            similarity_manager: 'EntityTypeSimilarityManager',
            batch_size: int = 64,
            anchor_examples: int = 16,
            similar_type_examples: int = 10,
            medium_type_examples: int = 8,
            dissimilar_type_examples: int = 4,
            n_similar_types: int = 2,
            n_medium_types: int = 2,
            n_dissimilar_types: int = 3,
            similarity_strata: List[Tuple[float, float]] = None,
            drop_last: bool = True,
            similarity_temperature: float = 0.5,
            balance_strategy: str = 'adaptive',
            min_dissimilar_threshold: float = 0.4,
            force_diversity: bool = True
    ):
        """
        Args:
            dataset: EntityDataset instance
            similarity_manager: Manager containing similarity matrix
            batch_size: Total batch size (default 128)
            anchor_examples: Number of examples for anchor type (20)
            similar_type_examples: Examples per similar type (15)
            medium_type_examples: Examples per medium similarity type (12)
            dissimilar_type_examples: Examples per dissimilar type (4)
            n_similar_types: Number of similar types to include (3)
            n_medium_types: Number of medium similarity types (4)
            n_dissimilar_types: Number of dissimilar types (4)
            similarity_strata: Similarity ranges for categorization
            drop_last: Whether to drop incomplete batches
            similarity_temperature: Temperature for probability sampling
            balance_strategy: 'fixed' for exact counts, 'adaptive' for flexibility
            min_dissimilar_threshold: Maximum similarity for dissimilar types
            force_diversity: Force selection of truly dissimilar types
        """
        self.dataset = dataset
        self.similarity_manager = similarity_manager
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.similarity_temperature = similarity_temperature
        self.balance_strategy = balance_strategy
        self.min_dissimilar_threshold = min_dissimilar_threshold
        self.force_diversity = force_diversity

        self.anchor_examples = anchor_examples
        self.similar_type_examples = similar_type_examples
        self.medium_type_examples = medium_type_examples
        self.dissimilar_type_examples = dissimilar_type_examples
        self.n_similar_types = n_similar_types
        self.n_medium_types = n_medium_types
        self.n_dissimilar_types = n_dissimilar_types

        if similarity_strata is None:
            self.similarity_strata = [
                (0.7, 1.0),  # Very high similarity (near synonyms/duplicates)
                (0.5, 0.7),  # Medium similarity
                (0.2, 0.5),  # Low similarity
                (0.0, 0.2)   # Very low similarity (true negatives)
            ]
        else:
            self.similarity_strata = similarity_strata

        self._logger = CustomLogger(name='StratifiedContrastiveBatchSampler')
        self._precompute_stratified_relationships()


    def _precompute_stratified_relationships(self):
        """Precompute entity type relationships stratified by similarity levels."""
        self.stratified_types = defaultdict(lambda: defaultdict(list))

        for i, type1 in enumerate(self.similarity_manager.entity_types):
            type1_id = self.similarity_manager.entity_type_to_id[type1]
            for j, type2 in enumerate(self.similarity_manager.entity_types):
                if i != j:
                    type2_id = self.similarity_manager.entity_type_to_id[type2]
                    similarity = self.similarity_manager.get_similarity(type1_id, type2_id)

                    for stratum_idx, (min_sim, max_sim) in enumerate(self.similarity_strata):
                        if min_sim <= similarity < max_sim:
                            self.stratified_types[type1][stratum_idx].append((type2, similarity))
                            break

            for stratum_idx in self.stratified_types[type1]:
                self.stratified_types[type1][stratum_idx].sort(key=lambda x: x[1], reverse=True)

    def _sample_from_stratum(
            self,
            anchor_type: str,
            stratum_idx: int,
            n_types: int,
            excluded_types: Set[str],
            min_examples_required: int
    ) -> List[str]:
        """Sample entity types from a specific similarity stratum."""
        available_types = [
            (t, sim) for t, sim in self.stratified_types[anchor_type][stratum_idx]
            if t not in excluded_types and len(self.dataset.entities_by_type[t]) >= min_examples_required
        ]

        if not available_types:
            return []

        if len(available_types) <= n_types:
            return [t for t, _ in available_types]

        types, similarities = zip(*available_types)
        similarities = np.array(similarities)

        if stratum_idx >= 2:
            weights = np.exp(-similarities / self.similarity_temperature)
        else:
            weights = np.exp(similarities / self.similarity_temperature)

        probabilities = weights / weights.sum()

        selected_indices = np.random.choice(
            len(types),
            size=min(n_types, len(types)),
            replace=False,
            p=probabilities
        )

        return [types[i] for i in selected_indices]

    def _sample_from_type(self, entity_type: str, n_samples: int) -> List[int]:
        """Sample n examples from a specific entity type."""
        candidates = self.dataset.entities_by_type[entity_type]

        if len(candidates) <= n_samples:
            return candidates.copy()

        return random.sample(candidates, n_samples)

    def _ensure_true_negatives(
            self,
            anchor_type: str,
            selected_types: Set[str],
            n_required: int
    ) -> List[str]:
        """Ensure we have truly dissimilar types as negatives"""
        anchor_id = self.similarity_manager.entity_type_to_id[anchor_type]

        true_negatives = []
        for entity_type in self.similarity_manager.entity_types:
            if entity_type not in selected_types and entity_type != anchor_type:
                type_id = self.similarity_manager.entity_type_to_id[entity_type]
                similarity = self.similarity_manager.get_similarity(anchor_id, type_id)

                if similarity < self.min_dissimilar_threshold:
                    if len(self.dataset.entities_by_type[entity_type]) >= self.dissimilar_type_examples:
                        true_negatives.append((entity_type, similarity))

        true_negatives.sort(key=lambda x: x[1])

        return [t for t, _ in true_negatives[:n_required]]

    def _create_stratified_batch(self) -> List[int]:
        """Create a batch with optimized composition for batch size 128."""
        batch_indices = []
        selected_types = set()
        type_to_indices = {}

        valid_anchor_types = [
            et for et, examples in self.dataset.entities_by_type.items()
            if len(examples) >= self.anchor_examples
        ]

        if not valid_anchor_types:
            return []

        anchor_type = random.choice(valid_anchor_types)
        selected_types.add(anchor_type)

        anchor_indices = self._sample_from_type(anchor_type, self.anchor_examples)
        batch_indices.extend(anchor_indices)
        type_to_indices[anchor_type] = anchor_indices

        # High Similarity Entities
        similar_types = self._sample_from_stratum(
            anchor_type, 0, self.n_similar_types, selected_types, self.similar_type_examples
        )

        for entity_type in similar_types:
            indices = self._sample_from_type(entity_type, self.similar_type_examples)
            batch_indices.extend(indices)
            selected_types.add(entity_type)
            type_to_indices[entity_type] = indices

        # Medium Similarity Entities
        medium_types = self._sample_from_stratum(
            anchor_type, 1, self.n_medium_types, selected_types, self.medium_type_examples
        )

        for entity_type in medium_types:
            indices = self._sample_from_type(entity_type, self.medium_type_examples)
            batch_indices.extend(indices)
            selected_types.add(entity_type)
            type_to_indices[entity_type] = indices

        # Dissimilar Entities
        dissimilar_types = []

        if self.force_diversity:
            true_negatives = self._ensure_true_negatives(
                anchor_type, selected_types, self.n_dissimilar_types
            )
            dissimilar_types.extend(true_negatives)

        if len(dissimilar_types) < self.n_dissimilar_types:
            needed = self.n_dissimilar_types - len(dissimilar_types)

            low_sim_types = self._sample_from_stratum(
                anchor_type, 2, needed // 2, selected_types, self.dissimilar_type_examples
            )
            dissimilar_types.extend(low_sim_types)

            very_low_sim_types = self._sample_from_stratum(
                anchor_type, 3, needed - len(low_sim_types), selected_types, self.dissimilar_type_examples
            )
            dissimilar_types.extend(very_low_sim_types)

        for entity_type in dissimilar_types[:self.n_dissimilar_types]:
            indices = self._sample_from_type(entity_type, self.dissimilar_type_examples)
            batch_indices.extend(indices)
            selected_types.add(entity_type)
            type_to_indices[entity_type] = indices

        if self.balance_strategy == 'adaptive':
            while len(batch_indices) < self.batch_size:
                candidate_types = [anchor_type] + similar_types + dissimilar_types
                for entity_type in candidate_types:
                    if entity_type in self.dataset.entities_by_type:
                        available = set(self.dataset.entities_by_type[entity_type]) - set(
                            type_to_indices.get(entity_type, []))
                        if available and len(batch_indices) < self.batch_size:
                            extra = random.choice(list(available))
                            batch_indices.append(extra)
                            if entity_type in type_to_indices:
                                type_to_indices[entity_type].append(extra)

            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]

        return batch_indices

    def __iter__(self):
        """Generate stratified batches."""
        n_batches = len(self)

        for _ in range(n_batches):
            batch = self._create_stratified_batch()

            min_batch_size = self.batch_size // 2 if not self.drop_last else self.batch_size
            if batch and len(batch) >= min_batch_size:
                yield batch

    def __len__(self):
        """Estimate number of batches."""
        total_examples = len(self.dataset)
        if self.drop_last:
            return total_examples // self.batch_size
        else:
            return (total_examples + self.batch_size - 1) // self.batch_size