import gc
import warnings
import requests
import torch
from huggingface_hub import configure_http_backend
from typing import List, Dict, Set
from logger import CustomLogger
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# -------------------------------------------------------------------
# Backend Factory to Avoid Certificate Issues
# -------------------------------------------------------------------
def backend_factory() -> requests.Session:
    """Create a Requests session that disables SSL verification."""
    session = requests.Session()
    session.verify = False
    return session

warnings.filterwarnings('ignore')
configure_http_backend(backend_factory=backend_factory)


# -------------------------------------------------------------------
# Encoding for Cosine Similarity (CPU)
# -------------------------------------------------------------------
class EntityTypeSimilarityManager:
    """Manages semantic similarity between entity types"""

    def __init__(
            self,
            train_data: List[Dict],
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold: float = 0.7
    ):
        self.entity_types = self._extract_entity_labels(train_data)
        self.entity_type_to_id = {et: idx for idx, et in enumerate(self.entity_types)}
        self.similarity_threshold = similarity_threshold
        self._logger = CustomLogger(name="EntityTypeSimilarityManager")

        self._logger.info(f"Loading sentence transformer: {model_name}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            self._logger.warning("CUDA not available, falling back to CPU")

        self.sentence_model = SentenceTransformer(model_name, device=self._device)
        self._compute_similarity_matrix()

    def _extract_entity_labels(self, train_data: List[Dict]) -> List[str]:
        entity_types = set()
        for example in train_data:
            for _, _, entity_type in example.get('ner', []):
                entity_types.add(entity_type)

        entity_types = sorted(entity_types)
        return entity_types

    def _compute_similarity_matrix(self):
        """Compute semantic similarity matrix between entity types"""
        self._logger.info("Computing semantic similarity between entity types...")

        entity_type_embeddings = self.sentence_model.encode(
            self.entity_types,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        self.similarity_matrix = cosine_similarity(entity_type_embeddings)

        if self._device == "cuda":
            del entity_type_embeddings
            del self.sentence_model

            gc.collect()

            torch.cuda.empty_cache()

            self._logger.info("GPU memory freed successfully")

            if torch.cuda.is_available():
                self._logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
                self._logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    def _create_random_similarity_matrix(self):
        """Create random similarity matrix as fallback"""
        self._logger.warning("Creating random similarity matrix")
        self.similarity_matrix = np.random.rand(len(self.entity_types), len(self.entity_types))
        self.similarity_matrix = (self.similarity_matrix + self.similarity_matrix.T) / 2
        np.fill_diagonal(self.similarity_matrix, 1.0)

    def get_similarity(self, type1_id: int, type2_id: int) -> float:
        """Get similarity between two entity type IDs"""
        return float(self.similarity_matrix[type1_id, type2_id])

    def get_entity_type_to_id(self) -> Dict[str, int]:
        """Get entity type IDs dictionary"""
        return self.entity_type_to_id