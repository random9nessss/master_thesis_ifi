import re
import json
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticDiversity:
    """
    Class to compute semantic diversity of email texts using sentence embeddings.

    The method mirrors the TF–IDF approach: it preprocesses texts, encodes them with a
    transformer model, computes pairwise cosine similarity, and defines semantic diversity
    as 1 minus the average cosine similarity.
    """

    def __init__(self, file_path: str, field: str = "body", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the SentenceEmbeddingDiversity object.

        Args:
            file_path (str): Path to the JSON file containing email data.
            field (str): The field to analyze from each email (either "body" or "subject").
            model_name (str): The transformer model to use for sentence embeddings.
        Raises:
            ValueError: If field is not "body" or "subject".
        """
        if field not in ["body", "subject"]:
            raise ValueError("field must be either 'body' or 'subject'.")
        self.file_path = file_path
        self.field = field
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self.emails = pd.read_json(self.file_path)

    def _extract_texts(self):
        """
        Extract the specified field (body or subject) from all email objects.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []

        for idx, email in self.emails.iterrows():
            email = email.get("email_chain")[0]

            if self.field in email and isinstance(email[self.field], str) and email[self.field].strip():
                texts.append(email[self.field])
        return texts

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess a text string:
          - Convert to lowercase.
          - Remove excessive whitespace.
          - Ensure decimals remain intact (e.g., "66 . 50" becomes "66.50").

        Args:
            text (str): The text to preprocess.
        Returns:
            str: The preprocessed text.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        return text

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a list of texts.

        Args:
            texts (List[str]): The list of texts.
        Returns:
            List[str]: A list of preprocessed texts.
        """
        return [self._preprocess_text(t) for t in texts if t.strip()]

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts to embeddings using the transformer model.

        Args:
            texts (List[str]): The list of texts.
        Returns:
            np.ndarray: Array of embeddings.
        """
        inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

    def compute_overall_semantic_diversity(self) -> float:
        """
        Compute semantic diversity as 1 minus the average cosine similarity among all texts.

        Returns:
            float: The semantic diversity score.
        """
        texts = self._extract_texts()
        processed_texts = self._preprocess_texts(texts)
        if len(processed_texts) < 2:
            return 0.0

        embeddings = self._encode_texts(processed_texts)
        sim_matrix = cosine_similarity(embeddings)
        n = len(processed_texts)
        sims = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        avg_similarity = np.mean(sims)
        diversity = 1 - avg_similarity
        return diversity


# -------------------------------------------------------------------
# CSV Class for ENRON Emails
# -------------------------------------------------------------------
class SemanticDiversityCSV:
    """
    Modified SemanticDiversity class to work with CSV files containing a parsed_content column.

    Computes semantic diversity of email texts using sentence embeddings.
    The method mirrors the TF–IDF approach: it preprocesses texts, encodes them with a
    transformer model, computes pairwise cosine similarity, and defines semantic diversity
    as 1 minus the average cosine similarity.
    """

    def __init__(self, file_path: str,
                 content_column: str = "parsed_content",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the SemanticDiversityCSV object.

        Args:
            file_path (str): Path to the CSV file containing email data.
            content_column (str): The column name containing the email content.
            model_name (str): The transformer model to use for sentence embeddings.
        Raises:
            ValueError: If the content_column is not found in the CSV file.
        """
        self.file_path = file_path
        self.content_column = content_column
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self.data = pd.read_csv(self.file_path)

        if self.content_column not in self.data.columns:
            raise ValueError(f"Column '{self.content_column}' not found in the CSV file.")

    def _extract_texts(self) -> List[str]:
        """
        Extract content from the specified column.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        for _, row in self.data.iterrows():
            content = row.get(self.content_column)
            if isinstance(content, str) and content.strip():
                texts.append(content)
        return texts

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess a text string:
          - Convert to lowercase.
          - Remove excessive whitespace.
          - Ensure decimals remain intact (e.g., "66 . 50" becomes "66.50").

        Args:
            text (str): The text to preprocess.
        Returns:
            str: The preprocessed text.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        return text

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a list of texts.

        Args:
            texts (List[str]): The list of texts.
        Returns:
            List[str]: A list of preprocessed texts.
        """
        return [self._preprocess_text(t) for t in texts if t.strip()]

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts to embeddings using the transformer model.

        Args:
            texts (List[str]): The list of texts.
        Returns:
            np.ndarray: Array of embeddings.
        """
        inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

    def compute_overall_semantic_diversity(self) -> float:
        """
        Compute semantic diversity as 1 minus the average cosine similarity among all texts.

        Returns:
            float: The semantic diversity score.
        """
        texts = self._extract_texts()
        processed_texts = self._preprocess_texts(texts)
        if len(processed_texts) < 2:
            return 0.0

        embeddings = self._encode_texts(processed_texts)
        sim_matrix = cosine_similarity(embeddings)
        n = len(processed_texts)
        sims = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        avg_similarity = np.mean(sims)
        diversity = 1 - avg_similarity
        return diversity