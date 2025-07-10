import os
import json
import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk import ngrams
from typing import List, Dict, Tuple, Set
from collections import Counter


class LexicalDiversity:
    """
    A class to compute lexical diversity of email texts using multiple methods:
    1. Type-Token Ratio (TTR) at unigram, bigram, and trigram levels
    2. Distinct-1 and Distinct-2 metrics from Li et al. (2016)
    3. Normalized type-token ratio to account for text length

    This allows for a more comprehensive evaluation of lexical diversity.
    """

    def __init__(self, file_path: str, field: str = "body"):
        """
        Initialize the EnhancedLexicalDiversity object.

        Args:
            file_path (str): Path to the JSON file containing email data.
            field (str): The field to analyze from each email (either "body" or "subject").
        """
        if field not in ["body", "subject"]:
            raise ValueError("field must be either 'body' or 'subject'.")
        self.file_path = file_path
        self.field = field
        self.emails = pd.read_json(self.file_path) if self.file_path.endswith("json") else pd.read_csv(self.file_path)
        self.tokenizer = WordPunctTokenizer()
        self.punctuation = set(string.punctuation)

    def _extract_texts(self) -> List[str]:
        """
        Extract the specified field (body or subject) from all email objects.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        for idx, row in self.emails.iterrows():
            email_chain = row.get("email_chain")

            if email_chain and isinstance(email_chain, list):
                # For all emails in the chain, not just the first one
                for email in email_chain:
                    if (self.field in email and
                            isinstance(email[self.field], str) and
                            email[self.field].strip()):
                        texts.append(email[self.field])
        return texts

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text and filter out punctuation.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[str]: Filtered tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        return [t.lower() for t in tokens if t not in self.punctuation]

    @staticmethod
    def _ttr(tokens: List[str]) -> float:
        """
        Computes the Type-Token Ratio (TTR) for a list of tokens.

        Args:
            tokens (List[str]): A list of tokens.

        Returns:
            float: The TTR value (ratio of unique tokens to total tokens).
        """
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    # -------------------------------------------------------------------
    # Traditional Type Token Ratio
    # -------------------------------------------------------------------
    def compute_traditional_ttr(self, text: str) -> Dict[str, float]:
        """
        Computes traditional TTR metrics at unigram, bigram, and trigram levels.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: A dictionary with TTR values for different n-gram levels.
        """
        tokens = self._tokenize(text)

        if not tokens:
            return {"ttr_unigram": 0.0, "ttr_bigram": 0.0, "ttr_trigram": 0.0, "ttr_avg": 0.0}

        unigrams = tokens
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))

        ttr_uni = self._ttr(unigrams)
        ttr_bi = self._ttr(bigrams) if bigrams else 0.0
        ttr_tri = self._ttr(trigrams) if trigrams else 0.0
        ttr_avg = (ttr_uni + ttr_bi + ttr_tri) / 3

        return {
            "ttr_unigram": ttr_uni,
            "ttr_bigram": ttr_bi,
            "ttr_trigram": ttr_tri,
            "ttr_avg": ttr_avg
        }

    # -------------------------------------------------------------------
    # Distinct N Diversity Metric (Li et. al, 2016)
    # -------------------------------------------------------------------
    def compute_li_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Computes diversity metrics as defined in Li et al. (2016):
        distinct-1 and distinct-2 (ratio of unique unigrams/bigrams to total)

        Args:
            texts (List[str]): List of texts to analyze.

        Returns:
            Dict[str, float]: Dictionary with diversity metrics.
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))

        if not all_tokens:
            return {"distinct_1": 0.0, "distinct_2": 0.0}

        unigrams = all_tokens
        bigrams = list(ngrams(all_tokens, 2))
        trigrams = list(ngrams(all_tokens, 3))

        distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0.0

        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3
        }

    # -------------------------------------------------------------------
    # Normalized Type Token Ratio
    # -------------------------------------------------------------------
    def compute_normalized_ttr(self, text: str, window_size: int = 25, step_size: int = 10) -> float:
        """
        Computes a normalized TTR that accounts for text length by using a sliding window.

        Args:
            text (str): Text to analyze
            window_size (int): Size of the sliding window

        Returns:
            float: Normalized TTR value
        """
        tokens = self._tokenize(text)

        if len(tokens) < window_size:
            return self._ttr(tokens)

        ttrs = []
        for i in range(0, len(tokens) - window_size + 1, step_size):
            window = tokens[i:i + window_size]
            ttrs.append(self._ttr(window))

        return sum(ttrs) / len(ttrs) if ttrs else 0.0

    def compute_diversity_metrics(self) -> Dict[str, float]:
        """
        Computes a comprehensive set of lexical diversity metrics.

        Returns:
            Dict[str, float]: Dictionary with all diversity metrics.
        """
        texts = self._extract_texts()
        if not texts:
            return {
                "ttr_avg": 0.0,
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "distinct_3": 0.0,
                "normalized_ttr": 0.0
            }

        aggregated_text = " ".join(texts)
        traditional_metrics = self.compute_traditional_ttr(aggregated_text)

        li_metrics = self.compute_li_diversity_metrics(texts)

        normalized_ttr = self.compute_normalized_ttr(aggregated_text)

        all_metrics = {
            **traditional_metrics,
            **li_metrics,
            "normalized_ttr": normalized_ttr
        }

        return all_metrics


# -------------------------------------------------------------------
# CSV Class for ENRON Emails
# -------------------------------------------------------------------
class LexicalDiversityCSV:
    """
    A modified version of LexicalDiversity class to compute lexical diversity metrics
    for a CSV file with a 'parsed_content' column containing email text.

    Computes:
    1. Type-Token Ratio (TTR) at unigram, bigram, and trigram levels
    2. Distinct-1, Distinct-2, and Distinct-3 metrics from Li et al. (2016)
    3. Normalized type-token ratio to account for text length
    """

    def __init__(self, file_path: str, content_column: str = "parsed_content"):
        """
        Initialize the LexicalDiversityCSV object.

        Args:
            file_path (str): Path to the CSV file containing email data.
            content_column (str): The column name containing the email content.
        """
        self.file_path = file_path
        self.content_column = content_column
        self.data = pd.read_csv(self.file_path)

        if self.content_column not in self.data.columns:
            raise ValueError(f"Column '{self.content_column}' not found in the CSV file.")

        self.tokenizer = WordPunctTokenizer()
        self.punctuation = set(string.punctuation)

    def _extract_texts(self) -> List[str]:
        """
        Extract the content from the specified column.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        for _, row in self.data.iterrows():
            content = row.get(self.content_column)
            if isinstance(content, str) and content.strip():
                texts.append(content)
        return texts

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text and filter out punctuation.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[str]: Filtered tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        return [t.lower() for t in tokens if t not in self.punctuation]

    @staticmethod
    def _ttr(tokens: List[str]) -> float:
        """
        Computes the Type-Token Ratio (TTR) for a list of tokens.

        Args:
            tokens (List[str]): A list of tokens.

        Returns:
            float: The TTR value (ratio of unique tokens to total tokens).
        """
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    # -------------------------------------------------------------------
    # Traditional Type Token Ratio
    # -------------------------------------------------------------------
    def compute_traditional_ttr(self, text: str) -> Dict[str, float]:
        """
        Computes traditional TTR metrics at unigram, bigram, and trigram levels.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: A dictionary with TTR values for different n-gram levels.
        """
        tokens = self._tokenize(text)

        if not tokens:
            return {"ttr_unigram": 0.0, "ttr_bigram": 0.0, "ttr_trigram": 0.0, "ttr_avg": 0.0}

        unigrams = tokens
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))

        ttr_uni = self._ttr(unigrams)
        ttr_bi = self._ttr(bigrams) if bigrams else 0.0
        ttr_tri = self._ttr(trigrams) if trigrams else 0.0
        ttr_avg = (ttr_uni + ttr_bi + ttr_tri) / 3

        return {
            "ttr_unigram": ttr_uni,
            "ttr_bigram": ttr_bi,
            "ttr_trigram": ttr_tri,
            "ttr_avg": ttr_avg
        }

    # -------------------------------------------------------------------
    # Distinct N Diversity Metric (Li et. al, 2016)
    # -------------------------------------------------------------------
    def compute_li_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Computes diversity metrics as defined in Li et al. (2016):
        distinct-1, distinct-2, and distinct-3 (ratio of unique unigrams/bigrams/trigrams to total)

        Args:
            texts (List[str]): List of texts to analyze.

        Returns:
            Dict[str, float]: Dictionary with diversity metrics.
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))

        if not all_tokens:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0}

        unigrams = all_tokens
        bigrams = list(ngrams(all_tokens, 2))
        trigrams = list(ngrams(all_tokens, 3))

        distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0.0

        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3
        }

    # -------------------------------------------------------------------
    # Normalized Type Token Ratio
    # -------------------------------------------------------------------
    def compute_normalized_ttr(self, text: str, window_size: int = 25, step_size: int = 10) -> float:
        """
        Computes a normalized TTR that accounts for text length by using a sliding window.

        Args:
            text (str): Text to analyze
            window_size (int): Size of the sliding window
            step_size (int): Step size for the sliding window

        Returns:
            float: Normalized TTR value
        """
        tokens = self._tokenize(text)

        if len(tokens) < window_size:
            return self._ttr(tokens)

        ttrs = []
        for i in range(0, len(tokens) - window_size + 1, step_size):
            window = tokens[i:i + window_size]
            ttrs.append(self._ttr(window))

        return sum(ttrs) / len(ttrs) if ttrs else 0.0

    def compute_diversity_metrics(self) -> Dict[str, float]:
        """
        Computes a comprehensive set of lexical diversity metrics.

        Returns:
            Dict[str, float]: Dictionary with all diversity metrics.
        """
        texts = self._extract_texts()
        if not texts:
            return {
                "ttr_avg": 0.0,
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "distinct_3": 0.0,
                "normalized_ttr": 0.0
            }

        aggregated_text = " ".join(texts)
        traditional_metrics = self.compute_traditional_ttr(aggregated_text)

        li_metrics = self.compute_li_diversity_metrics(texts)

        normalized_ttr = self.compute_normalized_ttr(aggregated_text)

        all_metrics = {
            **traditional_metrics,
            **li_metrics,
            "normalized_ttr": normalized_ttr
        }

        return all_metrics