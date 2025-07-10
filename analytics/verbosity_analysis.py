import os
import pandas as pd
import json
import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')


class VerbosityAnalysis:
    """
    A class to compute verbosity metrics for email texts.

    The verbosity metrics include:
      - Average number of words per email.
      - Average number of sentences per email.
      - Average number of words per sentence.

    The class can be instantiated with either a file path (a JSON file with an "email_chain" column)
    and a field indicator ("body" or "subject"), or with a list of texts.
    """

    def __init__(self, file_path: str = None, texts: List[str] = None, field: str = "body"):
        """
        Initialize the VerbosityMetric object.

        Args:
            file_path (str, optional): Path to the JSON file containing email data.
            texts (List[str], optional): A list of text strings to analyze.
            field (str, optional): The field to extract from emails when using a file (either "body" or "subject").

        Raises:
            ValueError: If neither file_path nor texts is provided.
        """
        if file_path is None and texts is None:
            raise ValueError("Either file_path or texts must be provided.")

        self.field = field
        if file_path:
            self.file_path = file_path
            self.emails = pd.read_json(file_path)
            self.texts = self._extract_texts()
        else:
            self.texts = texts
        self.aggregated_text = " ".join(self.texts).strip()

    def _extract_texts(self) -> List[str]:
        """
        Extract the specified field (body or subject) from all email objects in the JSON file.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        for idx, row in self.emails.iterrows():
            email_chain = row.get("email_chain")
            if email_chain and isinstance(email_chain, list):
                email = email_chain[0]
                if (self.field in email and isinstance(email[self.field], str)
                        and email[self.field].strip()):
                    texts.append(email[self.field])
        return texts

    def compute_verbosity(self) -> Dict[str, float]:
        """
        Compute verbosity metrics for the aggregated texts.

        Returns:
            dict: A dictionary with keys:
                - avg_words_per_email
                - avg_sentences_per_email
                - avg_words_per_sentence
        """
        total_emails = len(self.texts)
        total_words = sum(len(word_tokenize(text)) for text in self.texts)
        total_sentences = sum(len(sent_tokenize(text)) for text in self.texts)

        avg_words_per_email = total_words / total_emails if total_emails > 0 else 0
        avg_sentences_per_email = total_sentences / total_emails if total_emails > 0 else 0
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

        return {
            "avg_words_per_email": avg_words_per_email,
            "avg_sentences_per_email": avg_sentences_per_email,
            "avg_words_per_sentence": avg_words_per_sentence
        }


# -------------------------------------------------------------------
# CSV Class for ENRON Emails
# -------------------------------------------------------------------
class VerbosityAnalysisCSV:
    """
    Modified VerbosityAnalysis class to work with CSV files containing a parsed_content column.

    The verbosity metrics include:
      - Average number of words per document.
      - Average number of sentences per document.
      - Average number of words per sentence.

    The class can be instantiated with either a file path (a CSV file with a content column)
    or with a list of texts.
    """

    def __init__(self, file_path: str = None, texts: List[str] = None, content_column: str = "parsed_content"):
        """
        Initialize the VerbosityAnalysisCSV object.

        Args:
            file_path (str, optional): Path to the CSV file containing email data.
            texts (List[str], optional): A list of text strings to analyze.
            content_column (str, optional): The column name containing the content to analyze.

        Raises:
            ValueError: If neither file_path nor texts is provided.
        """
        if file_path is None and texts is None:
            raise ValueError("Either file_path or texts must be provided.")

        self.content_column = content_column
        if file_path:
            self.file_path = file_path
            self.data = pd.read_csv(file_path)

            # Check if the content column exists
            if self.content_column not in self.data.columns:
                raise ValueError(f"Column '{self.content_column}' not found in the CSV file.")

            self.texts = self._extract_texts()
        else:
            self.texts = texts
        self.aggregated_text = " ".join([t for t in self.texts if isinstance(t, str)]).strip()

    def _extract_texts(self) -> List[str]:
        """
        Extract the specified content column from all rows in the CSV file.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        for _, row in self.data.iterrows():
            content = row.get(self.content_column)
            if isinstance(content, str) and content.strip():
                texts.append(content)
        return texts

    def compute_verbosity(self) -> Dict[str, float]:
        """
        Compute verbosity metrics for the texts.

        Returns:
            dict: A dictionary with keys:
                - avg_words_per_document
                - avg_sentences_per_document
                - avg_words_per_sentence
        """
        # Filter out any non-string items that might have slipped through
        valid_texts = [text for text in self.texts if isinstance(text, str) and text.strip()]
        total_documents = len(valid_texts)

        if total_documents == 0:
            return {
                "avg_words_per_document": 0,
                "avg_sentences_per_document": 0,
                "avg_words_per_sentence": 0
            }

        # Count words and sentences
        word_counts = []
        sentence_counts = []

        for text in valid_texts:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            word_counts.append(len(words))
            sentence_counts.append(len(sentences))

        total_words = sum(word_counts)
        total_sentences = sum(sentence_counts)

        avg_words_per_document = total_words / total_documents
        avg_sentences_per_document = total_sentences / total_documents
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

        return {
            "avg_words_per_document": avg_words_per_document,
            "avg_sentences_per_document": avg_sentences_per_document,
            "avg_words_per_sentence": avg_words_per_sentence,
            "total_documents": total_documents,
            "total_words": total_words,
            "total_sentences": total_sentences
        }

    def compute_verbosity_by_group(self, group_column: str) -> Dict[str, Dict[str, float]]:
        """
        Compute verbosity metrics grouped by a specific column in the CSV.

        Args:
            group_column (str): The column name to group by.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary mapping group values to their verbosity metrics.
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Column '{group_column}' not found in the CSV file.")

        # Create a DataFrame with the content and group column
        df = self.data[[self.content_column, group_column]].copy()
        df = df[df[self.content_column].apply(lambda x: isinstance(x, str) and bool(x.strip()))]

        results = {}
        for group_value, group_df in df.groupby(group_column):
            texts = group_df[self.content_column].tolist()

            # Count words and sentences
            word_counts = []
            sentence_counts = []

            for text in texts:
                if isinstance(text, str) and text.strip():
                    words = word_tokenize(text)
                    sentences = sent_tokenize(text)
                    word_counts.append(len(words))
                    sentence_counts.append(len(sentences))

            total_documents = len(word_counts)
            total_words = sum(word_counts)
            total_sentences = sum(sentence_counts)

            if total_documents == 0:
                results[group_value] = {
                    "avg_words_per_document": 0,
                    "avg_sentences_per_document": 0,
                    "avg_words_per_sentence": 0,
                    "total_documents": 0,
                    "total_words": 0,
                    "total_sentences": 0
                }
                continue

            avg_words_per_document = total_words / total_documents
            avg_sentences_per_document = total_sentences / total_documents
            avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

            results[group_value] = {
                "avg_words_per_document": avg_words_per_document,
                "avg_sentences_per_document": avg_sentences_per_document,
                "avg_words_per_sentence": avg_words_per_sentence,
                "total_documents": total_documents,
                "total_words": total_words,
                "total_sentences": total_sentences
            }

        return results

    def save_verbosity_metrics(self, output_path: str = "verbosity_metrics.csv") -> str:
        """
        Compute verbosity metrics and save them to a CSV file.

        Args:
            output_path (str, optional): Path where to save the CSV file.

        Returns:
            str: The path to the saved CSV file.
        """
        metrics = self.compute_verbosity()
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False)
        return output_path