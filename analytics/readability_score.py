import os
import pandas as pd
import json
import re
from typing import List
import textstat
import nltk


class ReadabilityScore:
    """
    A class to compute the readability of email texts using the Flesch Reading Ease score.

    The class can be instantiated with either a file path (a JSON file containing an "email_chain"
    column) and a specified field ("body" or "subject"), or with a list of texts.
    """
    def __init__(self, file_path: str = None, texts: List[str] = None, field: str = "body"):
        """
        Initialize the ReadabilityScore object.

        Args:
            file_path (str, optional): Path to the JSON file containing email data.
            texts (List[str], optional): A list of text strings to analyze.
            field (str, optional): The email field to analyze ("body" or "subject") when using file_path.
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
                if self.field in email and isinstance(email[self.field], str) and email[self.field].strip():
                    texts.append(email[self.field])
        return texts

    def compute_readability(self) -> float:
        """
        Compute the Flesch Reading Ease score of the aggregated text.

        Returns:
            float: The Flesch Reading Ease score.
        """
        if not self.aggregated_text:
            return 0.0
        return textstat.flesch_reading_ease(self.aggregated_text)