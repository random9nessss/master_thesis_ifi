import os
import re
import json
from typing import List, Dict
import pandas as pd
import warnings
import requests
import torch
import numpy as np
import torch.nn.functional as F
from email_reply_parser import EmailReplyParser
from transformers import logging as hf_logging, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import configure_http_backend
from flair.data import Sentence
from flair.models import TextClassifier


# -------------------------------------------------------------------
# Warning Silencing
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# -------------------------------------------------------------------
# SSL Backend Factoring
# -------------------------------------------------------------------
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)


# -------------------------------------------------------------------
# Email Preprocessor Class
# -------------------------------------------------------------------
class EmailPreprocessor:
    def __init__(self):

        # -------------------------------------------------------------------
        # Disclaimer Patterns
        # -------------------------------------------------------------------
        self._disclaimer_patterns = [
            r"(?i)(disclaimer:.*?)(?=\n[A-Z]|$)",
            r"(?i)(confidential(ity)? notice:.*?)(?=\n[A-Z]|$)",
            r"(?i)(legal notice:.*?)(?=\n[A-Z]|$)",
            r"(?i)(if you are not the intended recipient.*?)(?=\n[A-Z]|$)",
            r"(?i)(privileged and confidential.*?)(?=\n[A-Z]|$)",
            r"(?i)(this (e-?)?mail (message )?is intended only for.*?)(?=\n[A-Z]|$)",
            r"(?i)(this communication is intended solely for.*?)(?=\n[A-Z]|$)",
            r"(?i)(this message contains confidential information.*?)(?=\n[A-Z]|$)",
        ]

        # -------------------------------------------------------------------
        # Promotion Patterns
        # -------------------------------------------------------------------
        self._promo_patterns = [
            r"(?i)follow us on [\w\s]+",
            r"(?i)like us on [\w\s]+",
            r"(?i)connect with us on [\w\s]+",
            r"(?i)subscribe( to)? [\w\s]+",
            r"(?i)unsubscrib(e|ing|tion).*?(\.|$)",
            r"(?i)view (this|it) in your browser.*?(\.|$)",
            r"(?i)visit our website.*?(\.|$)",
            r"(?i)sent from my (iphone|ipad|android|smartphone|blackberry|mobile device).*?(\.|$)",
            r"(?i)download our app.*?(\.|$)",
            r"(?i)get it on (the app store|google play).*?(\.|$)",
            r"(?i)having trouble viewing this email\?.*?(\.|$)",
            r"(?i)to ensure delivery to your inbox.*?(\.|$)",
        ]

        # -------------------------------------------------------------------
        # Forwarding Headers
        # -------------------------------------------------------------------
        self._forwarding_headers = [
            r"(?i)^-{3,}[\s\S]*?Forwarded[^\n]*\n",
            r"(?i)^From:.*?Sent:.*?To:.*?Subject:.*?\n",
            r"(?i)^On.*?wrote:.*?\n",
            r"(?i)^_+[\s\S]*?From:.*?\n",
        ]

    def process_email(self, email_body: str) -> str:
        # -------------------------------------------------------------------
        # Removal of HTML Tags
        # -------------------------------------------------------------------
        email_body = re.sub(r'<[^>]+>', '', email_body)

        # -------------------------------------------------------------------
        # Normalization of Whitespace and Line Breaks
        # -------------------------------------------------------------------
        email_body = re.sub(r'\s+', ' ', email_body).strip()

        # -------------------------------------------------------------------
        # Discarding Quoted History
        # -------------------------------------------------------------------
        email_body = EmailReplyParser.parse_reply(email_body)
        # Removal of Disclaimer Patterns

        # -------------------------------------------------------------------
        # Removal of Disclaimer Patterns
        # -------------------------------------------------------------------
        for pattern in self._disclaimer_patterns:
            email_body = re.sub(pattern, '', email_body, flags=re.DOTALL)

        # -------------------------------------------------------------------
        # Removal of Promotion Patterns
        # -------------------------------------------------------------------
        for pattern in self._promo_patterns:
            email_body = re.sub(pattern, '', email_body)

        # -------------------------------------------------------------------
        # Removal of Forwardings
        # -------------------------------------------------------------------
        for pattern in self._forwarding_headers:
            email_body = re.sub(pattern, '', email_body, flags=re.MULTILINE)
        email_body = re.sub(r'\s+', ' ', email_body).strip()
        return email_body


# -------------------------------------------------------------------
# Ensemble Sentiment Analyzer
# -------------------------------------------------------------------
class EnsembleSentimentAnalyzer:
    """
    An ensemble sentiment analyzer that combines Flair and RoBERTa models to compute sentiment scores.
    """

    def __init__(self, flair_weight: float = 0.5, roberta_weight: float = 0.5):
        """
        Initialize the ensemble sentiment analyzer with specified weights and load the models.

        Args:
            flair_weight (float, optional): Weight for the Flair model's sentiment score. Defaults to 0.5.
            roberta_weight (float, optional): Weight for the RoBERTa model's sentiment score. Defaults to 0.5.
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------------------
        # Ensembling Weights
        # -------------------------------------------------------------------
        self._flair_weight = flair_weight
        self._roberta_weight = roberta_weight

        # -------------------------------------------------------------------
        # Loading Flair Model
        # -------------------------------------------------------------------
        self.flair_model = TextClassifier.load('en-sentiment')
        self.flair_model.to(self.device)

        # -------------------------------------------------------------------
        # Loading Roberta-large Model
        # -------------------------------------------------------------------
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('siebert/sentiment-roberta-large-english')
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
            'siebert/sentiment-roberta-large-english')
        self.roberta_model.to(self.device)

    def split_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Splits the input text into smaller chunks such that each chunk has at most max_tokens tokens,
        as determined by the RoBERTa tokenizer.

        Args:
            text (str): The text to be split.
            max_tokens (int, optional): Maximum number of tokens per chunk. Defaults to 512.

        Returns:
            List[str]: A list of text chunks.
        """
        tokens = self.roberta_tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.roberta_tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        return chunks

    def get_flair_sentiment(self, text: str) -> float:
        """
        Computes the positive sentiment probability for the given text using the Flair model.
        If the label is POSITIVE, the confidence score is used directly; if NEGATIVE, (1 - score) is used.

        Args:
            text (str): The text for which to compute the sentiment.

        Returns:
            float: The positive sentiment probability.
        """
        sentence = Sentence(text)
        self.flair_model.predict(sentence)
        label = sentence.labels[0].value
        score = sentence.labels[0].score
        pos_prob = score if label == "POSITIVE" else 1 - score
        return pos_prob

    def get_roberta_sentiment(self, text: str) -> float:
        """
        Computes the positive sentiment probability for the given text using the RoBERTa model.
        Applies tokenization, truncation, and softmax to the output logits.
        Assumes that index 1 corresponds to the positive sentiment class.

        Args:
            text (str): The text for which to compute the sentiment.

        Returns:
            float: The positive sentiment probability.
        """
        inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pos_prob = probs[0][1].item()
        return pos_prob

    def _remove_email_metadata(self, text: str) -> str:
        """
        Removes email metadata such as 'Sent:', 'To:', 'From:', or 'Cc:' lines from the text.

        Args:
            text (str): The text from which to remove metadata.

        Returns:
            str: The text without metadata.
        """
        pattern = r'(?:Sent|To|From|Cc):.*?(?=(?:Sent|To|From|Cc):|$)'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        return cleaned_text.strip()

    def predict_sentiment(self, text: str) -> dict:
        """
        Predicts the sentiment scores for the given text.

        Returns a dictionary with three sentiment probabilities:
        - sentiment_neg: probability of negative sentiment (0-1)
        - sentiment_neu: probability of neutral sentiment (0-1)
        - sentiment_pos: probability of positive sentiment (0-1)

        Args:
            text (str): The text for which to predict sentiment.

        Returns:
            dict: A dictionary containing sentiment probabilities.
        """
        # -------------------------------------------------------------------
        # Removal of Email Metadata
        # -------------------------------------------------------------------
        text = self._remove_email_metadata(text)

        # -------------------------------------------------------------------
        # Text Chunking
        # -------------------------------------------------------------------
        chunks = self.split_text(text)
        ensemble_scores = []
        for chunk in chunks:
            flair_prob = self.get_flair_sentiment(chunk)
            roberta_prob = self.get_roberta_sentiment(chunk)
            ensemble_prob = (
                                    self._flair_weight * flair_prob + self._roberta_weight * roberta_prob
                            ) / (self._flair_weight + self._roberta_weight)
            ensemble_scores.append(ensemble_prob)

        # -------------------------------------------------------------------
        # Average Score across Chunks
        # -------------------------------------------------------------------
        avg_ensemble_prob = np.mean(ensemble_scores)

        ################
        # Ranges Used
        ################
        # [0.0-0.4)
        # Strong negative sentiment
        # [0.4-0.45)
        # Weak negative sentiment
        # [0.45-0.55)
        # Neutral sentiment
        # [0.55-0.6)
        # Weak positive sentiment
        # [0.6-1.0]
        # Strong positive sentiment

        if avg_ensemble_prob < 0.4:
            neg_prob = 1.0 - avg_ensemble_prob

        elif avg_ensemble_prob < 0.45:
            neg_prob = 0.6 - avg_ensemble_prob

        else:
            neg_prob = max(0, 0.4 - avg_ensemble_prob * 0.5)

        if avg_ensemble_prob > 0.6:
            pos_prob = avg_ensemble_prob
        elif avg_ensemble_prob > 0.55:
            pos_prob = avg_ensemble_prob - 0.1
        else:
            pos_prob = max(0, avg_ensemble_prob * 0.5 - 0.1)

        neu_prob = 1.0 - (neg_prob + pos_prob)

        neg_prob = max(0, min(1, neg_prob))
        neu_prob = max(0, min(1, neu_prob))
        pos_prob = max(0, min(1, pos_prob))

        total = neg_prob + neu_prob + pos_prob
        if total > 0:
            neg_prob /= total
            neu_prob /= total
            pos_prob /= total

        return {
            "sentiment_neg": neg_prob,
            "sentiment_neu": neu_prob,
            "sentiment_pos": pos_prob
        }


# -------------------------------------------------------------------
# Ensemble Sentiment Analysis (Aggregating Multiple Emails)
# -------------------------------------------------------------------
class EnsembleSentimentAnalysis:
    """
    Computes the sentiment analysis for multiple emails by aggregating their texts.
    Accepts either a JSON file with an "email_chain" column or a list of email texts.
    Also supports analyzing and saving sentiment distributions across email chains.
    """

    def __init__(self, file_path: str = None, texts: List[str] = None, field: str = "body"):
        """
        Initialize the EnsembleSentimentAnalysis class with either a file path or a list of texts.
        Processes each email text using the EmailPreprocessor and aggregates them.

        Args:
            file_path (str, optional): Path to a JSON file containing email data.
            texts (List[str], optional): A list of email text strings.
            field (str, optional): The field to extract from each email ("body" or "subject"). Defaults to "body".

        Raises:
            ValueError: If neither file_path nor texts is provided.
        """
        if file_path is None and texts is None:
            raise ValueError("Either file_path or texts must be provided.")

        self.field = field
        self.model_name = None
        self.method_name = None
        self.individual_sentiments = []

        if file_path:
            self.file_path = file_path
            self._parse_model_method_from_path(file_path)
            self.emails = pd.read_json(file_path)
            self.texts = self._extract_texts()
        else:
            self.texts = texts

        # -------------------------------------------------------------------
        # Text Preprocessing
        # -------------------------------------------------------------------
        self.preprocessor = EmailPreprocessor()
        self.processed_texts = [self.preprocessor.process_email(text) for text in self.texts if text.strip()]
        self.aggregated_text = " ".join(self.processed_texts).strip()
        self.ensemble_analyzer = EnsembleSentimentAnalyzer()

    def _parse_model_method_from_path(self, file_path: str):
        """
        Extract model and method information from the file path.
        This helps organize and identify the source of sentiment data.

        Args:
            file_path (str): Path to the JSON file containing email data.
        """
        try:
            path_lower = file_path.lower()

            if "attrprompting" in path_lower:
                self.method_name = "Attr Prompting"
            elif "baserefine" in path_lower and "llama3b" in path_lower:
                self.method_name = "Bare Llama3B"
            elif "baserefine" in path_lower and "llama8b" in path_lower:
                self.method_name = "Bare Llama8B"
            elif "fewshot" in path_lower:
                self.method_name = "Fewshot"
            elif "zeroshot" in path_lower:
                self.method_name = "Zeroshot"
            else:
                self.method_name = "Unknown"

            if "claude" in path_lower:
                self.model_name = "Claude"
            elif "deepseek" in path_lower:
                self.model_name = "Deepseek"
            elif "gemini" in path_lower:
                self.model_name = "Gemini"
            elif "gpt-4" in path_lower or "gpt4" in path_lower:
                self.model_name = "GPT4"
            elif "mistral" in path_lower:
                self.model_name = "Mistral"
            else:
                self.model_name = "Unknown"
        except Exception:
            self.model_name = "Unknown"
            self.method_name = "Unknown"

    def _extract_texts(self) -> List[str]:
        """
        Extracts email texts from a JSON file by selecting the specified field from ALL emails in each email chain.

        Returns:
            List[str]: A list of all extracted email texts.
        """
        texts = []
        for idx, row in self.emails.iterrows():
            email_chain = row.get("email_chain")
            if email_chain and isinstance(email_chain, list):
                for email in email_chain:
                    if self.field in email and isinstance(email[self.field], str) and email[self.field].strip():
                        texts.append(email[self.field])
        return texts

    def compute_individual_sentiments(self) -> List[dict]:
        """
        Computes sentiment for each individual email chain separately.

        Returns:
            List[dict]: A list of dictionaries containing sentiment scores for each chain
        """
        self.individual_sentiments = []

        for idx, row in self.emails.iterrows():
            email_chain = row.get("email_chain")
            if email_chain and isinstance(email_chain, list):
                chain_texts = []
                for email in email_chain:
                    if self.field in email and isinstance(email[self.field], str) and email[self.field].strip():
                        chain_texts.append(email[self.field])

                if not chain_texts:
                    continue

                processed_texts = [self.preprocessor.process_email(text) for text in chain_texts]
                chain_text = " ".join([text for text in processed_texts if text.strip()])

                if not chain_text:
                    continue

                sentiment = self.ensemble_analyzer.predict_sentiment(chain_text)

                result = {
                    "chain_id": idx,
                    "model": self.model_name or "Unknown",
                    "method": self.method_name or "Unknown",
                    "num_emails": len(chain_texts),
                    "sentiment_neg": sentiment["sentiment_neg"],
                    "sentiment_neu": sentiment["sentiment_neu"],
                    "sentiment_pos": sentiment["sentiment_pos"]
                }
                self.individual_sentiments.append(result)

        return self.individual_sentiments

    def save_sentiment_distribution(self, output_dir: str = "output/sentiment_distribution") -> str:
        """
        Saves the individual sentiment scores to a CSV file for distribution analysis.

        Args:
            output_dir (str): Directory where the CSV file will be saved

        Returns:
            str: Path to the saved CSV file
        """
        if not self.individual_sentiments:
            self.compute_individual_sentiments()

        os.makedirs(output_dir, exist_ok=True)

        if self.model_name and self.method_name:
            method_str = self.method_name.lower().replace(' ', '_')
            model_str = self.model_name.lower()
            filename = f"sentiment_distribution_{method_str}_{model_str}.csv"
        else:
            filename = "sentiment_distribution.csv"

        file_path = os.path.join(output_dir, filename)
        df = pd.DataFrame(self.individual_sentiments)
        df.to_csv(file_path, index=False)

        return file_path

    def compute_sentiment(self) -> dict:
        """
        Aggregates the processed email texts and computes the overall sentiment using the ensemble analyzer.
        Also computes individual chain sentiments for distribution analysis.

        Returns:
            dict: A dictionary with the sentiment probabilities:
                - "sentiment_neg": Probability of negative sentiment (0-1)
                - "sentiment_neu": Probability of neutral sentiment (0-1)
                - "sentiment_pos": Probability of positive sentiment (0-1)
        """
        self.compute_individual_sentiments()

        if not self.aggregated_text:
            return {
                "sentiment_neg": 0.0,
                "sentiment_neu": 1.0,
                "sentiment_pos": 0.0
            }
        return self.ensemble_analyzer.predict_sentiment(self.aggregated_text)

    def get_individual_sentiments(self) -> List[dict]:
        """
        Returns the individual sentiment scores for each email chain.
        Computes them if not already computed.

        Returns:
            List[dict]: A list of dictionaries containing sentiment scores
        """
        if not self.individual_sentiments:
            self.compute_individual_sentiments()
        return self.individual_sentiments


# -------------------------------------------------------------------
# CSV Class for ENRON Emails
# -------------------------------------------------------------------
class EnsembleSentimentAnalysisCSV:
    """
    Modified version of EnsembleSentimentAnalysis to work with CSV files containing a parsed_content column.
    Computes sentiment analysis for multiple emails by aggregating their texts.
    """

    def __init__(self, file_path: str = None, texts: List[str] = None, content_column: str = "parsed_content"):
        """
        Initialize the EnsembleSentimentAnalysisCSV class with either a file path or a list of texts.
        Processes each email text using the EmailPreprocessor and aggregates them.

        Args:
            file_path (str, optional): Path to a CSV file containing email data.
            texts (List[str], optional): A list of email text strings.
            content_column (str, optional): The column name containing the email content. Defaults to "parsed_content".

        Raises:
            ValueError: If neither file_path nor texts is provided.
        """
        if file_path is None and texts is None:
            raise ValueError("Either file_path or texts must be provided.")

        self.content_column = content_column
        self.model_name = None
        self.method_name = None
        self.individual_sentiments = []

        if file_path:
            self.file_path = file_path
            self._parse_model_method_from_path(file_path)
            self.data = pd.read_csv(file_path)

            if self.content_column not in self.data.columns:
                raise ValueError(f"Column '{self.content_column}' not found in the CSV file.")

            self.texts = self._extract_texts()
        else:
            self.texts = texts

        # -------------------------------------------------------------------
        # Text Preprocessing
        # -------------------------------------------------------------------
        self.preprocessor = EmailPreprocessor()
        self.processed_texts = [self.preprocessor.process_email(text) for text in self.texts if
                                isinstance(text, str) and text.strip()]
        self.aggregated_text = " ".join(self.processed_texts).strip()
        self.ensemble_analyzer = EnsembleSentimentAnalyzer()

    def _parse_model_method_from_path(self, file_path: str):
        """
        Extract model and method information from the file path.
        This helps organize and identify the source of sentiment data.

        Args:
            file_path (str): Path to the CSV file containing email data.
        """
        try:
            path_lower = file_path.lower()

            if "attrprompting" in path_lower:
                self.method_name = "Attr Prompting"
            elif "baserefine" in path_lower and "llama3b" in path_lower:
                self.method_name = "Bare Llama3B"
            elif "baserefine" in path_lower and "llama8b" in path_lower:
                self.method_name = "Bare Llama8B"
            elif "fewshot" in path_lower:
                self.method_name = "Fewshot"
            elif "zeroshot" in path_lower:
                self.method_name = "Zeroshot"
            else:
                self.method_name = "Unknown"

            if "claude" in path_lower:
                self.model_name = "Claude"
            elif "deepseek" in path_lower:
                self.model_name = "Deepseek"
            elif "gemini" in path_lower:
                self.model_name = "Gemini"
            elif "gpt-4" in path_lower or "gpt4" in path_lower:
                self.model_name = "GPT4"
            elif "mistral" in path_lower:
                self.model_name = "Mistral"
            else:
                self.model_name = "Unknown"
        except Exception:
            self.model_name = "Unknown"
            self.method_name = "Unknown"

    def _extract_texts(self) -> List[str]:
        """
        Extracts email texts from the CSV file by selecting the specified content column.

        Returns:
            List[str]: A list of all extracted email texts.
        """
        texts = []
        for _, row in self.data.iterrows():
            content = row.get(self.content_column)
            if isinstance(content, str) and content.strip():
                texts.append(content)
        return texts

    def compute_individual_sentiments(self) -> List[Dict]:
        """
        Computes sentiment for each individual email separately.

        Returns:
            List[Dict]: A list of dictionaries containing sentiment scores for each email
        """
        self.individual_sentiments = []

        for idx, row in self.data.iterrows():
            content = row.get(self.content_column)

            if not isinstance(content, str) or not content.strip():
                continue

            processed_text = self.preprocessor.process_email(content)

            if not processed_text:
                continue

            sentiment = self.ensemble_analyzer.predict_sentiment(processed_text)

            metadata = {}
            for col in self.data.columns:
                if col != self.content_column and not pd.isna(row.get(col)):
                    metadata[col] = row.get(col)

            result = {
                "email_id": idx,
                "model": self.model_name or "Unknown",
                "method": self.method_name or "Unknown",
                "sentiment_neg": sentiment["sentiment_neg"],
                "sentiment_neu": sentiment["sentiment_neu"],
                "sentiment_pos": sentiment["sentiment_pos"],
                **metadata
            }
            self.individual_sentiments.append(result)

        return self.individual_sentiments

    def save_sentiment_distribution(self, output_dir: str = "output/sentiment_distribution") -> str:
        """
        Saves the individual sentiment scores to a CSV file for distribution analysis.

        Args:
            output_dir (str): Directory where the CSV file will be saved

        Returns:
            str: Path to the saved CSV file
        """
        if not self.individual_sentiments:
            self.compute_individual_sentiments()

        os.makedirs(output_dir, exist_ok=True)

        if self.model_name and self.method_name:
            method_str = self.method_name.lower().replace(' ', '_')
            model_str = self.model_name.lower()
            filename = f"sentiment_distribution_{method_str}_{model_str}.csv"
        else:
            filename = "sentiment_distribution.csv"

        file_path = os.path.join(output_dir, filename)
        df = pd.DataFrame(self.individual_sentiments)
        df.to_csv(file_path, index=False)

        return file_path

    def compute_sentiment(self) -> Dict:
        """
        Aggregates the processed email texts and computes the overall sentiment using the ensemble analyzer.
        Also computes individual email sentiments for distribution analysis.

        Returns:
            Dict: A dictionary with the sentiment probabilities:
                - "sentiment_neg": Probability of negative sentiment (0-1)
                - "sentiment_neu": Probability of neutral sentiment (0-1)
                - "sentiment_pos": Probability of positive sentiment (0-1)
        """
        self.compute_individual_sentiments()

        if not self.aggregated_text:
            return {
                "sentiment_neg": 0.0,
                "sentiment_neu": 1.0,
                "sentiment_pos": 0.0
            }
        return self.ensemble_analyzer.predict_sentiment(self.aggregated_text)

    def get_individual_sentiments(self) -> List[Dict]:
        """
        Returns the individual sentiment scores for each email.
        Computes them if not already computed.

        Returns:
            List[Dict]: A list of dictionaries containing sentiment scores
        """
        if not self.individual_sentiments:
            self.compute_individual_sentiments()
        return self.individual_sentiments