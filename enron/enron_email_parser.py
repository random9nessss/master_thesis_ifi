import pandas as pd
import email
from email import policy

from config.logger import CustomLogger


class EnronEmailParser:
    """
    A class to parse the Enron email corpus from a CSV file containing raw email messages.

    The parser reads a CSV file with a 'message' column that contains the raw email text,
    extracts the email content from each message, and stores the result in a DataFrame that
    contains only the parsed email content.
    """

    def __init__(self, enron_path_raw: str, enron_path_processed: str = None, truncated: bool = False):
        """
        Initializes the EnronEmailParser instance.

        If a processed CSV file is not provided, the raw CSV file is processed to extract
        the email content. Otherwise, the processed corpus is loaded directly.

        Parameters:
            enron_path_raw (str): Path to the raw Enron email corpus CSV file.
            enron_path_processed (str, optional): Path to an already processed Enron email CSV file.
                                                  If provided, the processed corpus is loaded from this file.
                                                  Otherwise, the raw corpus is processed.
            truncated (bool, optional): Indicates if only a subset of the enron messages shall be parsed.
        """
        self._enron_email_corpus_raw = pd.read_csv(enron_path_raw)

        # Logging Functionality
        self.logger = CustomLogger(name="EnronEmailParser")
        self.logger.ok("EnronEmailParser initialized")

        # -------------------------------------------------------------------
        # Conditional Processing of Raw Email Corpus
        # -------------------------------------------------------------------
        if enron_path_processed is None:
            self._enron_email_corpus_processed = self._process_enron_corpus(truncated)
        else:
            self._enron_email_corpus_processed = pd.read_csv(enron_path_processed)

    def _parse_email_message(self, message_text: str):
        """
        Parses a raw email message (as a string) and extracts key headers and the email content.

        Parameters:
            message_text (str): The raw email message as a string.

        Returns:
            tuple: A tuple containing:
                - headers (dict): A dictionary of extracted headers.
                - content (str): A string containing the email's plain text content.
        """

        msg = email.message_from_string(message_text, policy=policy.default)

        # -------------------------------------------------------------------
        # Header Extraction
        # -------------------------------------------------------------------
        try:
            headers = {
                "Message-ID": msg.get("Message-ID"),
                "Date": msg.get("Date"),
                "From": msg.get("From"),
                "To": msg.get("To"),
                "Cc": msg.get("Cc"),
                "Subject": msg.get("Subject"),
                "Content-Type": msg.get("Content-Type"),
                "Folder": msg.get("Folder")
            }

        except Exception as e:
            return None, None

        # -------------------------------------------------------------------
        # Content Extraction
        # -------------------------------------------------------------------
        content = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and not part.get_filename():
                    content += part.get_content()
        else:
            content = msg.get_content()

        return headers, content

    def _process_enron_corpus(self, truncated: bool = False) -> pd.DataFrame:
        """
        Processes the raw Enron email corpus DataFrame by parsing each email message to extract
        both the headers and the plain text content.

        This method iterates over each email in the raw DataFrame, parses the message to extract
        the headers and the plain text content, and creates a new DataFrame that contains a column
        for each header as well as a 'parsed_content' column for the email body.

        Parameters:
            truncated (bool, optional): An indicator if only a subset of the dataframe shall be parsed.

        Returns:
            pd.DataFrame: A DataFrame where each column corresponds to a header (e.g., 'Message-ID',
                          'Date', etc.) or the email's parsed content.
        """
        parsed_data = []

        # -------------------------------------------------------------------
        # Sampling 50% of Data
        # -------------------------------------------------------------------
        if truncated:
            self._enron_email_corpus_raw = self._enron_email_corpus_raw.sample(frac=0.5, random_state=42)
            self.logger.info("Parsing 50% of the original corpus")

        total_emails = len(self._enron_email_corpus_raw)
        # -------------------------------------------------------------------
        # Parsing
        # -------------------------------------------------------------------
        with self.logger.progress_bar(total=total_emails, desc="Parsing Enron Emails") as pbar:
            for index, row in self._enron_email_corpus_raw.iterrows():
                message_text = row["message"]
                headers, content = self._parse_email_message(message_text)

                if headers is None or content is None:
                    pbar.update(1)
                    continue

                headers["parsed_content"] = content
                parsed_data.append(headers)
                pbar.update(1)

        processed_df = pd.DataFrame(parsed_data)
        self.logger.info("Enron Corpus parsed")
        return processed_df

    def get_processed_emails(self) -> pd.DataFrame:
        """
        Returns the processed Enron email corpus containing only the parsed email content.

        Returns:
            pd.DataFrame: The processed DataFrame with a single column 'parsed_content'.
        """
        return self._enron_email_corpus_processed

    def serialize_to_csv(self, output_csv_path: str) -> None:
        """
        Serializes the processed Enron email corpus DataFrame to a CSV file.

        Parameters:
            output_csv_path (str): The path to the output CSV file.

        Returns:
            None
        """
        self._enron_email_corpus_processed.to_csv(output_csv_path, index=False)
        self.logger.info(f"Processed Enron corpus serialized to {output_csv_path}")