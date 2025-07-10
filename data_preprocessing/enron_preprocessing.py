import re
from config.logger import CustomLogger


class EmailPreprocessor:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.logger = CustomLogger(name="Preprocessor")

        # -------------------------------
        # Disclaimers
        # -------------------------------
        self._disclaimer_patterns = [
            r"(?i)(disclaimer:.*?)(?=\n[A-Z]|$)",
            r"(?i)(confidential(ity)? notice:.*?)(?=\n[A-Z]|$)",
            r"(?i)(legal notice:.*?)(?=\n[A-Z]|$)",
            r"(?i)(if you are not the intended recipient.*?)(?=\n[A-Z]|$)",
            r"(?i)(privileged and confidential.*?)(?=\n[A-Z]|$)",
            r"(?i)(this (e-?)?mail (message )?is intended only for.*?)(?=\n[A-Z]|$)",
            r"(?i)(this communication is intended solely for.*?)(?=\n[A-Z]|$)",
            r"(?i)(this message contains confidential information.*?)(?=\n[A-Z]|$)",
            r"(?i)(please do not reply to this email.*?)(?=\n[A-Z]|$)",
            r"(?i)(this email may contain.*?confidential.*?)(?=\n[A-Z]|$)",
            r"(?i)(the information (contained )?in this (e-?)?mail.*?)(?=\n[A-Z]|$)",
            r"(?i)(this message is for the designated recipient.*?)(?=\n[A-Z]|$)",
        ]

        # -------------------------------
        # Personal Identifiable Information
        # -------------------------------
        self._pii_patterns = [
            # Emails
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'<[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}>',
            # Phone
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            # Typical email headers:
            r'(?i)(^|\n)(to|cc|bcc|from|subject|date|sent):.*?(\n|$)',
            # SSN
            r'\d{3}-\d{2}-\d{4}',
        ]

        ##################################################################
        # Promotion Patterns
        ##################################################################
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
            r"(?i)to stop receiving (these|our) emails.*?(\.|$)",
            r"(?i)click here to (view|update).*?(\.|$)",
            r"(?i)you( are)? receiv(ed|ing) this email because.*?(\.|$)",
            r"(?i)this email was sent by.*?(\.|$)",
        ]

        ##################################################################
        # Forwarding Headers
        ##################################################################
        self._forwarding_headers = [
            r"(?i)^-{3,}[\s\S]*?forwarded[^\n]*\n",
            r"(?i)^from:.*?sent:.*?to:.*?subject:.*?\n",
            r"(?i)^on.*?wrote:.*?\n",
            r"(?i)^_+[\s\S]*?from:.*?\n",
            r"(?i)-{3,}\s*forwarded by[\s\S]*?-{3,}",
            r"(?i)\s*original message\s*[\s\S]*?subject:.*?\n",
            r"(?i)forwarded by.*?on.*?\n",
            r"(?i)begin forwarded message:[\s\S]*?\n>",
            r"(?i)\n-{3,}\s*forwarded by[\s\S]*?\n",
            r"(?i)>\s*from:.*?\n>\s*sent:.*?\n>\s*to:.*?\n>\s*subject:.*?\n",
        ]

        ##################################################################
        # Enron-specific Patterns
        ##################################################################
        self._enron_patterns = [
            r"(?i)\t[A-Za-z ]+@[A-Za-z]+\n\t\d{2}/\d{2}/\d{4} \d{2}:\d{2} (am|pm)(\n\t\t[\s\S]*?)+?(?=\n\n)",
            r"(?i)(\t+[^\n]*\n){2,}",
            r"(?i)[A-Za-z]+ [A-Za-z]+/[A-Z]{3}/ect@ect",
            r"(?i)\t\t (to|cc|subject|fw): .*?\n",
            r"(?i)please respond to.*?\n",
        ]

        ##################################################################
        # Signature Patterns
        ##################################################################
        self._signature_patterns = [
            r"(?i)^--+\s*\n[\s\S]*$",
            r"(?i)^regards,[\s\S]*$",
            r"(?i)^sincerely,[\s\S]*$",
            r"(?i)^best( regards)?[\s\S]*$",
            r"(?i)^thanks(,| & regards|\s*\n)[\s\S]*$",
            r"(?i)^cheers,[\s\S]*$",
            r"(?i)\n\d+ [A-Za-z]+ (avenue|street|road|lane|drive|place|blvd)[\s\S]*?\n[A-Za-z]+, [A-Z]{2} \d{5}",
            r"(?i)\n(tel|phone|mobile|fax): [\d\-\+\.]+(\n|$)",
            r"(?i)\nemail: [^\s@]+@[^\s@]+\.[^\s@]+(\n|$)",
            r"(?i)this (email|e-mail) (and any attachments)?.*confidential",
            r"(?i)^warm regards,[\s\S]*$",
            r"(?i)^kind regards,[\s\S]*$",
            r"(?i)^yours (truly|sincerely),[\s\S]*$",
            r"(?i)^all the best,[\s\S]*$",
        ]

        ##################################################################
        # Attachment Reference Patterns
        ##################################################################
        self._attachment_patterns = [
            r"(?i)^\s*- .*?\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|txt)$",
            r"(?i)\[attachment:.*?\]",
            r"(?i)<attached:.*?>",
            r"(?i)see attached( file)?[:\.]",
        ]

        ##################################################################
        # Extra: Known Email CSV Metadata
        ##################################################################
        self._csv_metadata_patterns = [
            r"(?i)^Message-ID.*$",
            r"(?i)^Date.*$",
            r"(?i)^From.*$",
            r"(?i)^To.*$",
            r"(?i)^Cc.*$",
            r"(?i)^Subject.*$",
            r"(?i)^Content-Type.*$",
            r"(?i)^Folder.*$",
        ]

    ######################################################################
    # Individual cleaning steps
    ######################################################################
    def remove_disclaimers(self, text):
        original = text
        for pattern in self._disclaimer_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed disclaimers, saved {len(original) - len(text)} chars")
        return text

    def remove_promotions(self, text):
        original = text
        for pattern in self._promo_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed promotions, saved {len(original) - len(text)} chars")
        return text

    def remove_signatures(self, text):
        original = text
        for pattern in self._signature_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed signatures, saved {len(original) - len(text)} chars")
        return text

    def remove_attachment_references(self, text):
        original = text
        for pattern in self._attachment_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed attachment references, saved {len(original) - len(text)} chars")
        return text

    def remove_forwarding_headers(self, text):
        original = text
        for pattern in self._forwarding_headers:
            text = re.sub(pattern, "", text)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed forwarding headers, saved {len(original) - len(text)} chars")
        return text

    def remove_enron_formatting(self, text):
        original = text
        for pattern in self._enron_patterns:
            text = re.sub(pattern, "", text)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed Enron formatting, saved {len(original) - len(text)} chars")
        return text

    def remove_pii(self, text):
        original = text
        for pattern in self._pii_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed PII, changed {len(original) - len(text)} chars")
        return text

    def remove_urls(self, text):
        original = text
        # Broadened URL pattern
        url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.(com|org|net|edu|gov)(\/[^\s<>"]*)?)'
        text = re.sub(url_pattern, "", text, flags=re.IGNORECASE)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed URLs, changed {len(original) - len(text)} chars")
        return text

    def remove_numbers(self, text):
        """
        Remove or replace all numbers.
        We remove them completely (instead of placeholders),
        to avoid leftover numeric tokens interfering with authorship detection.
        """
        original = text
        phone_pattern = r'(\+\d{1,3}[-\.\s]?)?(\(?\d{3}\)?[-\.\s]?)?\d{3}[-\.\s]?\d{4}'
        text = re.sub(phone_pattern, '', text)

        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        text = re.sub(date_pattern, '', text)

        time_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\b'
        text = re.sub(time_pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'\b\d+\b', '', text)

        if self.debug_mode and original != text:
            self.logger.info(f"Removed numeric data, changed {len(original) - len(text)} chars")

        return text

    def remove_email_headers(self, text):
        """
        Aggressively remove lines that are clearly email headers.
        """
        original = text

        header_line_patterns = [
            r'^.*?(to|cc|bcc|from|subject|fw|re|sent|date|message-id):?.*$\n?'
            r'^.*?<.*?>.*$\n?',
            r'^.*?@.*$\n?',
            r'^>.*$\n?',
            r'.*?(wrote|says):.*$\n?',
            r'^.*?\d{2}/\d{2}/\d{4}.*$\n?',
        ]
        for pattern in header_line_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        artifact_patterns = [
            r'<[^>]*>',
            r'\[.*?\]',
            r'(?i)(to|cc|bcc|from|subject|fw|re)(\s*:|\s+)',
            r'[\w\.-]+@[\w\.-]+\.\w+',
        ]
        for pattern in artifact_patterns:
            text = re.sub(pattern, '', text)

        if self.debug_mode and original != text:
            self.logger.info(f"Removed email headers, saved {len(original) - len(text)} chars")
        return text

    def remove_csv_metadata_lines(self, text):
        """
        Remove lines that look like CSV headers or repeated metadata fields.
        E.g., 'Message-ID,Date,From,To,...'
        """
        original = text
        for pattern in self._csv_metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        if self.debug_mode and original != text:
            self.logger.info(f"Removed CSV-like metadata lines, saved {len(original) - len(text)} chars")
        return text

    def remove_forwarded_content(self, text):
        """
        Remove entire forwarded email chain if we detect typical
        forward indicators. We check if 'forwarded' is found and remove
        everything from that substring to the end (if it’s obviously a big block).
        """
        original = text
        forward_indicators = [
            "forwarded message",
            "forwarded by",
            "original message",
            "begin forwarded message:"
        ]
        for indicator in forward_indicators:
            idx = text.lower().find(indicator.lower())
            if idx != -1:
                if idx > len(text) * 0.1:
                    text = text[:idx].strip()
                    if self.debug_mode:
                        removed = len(original) - len(text)
                        self.logger.info(
                            f"Removed forwarded content starting at '{indicator}', removed {removed} chars"
                        )
                break
        return text

    def normalize_whitespace(self, text):
        """
        Final whitespace normalization.
        """
        original = text
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        if self.debug_mode and original != text:
            self.logger.info(f"Normalized whitespace, changed {len(original) - len(text)} chars")

        return text

    ######################################################################
    # PREPROCESSOR
    ######################################################################
    def preprocess(self, text):
        """
        Apply all preprocessing steps to an email with extremely aggressive cleaning.
        The result is a final text that is well-sanitized and ready for
        downstream tasks (like authorship detection).

        Args:
            text (str): The email text to preprocess.

        Returns:
            str: The preprocessed email text.
        """
        if not text or not isinstance(text, str):
            return ""

        original_length = len(text)
        original_text = text

        if self.debug_mode:
            self.logger.info(f"Starting preprocessing of text ({original_length} chars)")

        # ------------------------------------------------------------------
        # Lowercase
        # ------------------------------------------------------------------
        text = text.lower()

        # ------------------------------------------------------------------
        # Remove obvious CSV metadata lines
        # ------------------------------------------------------------------
        text = self.remove_csv_metadata_lines(text)

        # ------------------------------------------------------------------
        # Remove or mask explicit patterns
        # ------------------------------------------------------------------
        explicit_patterns = [
            r'priceless\.jpg',
            r'travis.*?kelley',
        ]
        for pattern in explicit_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        separators = ["_", "-", "=", "*", ":", ";", "&", "[", "]", "?", "!", "|"]
        for sep in separators:
            text = text.replace(sep, " ")

        # ------------------------------------------------------------------
        # Remove major forward blocks / quoted messages
        # ------------------------------------------------------------------
        text = self.remove_forwarded_content(text)
        text = self.remove_forwarding_headers(text)

        # ------------------------------------------------------------------
        # Remove disclaimers, promotions, signatures, attachments
        # ------------------------------------------------------------------
        text = self.remove_disclaimers(text)
        text = self.remove_promotions(text)
        text = self.remove_signatures(text)
        text = self.remove_attachment_references(text)

        # ------------------------------------------------------------------
        # Remove email headers, Enron-specific formatting
        # ------------------------------------------------------------------
        text = self.remove_email_headers(text)
        text = self.remove_enron_formatting(text)

        # ------------------------------------------------------------------
        # Remove PII, URLs, numbers
        # ------------------------------------------------------------------
        text = self.remove_pii(text)
        text = self.remove_urls(text)
        text = self.remove_numbers(text)

        # ------------------------------------------------------------------
        # Purge any leftover digits or words with digits
        # ------------------------------------------------------------------
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\b\w*\d+\w*\b', ' ', text)

        # ------------------------------------------------------------------
        # Keep only letters, apostrophes, and spaces. Remove others.
        # ------------------------------------------------------------------
        text = re.sub(r"[^a-z'\s]", " ", text)

        # ------------------------------------------------------------------
        # Split on lines, keep only lines with enough length
        # ------------------------------------------------------------------
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if len(line) > 10]
        text = " ".join(lines)

        # ------------------------------------------------------------------
        # Final whitespace normalization
        # ------------------------------------------------------------------
        text = self.normalize_whitespace(text)

        # ------------------------------------------------------------------
        # If the result is too short, minimal cleaning fallback
        # ------------------------------------------------------------------
        final_length = len(text)
        reduction = 0
        if original_length > 0:
            reduction = (original_length - final_length) / original_length * 100

        if self.debug_mode:
            self.logger.info(
                f"Preprocessing complete: {original_length} → {final_length} chars "
                f"({reduction:.1f}% reduction)"
            )

        if final_length < 30:
            if self.debug_mode:
                self.logger.warning("Result too short, applying minimal fallback")

            minimal_text = original_text.lower()
            minimal_text = re.sub(r'[^a-z\'\s]', ' ', minimal_text)
            minimal_text = re.sub(r'\s+', ' ', minimal_text).strip()

            minimal_text = re.sub(r'subject:.*?(\n|$)', '', minimal_text, flags=re.IGNORECASE)
            minimal_text = re.sub(r'from:.*?(\n|$)', '', minimal_text, flags=re.IGNORECASE)
            minimal_text = re.sub(r'to:.*?(\n|$)', '', minimal_text, flags=re.IGNORECASE)

            minimal_text = re.sub(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', '', minimal_text, flags=re.IGNORECASE)
            minimal_text = re.sub(r'\s+', ' ', minimal_text).strip()

            return minimal_text

        return text