import argparse
import json
import pandas as pd
import numpy as np
import re
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass, asdict
import sys
import hashlib

# -------------------------------------------------------------------
# Logging Config
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    min_length: int = 50
    max_length: int = 8192
    min_words: int = 10
    max_words: int = 1200
    min_sentences: int = 2

    duplicate_threshold: float = 0.85
    maritime_relevance_threshold: float = 0.005
    quality_score_threshold: float = -0.02

    remove_short_sentences: bool = True
    min_sentence_length: int = 15
    fix_encoding_issues: bool = True
    remove_citations: bool = True
    standardize_quotes: bool = True
    aggressive_cleaning: bool = True

    segment_long_texts: bool = True
    max_segment_words: int = 800

    save_statistics: bool = True
    save_rejected_texts: bool = False

    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MaritimeTextPreprocessor:
    """Comprehensive preprocessor for maritime domain texts"""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

        self.maritime_patterns = self._create_maritime_patterns()
        self.cleaning_patterns = self._create_cleaning_patterns()
        self.quality_patterns = self._create_quality_patterns()

        self.stats = {
            'total_input': 0,
            'after_initial_cleaning': 0,
            'after_quality_filter': 0,
            'after_deduplication': 0,
            'final_count': 0,
            'segments_created': 0,
            'rejected_too_short': 0,
            'rejected_too_long': 0,
            'rejected_low_quality': 0,
            'rejected_duplicates': 0,
            'rejected_non_maritime': 0
        }

    def _create_maritime_patterns(self) -> Dict[str, List[str]]:
        """Create patterns for maritime domain detection"""
        return {
            'measurements': [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:DWT|dwt|deadweight)\b',
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:TEU|teu)\b',
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:tonnes?|tons?|metric\s+tons?)\b',
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:metres?|meters?|ft|feet|km|kilometres?|kilometers?|mi|miles?)\b',
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:knots?|mph|km/h)\b'
            ],
            'currency_trade': [
                r'US\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand))?',
                r'\$\d+(?:,\d{3})*(?:\.\d+)?\s*(?:per|/)',
                r'LE\s*\d+(?:,\d{3})*(?:\.\d+)?',  # Egyptian Pounds
                r'cost\s+of.*?\$\d+(?:,\d{3})*(?:\.\d+)?'
            ],
            'maritime_terms': [
                r'\b(?:charter-?party|charterparty|charterer|shipowner)\b',
                r'\b(?:demurrage|laytime|despatch|freight)\b',
                r'\b(?:vessel|ship|tanker|carrier|bulk\s+carrier|container\s+ship)\b',
                r'\b(?:port|harbour|harbor|terminal|berth|wharf)\b',
                r'\b(?:cargo|freight|shipment|loading|unloading|discharge)\b',
                r'\b(?:maritime|shipping|navigation|seaborne)\b',
                r'\b(?:Incoterms?|FOB|CIF|FAS|CFR|EXW|DAP|DDP)\b',
                r'\b(?:bill\s+of\s+lading|B/L|BOL|waybill)\b'
            ],
            'ship_types': [
                r'\b(?:oil\s+tanker|chemical\s+tanker|gas\s+carrier|LPG\s+carrier|LNG\s+carrier)\b',
                r'\b(?:bulk\s+carrier|bulker|container\s+ship|boxship|containership)\b',
                r'\b(?:reefer\s+ship|RORO|ro-ro|roll-on/roll-off)\b',
                r'\b(?:multi-purpose\s+vessel|MPV|handysize|handymax|supramax)\b',
                r'\b(?:ULCC|VLCC|suezmax|aframax|panamax)\b'
            ],
            'geographical': [
                r'\b(?:Strait\s+of\s+Malacca|Malacca\s+Strait)\b',
                r'\b(?:Suez\s+Canal|Panama\s+Canal)\b',
                r'\b(?:Mediterranean\s+Sea|Red\s+Sea|Arabian\s+Sea|South\s+China\s+Sea)\b',
                r'\b(?:Port\s+Said|Port\s+Tewfik|Singapore|Rotterdam|Shanghai)\b'
            ]
        }

    def _create_cleaning_patterns(self) -> List[Tuple[str, str]]:
        """Create comprehensive cleaning patterns"""
        return [
            # -------------------------------------------------------------------
            # AGGRESSIVE COORDINATE CLEANING (MUST BE FIRST)
            # -------------------------------------------------------------------
            # First normalize all zero-width spaces
            (r'[\u200B\u200C\u200D\uFEFF]+', ' '),

            # PATTERN 1: Full Wikipedia coordinate format (all three parts)
            # Matches: 50°49′24.60″N 1°7′22.08″W﻿ / ﻿50.8235000°N 1.1228000°W﻿ / 50.8235000; -1.1228000
            (
            r"\d+°\d+[′']+(?:\d+(?:\.\d+)?[″\"′]+)?[NSEW]\s+\d+°\d+[′']+(?:\d+(?:\.\d+)?[″\"′]+)?[NSEW]\s*/\s*\d+\.\d+°?[NSEW]\s+\d+\.\d+°?[NSEW]\s*/\s*-?\d+\.\d+;\s*-?\d+\.\d+",
            ''),

            # PATTERN 2: Coordinates at the beginning with slash
            # Matches: 50°49′24.60′′N 1°7′22.08′′W / Portsmouth Harbour
            # Matches: 21°42′N 72°32′E / Dahej
            (r"^\s*\d+°\d+[′']+(?:\d+(?:\.\d+)?[″\"′]+)?[NSEW]\s+\d+°\d+[′']+(?:\d+(?:\.\d+)?[″\"′]+)?[NSEW]\s*/", ''),

            # PATTERN 3: Coordinates anywhere followed by slash
            (r"\d+°\d+[′']+(?:\d+(?:\.\d+)?[″\"′]+)?[NSEW]\s+\d+°\d+[′']+(?:\d+(?:\.\d+)?[″\"′]+)?[NSEW]\s*/", ''),

            # PATTERN 4: DMS format (degrees, minutes, seconds) - handle all quote variations
            (r"\d+°\d+[′']+\d+(?:\.\d+)?[″\"′]+[NSEW]\s+\d+°\d+[′']+\d+(?:\.\d+)?[″\"′]+[NSEW]", ''),

            # PATTERN 5: DM format (degrees, minutes only)
            (r"\d+°\d+[′']+[NSEW]\s+\d+°\d+[′']+[NSEW]", ''),

            # PATTERN 6: Decimal degrees with letters
            (r'\d+\.\d+°[NSEW]\s+\d+\.\d+°[NSEW]', ''),

            # PATTERN 7: Decimal coordinates with semicolon (with optional negative)
            (r'-?\d+\.\d+;\s*-?\d+\.\d+', ''),

            # PATTERN 8: Slash followed by decimal coordinates
            (r'/\s*\d+\.\d+°?[NSEW]?\s+\d+\.\d+°?[NSEW]?', ''),
            (r'/\s*-?\d+\.\d+;\s*-?\d+\.\d+', ''),

            # PATTERN 9: Single coordinate fragments
            (r"\d+°\d+[′']+\d+(?:\.\d+)?[″\"′]+[NSEW]", ''),
            (r"\d+°\d+[′']+[NSEW]", ''),
            (r'\d+\.\d+°[NSEW]', ''),

            # PATTERN 10: Clean up any remaining slashes at start of text/line
            (r'^\s*/\s*', ''),
            (r'\n\s*/\s*', '\n'),

            # PATTERN 11: Aggressive - remove any line that starts with coordinates
            (r"^.*?\d+°\d+[′']+.*?[NSEW].*?$", ''),

            # -------------------------------------------------------------------
            # Foreign Language Content in Parentheses
            # -------------------------------------------------------------------
            (
            r'\([^)]*(?:Arabic|French|German|Spanish|Italian|Portuguese|Dutch|Chinese|Japanese|Korean|Hindi|Russian):[^)]*\)',
            ''),
            (r'\([^)]*[\u0600-\u06FF\u0750-\u077F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u0400-\u04FF][^)]*\)', ''),

            # -------------------------------------------------------------------
            # Pronunciation Patterns (IPA notation)
            # -------------------------------------------------------------------
            (r'/[ˈˌəɪiːɛæɑɒɔʊuːʌɜːɪəeɪaɪɔɪaʊɪəeəʊəbdfghjklmnpqrstvwxzʃʒθðŋʧʤ]+/', ''),

            # -------------------------------------------------------------------
            # Leading/Trailing Quotes
            # -------------------------------------------------------------------
            (r'^["\'""\u2018\u2019\u201C\u201D\u201E\u201A\u00AB\u00BB]+\s*', ''),
            (r'\s*["\'""''„‚«»]+$', ''),

            # -------------------------------------------------------------------
            # Multiple Newlines
            # -------------------------------------------------------------------
            (r'\n{3,}', '\n\n'),
            (r'^\n+|\n+$', ''),
            (r'\\n', '\n'),

            # -------------------------------------------------------------------
            # Wikipedia Artifacts
            # -------------------------------------------------------------------
            (r'\{\{[^}]*\}\}', ''),
            (r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2'),
            (r'\[http[^\s\]]*\s*([^\]]*)\]', r'\1'),
            (r'<ref[^>]*>.*?</ref>', ''),
            (r'<ref[^>]*/?>', ''),
            (r'<[^>]+>', ''),

            # -------------------------------------------------------------------
            # HTML Entities
            # -------------------------------------------------------------------
            (r'&nbsp;', ' '),
            (r'&amp;', '&'),
            (r'&lt;', '<'),
            (r'&gt;', '>'),
            (r'&quot;', '"'),
            (r'&apos;', "'"),
            (r'&#\d+;', ''),

            # -------------------------------------------------------------------
            # Citation Patterns
            # -------------------------------------------------------------------
            (r'\[\d+\]', ''),
            (r'\(citation\s+needed\)', ''),
            (r'\(verify\)', ''),
            (r'\(disambiguation\)', ''),

            # -------------------------------------------------------------------
            # Encoding Artifacts
            # -------------------------------------------------------------------
            (r'â€™', "'"),
            (r'â€œ', '"'),
            (r'â€', '"'),
            (r'â€"', '–'),
            (r'â€"', '—'),
            (r'Â', ''),
            (r'â€¦', '...'),

            # -------------------------------------------------------------------
            # Final Cleaning
            # -------------------------------------------------------------------
            (r'[ \t]+', ' '),
            (r' +', ' '),
            (r'^\s+|\s+$', ''),
        ]

    def _create_quality_patterns(self) -> Dict[str, List[str]]:
        """Create patterns for quality assessment"""
        return {
            'good_indicators': [
                r'\b(?:contract|agreement|terms|conditions)\b',
                r'\b(?:operated|maintained|constructed|designed)\b',
                r'\b(?:capacity|efficiency|transport|carriage)\b',
                r'\b(?:international|commercial|trade|business)\b',
                r'\b(?:according\s+to|under\s+the|in\s+accordance\s+with)\b'
            ],
            'bad_indicators': [
                r'click\s+here|read\s+more|see\s+also|more\s+information',
                r'this\s+article|this\s+page|above\s+mentioned|as\s+mentioned',
                r'copyright|all\s+rights\s+reserved|terms\s+of\s+use',
                r'^\s*$|^\.+$|^\d+\.$',
                r'lorem\s+ipsum|placeholder|example\s+text',
                r'edit\s+this\s+page|update\s+needed|stub\s+article'
            ],
            'structural_issues': [
                r'^[^a-zA-Z]*$',
                r'^.{1,10}$',
                r'(.)\1{10,}',
                r'[^\w\s\.,;:!?()-]{5,}',
            ]
        }

    def clean_text(self, text: str) -> str:
        """Apply comprehensive text cleaning"""
        if not text or not isinstance(text, str):
            return ""

        original_text = text

        try:

            if self.config.fix_encoding_issues:
                text = self._fix_encoding(text)

            # -------------------------------------------------------------------
            # Cleaning Patterns
            # -------------------------------------------------------------------
            for pattern, replacement in self.cleaning_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            if self.config.standardize_quotes:
                text = self._standardize_quotes(text)

            if self.config.aggressive_cleaning:
                text = self._aggressive_clean(text)

            return text

        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return original_text

    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues"""
        try:
            text = unicodedata.normalize('NFKC', text)

            if '\\x' in text:
                try:
                    text = text.encode().decode('unicode_escape').encode('latin1').decode('utf-8')
                except:
                    pass

            return text
        except Exception as e:
            logger.warning(f"Error fixing encoding: {e}")
            return text

    def _standardize_quotes(self, text: str) -> str:
        """Standardize quote marks"""
        quote_map = {
            '"': '"', '"': '"',
            ''': "'", ''': "'",
            '«': '"', '»': '"',
            '„': '"', '‚': "'",
            '‹': "'", '›': "'",
        }

        for smart, standard in quote_map.items():
            text = text.replace(smart, standard)

        return text

    def _aggressive_clean(self, text: str) -> str:
        """Additional aggressive cleaning"""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if (len(line) > 10 and
                    not re.match(r'^\d+\.?\s*$', line) and
                    not re.match(r'^[^\w]*$', line)):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def extract_sentences(self, text: str) -> List[str]:
        """Extract well-formed sentences"""
        if not text:
            return []

        maritime_abbrevs = [
            'DWT', 'TEU', 'FOB', 'CIF', 'FAS', 'CFR', 'Ltd', 'Co', 'Inc', 'Corp',
            'vs', 'etc', 'Mr', 'Mrs', 'Dr', 'Prof', 'St', 'Ave', 'Blvd',
            'U.S', 'U.K', 'i.e', 'e.g', 'SCA', 'ICC', 'MARPOL', 'IMO', 'SOLAS'
        ]

        protected_text = text
        abbrev_map = {}
        for i, abbrev in enumerate(maritime_abbrevs):
            placeholder = f"__ABBREV_{i}__"
            abbrev_map[placeholder] = abbrev
            protected_text = re.sub(
                re.escape(abbrev) + r'\.?',
                placeholder,
                protected_text,
                flags=re.IGNORECASE
            )

        sentences = re.split(
            r'[.!?]+(?=\s+[A-Z])',
            protected_text
        )

        cleaned_sentences = []
        for sentence in sentences:
            for placeholder, abbrev in abbrev_map.items():
                sentence = sentence.replace(placeholder, abbrev)

            sentence = sentence.strip()
            if (len(sentence) >= self.config.min_sentence_length and
                    self._is_valid_sentence(sentence)):
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _is_valid_sentence(self, sentence: str) -> bool:
        """Check if sentence is valid and meaningful"""
        # Basic length check
        if len(sentence.split()) < 3:
            return False

        for pattern in self.quality_patterns['structural_issues']:
            if re.search(pattern, sentence, re.IGNORECASE):
                return False

        for pattern in self.quality_patterns['bad_indicators']:
            if re.search(pattern, sentence, re.IGNORECASE):
                return False

        if not re.search(r'[a-zA-Z]{3,}', sentence):
            return False

        return True

    def calculate_maritime_relevance(self, text: str) -> float:
        """Calculate how relevant text is to maritime domain"""
        if not text:
            return 0.0

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        total_matches = 0
        for category, patterns in self.maritime_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))

                if category in ['maritime_terms', 'ship_types']:
                    matches *= 2
                total_matches += matches

        return total_matches / word_count

    def calculate_quality_score(self, text: str) -> float:
        """Calculate overall text quality score"""
        if not text:
            return -1.0

        word_count = len(text.split())
        if word_count == 0:
            return -1.0

        good_score = sum(len(re.findall(p, text, re.IGNORECASE))
                         for p in self.quality_patterns['good_indicators'])
        bad_score = sum(len(re.findall(p, text, re.IGNORECASE))
                        for p in self.quality_patterns['bad_indicators'])

        normalized_good = good_score / word_count
        normalized_bad = bad_score / word_count

        return normalized_good - normalized_bad

    def validate_text(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive text validation"""
        metrics = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(self.extract_sentences(text)),
            'maritime_relevance': self.calculate_maritime_relevance(text),
            'quality_score': self.calculate_quality_score(text),
            'rejection_reason': None
        }

        if metrics['length'] < self.config.min_length:
            metrics['rejection_reason'] = 'too_short'
            return False, metrics

        if metrics['length'] > self.config.max_length:
            metrics['rejection_reason'] = 'too_long'
            return False, metrics

        if metrics['word_count'] < self.config.min_words:
            metrics['rejection_reason'] = 'too_few_words'
            return False, metrics

        if metrics['word_count'] > self.config.max_words:
            metrics['rejection_reason'] = 'too_many_words'
            return False, metrics

        if metrics['sentence_count'] < self.config.min_sentences:
            metrics['rejection_reason'] = 'too_few_sentences'
            return False, metrics

        if metrics['maritime_relevance'] < self.config.maritime_relevance_threshold:
            metrics['rejection_reason'] = 'not_maritime_relevant'
            return False, metrics

        if metrics['quality_score'] < self.config.quality_score_threshold:
            metrics['rejection_reason'] = 'low_quality'
            return False, metrics

        return True, metrics

    def detect_duplicates(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        """Detect near-duplicate texts using shingling"""

        def get_shingles(text: str, k: int = 3) -> set:
            words = text.lower().split()
            return set(' '.join(words[i:i + k]) for i in range(len(words) - k + 1))

        duplicates = []
        text_shingles = [get_shingles(text) for text in texts]

        for i in range(len(texts)):
            if not text_shingles[i]:
                continue
            for j in range(i + 1, len(texts)):
                if not text_shingles[j]:
                    continue

                intersection = len(text_shingles[i] & text_shingles[j])
                union = len(text_shingles[i] | text_shingles[j])
                similarity = intersection / union if union > 0 else 0

                if similarity >= self.config.duplicate_threshold:
                    duplicates.append((i, j, similarity))

        return duplicates

    def segment_text(self, text: str) -> List[str]:
        """Segment long text into smaller coherent pieces"""
        if not self.config.segment_long_texts:
            return [text]

        sentences = self.extract_sentences(text)
        if not sentences:
            return []

        segments = []
        current_segment = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if (current_word_count + sentence_words > self.config.max_segment_words
                    and current_segment):
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_word_count = sentence_words
            else:
                current_segment.append(sentence)
                current_word_count += sentence_words

        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def preprocess_dataset(self, input_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process entire dataset"""
        logger.info(f"Starting preprocessing of {len(input_data)} texts")

        self.stats['total_input'] = len(input_data)
        processed_texts = []
        rejected_texts = []

        # Phase 1: Clean and validate individual texts
        logger.info("Phase 1: Cleaning and validation...")
        for idx, item in enumerate(input_data):
            if 'content' not in item:
                logger.warning(f"Item {idx} missing 'content' field")
                continue

            cleaned_text = self.clean_text(item['content'])
            if not cleaned_text:
                continue

            is_valid, metrics = self.validate_text(cleaned_text)

            if is_valid:
                segments = self.segment_text(cleaned_text)
                self.stats['segments_created'] += len(segments) - 1

                for seg_idx, segment in enumerate(segments):
                    processed_texts.append({
                        'content': segment,
                        'original_index': idx,
                        'segment_index': seg_idx,
                        'word_count': len(segment.split()),
                        'maritime_relevance': metrics['maritime_relevance'],
                        'quality_score': metrics['quality_score']
                    })
            else:
                reason = metrics['rejection_reason']
                if reason == 'too_short' or reason == 'too_few_words':
                    self.stats['rejected_too_short'] += 1
                elif reason == 'too_long' or reason == 'too_many_words':
                    self.stats['rejected_too_long'] += 1
                elif reason == 'low_quality':
                    self.stats['rejected_low_quality'] += 1
                elif reason == 'not_maritime_relevant':
                    self.stats['rejected_non_maritime'] += 1

                if self.config.save_rejected_texts:
                    rejected_texts.append({
                        'content': cleaned_text,
                        'original_index': idx,
                        'rejection_reason': reason,
                        'metrics': metrics
                    })

        self.stats['after_initial_cleaning'] = len(processed_texts)
        logger.info(f"After cleaning: {len(processed_texts)} texts")

        if len(processed_texts) > 1:
            logger.info("Phase 2: Duplicate detection...")
            contents = [item['content'] for item in processed_texts]
            duplicates = self.detect_duplicates(contents)

            duplicate_indices = set()
            for _, j, similarity in duplicates:
                duplicate_indices.add(j)
                logger.debug(f"Duplicate found: indices with similarity {similarity:.3f}")

            processed_texts = [item for i, item in enumerate(processed_texts)
                               if i not in duplicate_indices]
            self.stats['rejected_duplicates'] = len(duplicate_indices)

        self.stats['after_deduplication'] = len(processed_texts)
        self.stats['final_count'] = len(processed_texts)

        logger.info(f"Final dataset size: {len(processed_texts)} texts")

        return processed_texts, rejected_texts

    def print_statistics(self):
        """Print comprehensive preprocessing statistics"""
        print("\n" + "=" * 60)
        print("PREPROCESSING STATISTICS")
        print("=" * 60)

        print(f"Total input texts:          {self.stats['total_input']:>6}")
        print(f"After cleaning:             {self.stats['after_initial_cleaning']:>6}")
        print(f"After deduplication:        {self.stats['after_deduplication']:>6}")
        print(f"Final count:                {self.stats['final_count']:>6}")
        print(f"Segments created:           {self.stats['segments_created']:>6}")

        print("\nREJECTION BREAKDOWN:")
        print(f"Too short:                  {self.stats['rejected_too_short']:>6}")
        print(f"Too long:                   {self.stats['rejected_too_long']:>6}")
        print(f"Low quality:                {self.stats['rejected_low_quality']:>6}")
        print(f"Not maritime relevant:      {self.stats['rejected_non_maritime']:>6}")
        print(f"Duplicates:                 {self.stats['rejected_duplicates']:>6}")

        total_rejected = (self.stats['rejected_too_short'] +
                          self.stats['rejected_too_long'] +
                          self.stats['rejected_low_quality'] +
                          self.stats['rejected_non_maritime'] +
                          self.stats['rejected_duplicates'])

        print(f"Total rejected:             {total_rejected:>6}")

        if self.stats['total_input'] > 0:
            retention_rate = (self.stats['final_count'] / self.stats['total_input']) * 100
            print(f"Retention rate:             {retention_rate:>5.1f}%")

        print("=" * 60)


def load_data(filepath: str) -> List[Dict]:
    """Load data from various file formats"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    if filepath.suffix.lower() == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON file must contain a list or single dictionary")

        return data

    elif filepath.suffix.lower() == '.jsonl':
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_data(data: List[Dict], filepath: str):
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess maritime domain texts for MLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_maritime_data.py input.json output.json
  python preprocess_maritime_data.py raw_data.json clean_data.json --config custom_config.json
  python preprocess_maritime_data.py data.json processed.json --save-config config.json
        """
    )

    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--config', help='Configuration JSON file path')
    parser.add_argument('--save-config', help='Save configuration to specified path')
    parser.add_argument('--save-rejected', help='Save rejected texts to specified path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.config:
        config = PreprocessingConfig.load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = PreprocessingConfig()
        logger.info("Using default configuration")

    if args.save_config:
        config.save_config(args.save_config)
        logger.info(f"Configuration saved to {args.save_config}")

    if args.save_rejected:
        config.save_rejected_texts = True

    try:
        logger.info(f"Loading data from {args.input_file}...")
        input_data = load_data(args.input_file)
        logger.info(f"Loaded {len(input_data)} items")

        preprocessor = MaritimeTextPreprocessor(config)
        processed_data, rejected_data = preprocessor.preprocess_dataset(input_data)

        save_data(processed_data, args.output_file)
        logger.info(f"Processed data saved to {args.output_file}")

        if args.save_rejected and rejected_data:
            save_data(rejected_data, args.save_rejected)
            logger.info(f"Rejected data saved to {args.save_rejected}")

        preprocessor.print_statistics()

        if config.save_statistics:
            stats_file = Path(args.output_file).with_suffix('.stats.json')
            with open(stats_file, 'w') as f:
                json.dump(preprocessor.stats, f, indent=2)
            logger.info(f"Statistics saved to {stats_file}")

        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()