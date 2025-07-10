"""
Wikipedia Maritime Article Cleaner
---------------------------------
This script processes Wikipedia maritime articles by cleaning text,
normalizing measurement units, and saving the processed data to a JSON file.
"""

import re
import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_measurement_units(text):
    """
    Find numbers followed by units and join them together by removing the space.

    Args:
        text (str): Input text with potential spaced units

    Returns:
        str: Text with normalized measurement units
    """
    units = [
        'm', 'km', 'cm', 'mm', 'nm', 'ft', 'in', 'yd', 'mi', 'nmi',
        'kg', 'g', 'mg', 't', 'lb', 'oz', 'ton', 'tonne',
        'L', 'l', 'ml', 'gal', 'qt', 'pt', 'fl oz',
        'mph', 'km/h', 'm/s', 'kn', 'knots',
        'W', 'kW', 'MW', 'GW', 'V', 'A', 'kWh', 'MWh',
        '°C', '°F', 'K', 'Hz', 'kHz', 'MHz', 'GHz',
        'N', 'Pa', 'kPa', 'MPa', 'GPa', 'bar', 'atm', 'psi',
        's', 'min', 'h', 'd', 'yr',
        'B', 'KB', 'MB', 'GB', 'TB', 'bps', 'kbps', 'Mbps', 'Gbps',

        'DWT', 'TEU', 'GT', 'GRT'
    ]

    escaped_units = [re.escape(unit) for unit in units]
    unit_pattern = '|'.join(escaped_units)

    pattern = rf'(\d+(?:\.\d+)?)\s+({unit_pattern})\b'
    normalized_text = re.sub(pattern, r'\1\2', text)

    return normalized_text


def clean_wikipedia_text(text):
    """
    Enhanced text cleaning for Wikipedia text with unit normalization

    Args:
        text (str): Raw Wikipedia text

    Returns:
        str: Cleaned text with normalized formatting
    """
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Removing Coordinates
    text = re.sub(
        r'(\d{1,3})°(\d{1,2})′([\d.]+)″([NS])\s+(\d{1,3})°(\d{1,2})′([\d.]+)″([EW]).*?(\d+\.\d+)°([NS]).*?(\d+\.\d+)°([EW]).*?(\d+\.\d+);\s*(-\d+\.\d+)',
        '', text)

    # Remove Brackets
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)

    # Remove Wikipedia citation markup
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(citation needed\)', '', text, flags=re.IGNORECASE)

    # Normalize Unicode characters
    text = text.replace('–', '-').replace('—', '-')

    # Fix spacing artifacts
    text = re.sub(r'\s{2,}', ' ', text)

    # Normalize units
    text = normalize_measurement_units(text)

    # Pronounciation hints
    text = re.sub(r'\/(.)+\/', '', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.;:])', r'\1', text)

    return text.strip()


def process_wikipedia_articles(input_path, output_path):
    """
    Process Wikipedia articles by cleaning text and normalizing content

    Args:
        input_path (str): Path to input JSON file with Wikipedia articles
        output_path (str): Path where processed data will be saved
    """
    try:
        logger.info(f"Reading data from {input_path}")
        df = pd.read_json(input_path)

        df = df[["title", "url", "content", "categories"]]

        logger.info("Cleaning text content")
        df.content = df.content.apply(clean_wikipedia_text)

        df.categories = df.categories.apply(lambda x: "|".join(x))

        initial_count = len(df)
        df.drop_duplicates(subset=["title", "content"], inplace=True)

        logger.info(f"Removed {initial_count - len(df)} duplicate entries")

        logger.info(f"Saving processed data to {output_path}")
        df.to_json(output_path, orient="records")

        logger.info(f"Successfully processed {len(df)} articles")

    except Exception as e:
        logger.error(f"Error processing Wikipedia articles: {str(e)}")
        raise


def main(input_path: str, output_path: str):
    logger.info("Starting Wikipedia article processing")
    process_wikipedia_articles(input_path, output_path)
    logger.info("Processing complete")
