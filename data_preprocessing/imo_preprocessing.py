import pandas as pd
import re
import string

def clean_vessel_names(df_vessel):
    """
    Clean and format vessel names, removing nonsense entries and properly formatting names.

    Args:
        df_vessel: DataFrame with 'name' and 'type' columns

    Returns:
        DataFrame with cleaned vessel names
    """

    def is_valid_vessel_name(name):
        """
        Determine if a vessel name is valid (not nonsense).

        Conservative filtering criteria:
        - Length between 4-50 characters
        - Not all same character repeated
        - Not common test patterns
        - Not excessive special characters
        - Has at least some alphabetic content
        """
        if pd.isna(name) or not isinstance(name, str):
            return False

        name = name.strip()

        if len(name) < 4 or len(name) > 50:
            return False

        if len(set(name.replace(' ', ''))) <= 2:
            return False

        test_patterns = [
            r'^TEST\d*$',
            r'^DENEME\d*$',
            r'^DEMO\d*$',
            r'^SAMPLE\d*$'
        ]
        for pattern in test_patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return False

        special_chars = sum(1 for c in name if c in '!@#$%^&*(){}[]|\\:";\'<>?,./`~=+')
        if special_chars > len(name) * 0.3:
            return False

        alpha_chars = sum(1 for c in name if c.isalpha())
        if alpha_chars < 2:
            return False

        nonsense_patterns = [
            r'^[A-Z]{1,3}$',
            r'^[A-Z]\d*$',
            r'^[A-Z]{2}\d*$'
        ]

        for pattern in nonsense_patterns[:2]:
            if re.match(pattern, name):
                return False

        if re.match(r'^[A-Z]{2}\d+$', name) and len(name) <= 4:
            return False

        return True

    def format_vessel_name(name):
        """
        Format vessel name from UPPERCASE to proper case.
        Handles special cases for maritime terminology.
        """
        if pd.isna(name) or not isinstance(name, str):
            return name

        name = name.strip()

        maritime_abbrevs = {
            'MS', 'MV', 'MT', 'SS', 'RV', 'FV', 'HMS', 'USNS', 'USCGC',
            'LNG', 'LPG', 'CEO', 'II', 'III', 'IV', 'VI', 'VII', 'VIII', 'IX', 'XI',
            'USA', 'UK', 'EU', 'UAE', 'US', 'FRC', 'WB2'
        }

        roman_pattern = r'\b(?:I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3})\b'

        words = name.split()
        formatted_words = []

        for word in words:
            if '(' in word and ')' in word:
                parts = re.split(r'([()])', word)
                formatted_parts = []
                for part in parts:
                    if part in '()':
                        formatted_parts.append(part)
                    elif part.upper() in maritime_abbrevs:
                        formatted_parts.append(part.upper())
                    else:
                        formatted_parts.append(part.title())
                formatted_words.append(''.join(formatted_parts))
            elif word.upper() in maritime_abbrevs:
                formatted_words.append(word.upper())
            elif re.match(roman_pattern, word, re.IGNORECASE):
                formatted_words.append(word.upper())
            elif any(c.isdigit() for c in word):
                if len(word) <= 4 and word.isalnum():
                    formatted_words.append(word.upper())
                else:
                    formatted_words.append(word.title())
            elif '-' in word:
                parts = word.split('-')
                formatted_parts = [part.title() for part in parts]
                formatted_words.append('-'.join(formatted_parts))
            else:
                formatted_words.append(word.title())

        return ' '.join(formatted_words)

    df_clean = df_vessel.copy()

    valid_mask = df_clean['name'].apply(is_valid_vessel_name)
    df_clean = df_clean[valid_mask].copy()

    df_clean['name'] = df_clean['name'].apply(format_vessel_name)

    df_clean = df_clean.drop_duplicates(subset=['name']).reset_index(drop=True)

    return df_clean

def analyze_filtering_results(df_original, df_clean):
    """
    Analyze what was filtered out and provide summary statistics.
    """
    print(f"Original dataset: {len(df_original)} vessels")
    print(f"Cleaned dataset: {len(df_clean)} vessels")
    print(f"Removed: {len(df_original) - len(df_clean)} vessels ({(len(df_original) - len(df_clean))/len(df_original)*100:.1f}%)")

    filtered_out = df_original[~df_original['name'].isin(df_clean['name'].str.upper())]['name'].unique()
    print(f"\nExamples of filtered out names:")
    for name in sorted(filtered_out)[:10]:
        print(f"  - {name}")

    if len(filtered_out) > 10:
        print(f"  ... and {len(filtered_out) - 10} more")

if __name__ == "__main__":
    df_vessel = pd.read_csv("/home/ANYACCESS.NET/brk.ch/Downloads/imo-vessel-codes.csv")[["name", "type"]]
    df_cleaned = clean_vessel_names(df_vessel)
    df_cleaned = df_cleaned[df_cleaned.type.isin(["Bulk Carrier", "Oil/Chemical Tanker"])]
    df_cleaned.name = df_cleaned.name.apply(lambda x: x.replace("/", "").replace("-", "").replace(".", "").replace("_", "").replace("!", "").replace(",", "").replace("  ", " ").strip())
    df_cleaned.to_csv("imo_vessel_data_cleaned.csv")

