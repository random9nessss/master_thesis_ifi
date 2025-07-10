import pandas as pd
import requests
from typing import List, Dict, Optional
import time
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------------------------------
# Scraping Country Ports from UNLOCODE: https://unece.org/trade/cefact/unlocode-code-list-country-and-territory
# -------------------------------------------------------------------

def get_iso_country_codes() -> List[str]:
    """
    Get list of all ISO 3166-1 alpha-2 country codes.
    This includes all countries that might have UN/LOCODE entries.
    """
    country_codes = [
        'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AQ', 'AR', 'AS', 'AT',
        'AU', 'AW', 'AX', 'AZ', 'BA', 'BB', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI',
        'BJ', 'BL', 'BM', 'BN', 'BO', 'BQ', 'BR', 'BS', 'BT', 'BV', 'BW', 'BY',
        'BZ', 'CA', 'CC', 'CD', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN',
        'CO', 'CR', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM',
        'DO', 'DZ', 'EC', 'EE', 'EG', 'EH', 'ER', 'ES', 'ET', 'FI', 'FJ', 'FK',
        'FM', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GL',
        'GM', 'GN', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU', 'GW', 'GY', 'HK', 'HM',
        'HN', 'HR', 'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR',
        'IS', 'IT', 'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 'KN',
        'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LS',
        'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MK',
        'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW',
        'MX', 'MY', 'MZ', 'NA', 'NC', 'NE', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP',
        'NR', 'NU', 'NZ', 'OM', 'PA', 'PE', 'PF', 'PG', 'PH', 'PK', 'PL', 'PM',
        'PN', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO', 'RS', 'RU', 'RW',
        'SA', 'SB', 'SC', 'SD', 'SE', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM',
        'SN', 'SO', 'SR', 'SS', 'ST', 'SV', 'SX', 'SY', 'SZ', 'TC', 'TD', 'TF',
        'TG', 'TH', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TV', 'TW',
        'TZ', 'UA', 'UG', 'UM', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG', 'VI',
        'VN', 'VU', 'WF', 'WS', 'YE', 'YT', 'ZA', 'ZM', 'ZW'
    ]
    return country_codes

def extract_unlocode_data(country_code: str, retry_count: int = 3) -> Optional[pd.DataFrame]:
    """
    Extract UN/LOCODE data for a specific country code.

    Args:
        country_code: Two-letter ISO country code
        retry_count: Number of retries on failure

    Returns:
        DataFrame with UN/LOCODE data or None if extraction fails
    """
    url = f"https://service.unece.org/trade/locode/{country_code.lower()}.htm"

    for attempt in range(retry_count):
        try:
            tables = pd.read_html(url)

            for i, table in enumerate(tables):
                if len(table.columns) >= 5 and len(table) > 1:

                    if table.iloc[0].notna().sum() >= 5:
                        df = table.copy()

                        df.columns = df.iloc[0]
                        df = df.drop(0).reset_index(drop=True)

                        df['Country'] = country_code

                        df.columns = [str(col).strip() for col in df.columns]

                        logging.info(f"Successfully extracted data for {country_code}: {len(df)} locations")
                        return df

            logging.warning(f"No suitable table found for {country_code}")
            return None

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {country_code}: {str(e)}")
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                logging.error(f"Failed to extract data for {country_code} after {retry_count} attempts")
                return None

def scrape_all_unlocode_data(
    country_codes: Optional[List[str]] = None,
    delay: float = 1.0,
    save_progress: bool = True,
    checkpoint_file: str = "unlocode_checkpoint.csv"
) -> pd.DataFrame:
    """
    Scrape UN/LOCODE data for all countries.

    Args:
        country_codes: List of country codes to process (None for all)
        delay: Delay between requests in seconds
        save_progress: Whether to save progress periodically
        checkpoint_file: File to save progress to

    Returns:
        Combined DataFrame with all UN/LOCODE data
    """
    if country_codes is None:
        country_codes = get_iso_country_codes()

    all_data = []
    failed_countries = []

    try:
        checkpoint_df = pd.read_csv(checkpoint_file)
        all_data.append(checkpoint_df)
        processed_countries = set(checkpoint_df['Country'].unique())
        country_codes = [cc for cc in country_codes if cc not in processed_countries]
        logging.info(f"Resuming from checkpoint. {len(processed_countries)} countries already processed.")
    except:
        logging.info("Starting fresh scraping session.")

    for i, country_code in enumerate(tqdm(country_codes, desc="Processing countries")):
        df = extract_unlocode_data(country_code)

        if df is not None and not df.empty:
            all_data.append(df)

            if save_progress and (i + 1) % 10 == 0:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df.to_csv(checkpoint_file, index=False)
                logging.info(f"Progress saved: {i + 1} countries processed")
        else:
            failed_countries.append(country_code)

        time.sleep(delay)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        final_df = clean_unlocode_data(final_df)

        logging.info(f"Successfully scraped {len(final_df)} locations from {len(all_data)} countries")
        if failed_countries:
            logging.warning(f"Failed to scrape data for: {', '.join(failed_countries)}")

        return final_df
    else:
        logging.error("No data was successfully scraped")
        return pd.DataFrame()

def clean_unlocode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the UN/LOCODE data.

    Args:
        df: Raw DataFrame with UN/LOCODE data

    Returns:
        Cleaned DataFrame
    """
    if 'LOCODE' in df.columns:
        df = df[df['LOCODE'].notna()].copy()

    column_mapping = {
        'Ch': 'Change',
        'LOCODE': 'LOCODE',
        'Name': 'Name',
        'NameWoDiacritics': 'NameWithoutDiacritics',
        'SubDiv': 'Subdivision',
        'Function': 'Function',
        'Status': 'Status',
        'Date': 'Date',
        'IATA': 'IATA',
        'Coordinates': 'Coordinates',
        'Remarks': 'Remarks'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    if 'Function' in df.columns:
        df['IsPort'] = df['Function'].astype(str).str.contains('1', na=False)
        df['IsAirport'] = df['Function'].astype(str).str.contains('4', na=False)
        df['IsMultimodal'] = df['Function'].astype(str).str.contains('6', na=False)

    if 'Coordinates' in df.columns:
        df = parse_coordinates(df)

    df = df.drop_duplicates(subset=['Country', 'LOCODE'], keep='first')
    df = df.sort_values(['Country', 'Name'])

    return df

def parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse coordinate string into separate latitude and longitude columns.

    Args:
        df: DataFrame with 'Coordinates' column

    Returns:
        DataFrame with added 'Latitude' and 'Longitude' columns
    """
    def extract_coords(coord_str):
        if pd.isna(coord_str):
            return pd.Series([None, None])

        try:
            coord_str = str(coord_str).strip()
            parts = coord_str.split()

            if len(parts) == 2:
                lat_str, lon_str = parts

                lat_deg = int(lat_str[:2])
                lat_min = int(lat_str[2:4])
                lat_dir = lat_str[-1]
                latitude = lat_deg + lat_min / 60
                if lat_dir == 'S':
                    latitude = -latitude

                lon_deg = int(lon_str[:3])
                lon_min = int(lon_str[3:5])
                lon_dir = lon_str[-1]
                longitude = lon_deg + lon_min / 60
                if lon_dir == 'W':
                    longitude = -longitude

                return pd.Series([latitude, longitude])
        except:
            pass

        return pd.Series([None, None])

    df[['Latitude', 'Longitude']] = df['Coordinates'].apply(extract_coords)

    return df

def extract_ports_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only port locations from the full UN/LOCODE dataset.

    Args:
        df: Full UN/LOCODE DataFrame

    Returns:
        DataFrame containing only ports
    """
    if 'IsPort' in df.columns:
        ports_df = df[df['IsPort'] == True].copy()
    elif 'Function' in df.columns:
        ports_df = df[df['Function'].astype(str).str.contains('1', na=False)].copy()
    else:
        logging.warning("Cannot identify ports - Function column not found")
        return df

    columns_to_keep = [
        'Country',
        'LOCODE',
        'Name',
        'NameWithoutDiacritics',
        'Subdivision',
        'Function',
        'IATA',
        'Latitude',
        'Longitude'
    ]

    columns_to_keep = [col for col in columns_to_keep if col in ports_df.columns]
    return ports_df[columns_to_keep]

if __name__ == "__main__":
    print("Starting UN/LOCODE scraping...")

    df = scrape_all_unlocode_data()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"unlocode_complete_{timestamp}.csv", index=False)
    print(f"Saved {len(df)} locations to unlocode_complete_{timestamp}.csv")

    ports_df = extract_ports_only(df)
    ports_df.to_csv(f"unlocode_ports_only_{timestamp}.csv", index=False)
    print(f"Saved {len(ports_df)} ports to unlocode_ports_only_{timestamp}.csv")

    print("\nSummary Statistics:")
    print(f"Total locations: {len(df)}")
    print(f"Countries: {df['Country'].nunique()}")
    if 'IsPort' in df.columns:
        print(f"Ports: {df['IsPort'].sum()}")
    if 'IsAirport' in df.columns:
        print(f"Airports: {df['IsAirport'].sum()}")

    if len(ports_df) > 0:
        print("\nSample port names for ML training:")
        print(ports_df[['Country', 'Name', 'LOCODE']].head(20))