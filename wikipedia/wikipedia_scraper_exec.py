import os
import time
from datetime import datetime
import argparse

from wikipedia_scraper import WikipediaCharteringScraper


def run_large_scale_scraper(max_pages=10000):
    """Run the scraper with expanded seed URLs to target 10,000 articles"""

    scraper = WikipediaCharteringScraper()

    # -------------------------------------------------------------------
    # Chartering Related Seed URLS
    # -------------------------------------------------------------------
    seed_urls = [
        # Charter-specific URLs
        "https://en.wikipedia.org/wiki/Charterparty",
        "https://en.wikipedia.org/wiki/Ship_chartering",
        "https://en.wikipedia.org/wiki/Time_charter",
        "https://en.wikipedia.org/wiki/Voyage_charter",
        "https://en.wikipedia.org/wiki/Bareboat_charter",
        "https://en.wikipedia.org/wiki/Demurrage",
        "https://en.wikipedia.org/wiki/Laycan",
        "https://en.wikipedia.org/wiki/Laytime",
        "https://en.wikipedia.org/wiki/Charter_rate",

        # Trade and shipping documents
        "https://en.wikipedia.org/wiki/Incoterms",
        "https://en.wikipedia.org/wiki/Bill_of_lading",
        "https://en.wikipedia.org/wiki/Sea_waybill",
        "https://en.wikipedia.org/wiki/Letter_of_credit",
        "https://en.wikipedia.org/wiki/Marine_insurance",
        "https://en.wikipedia.org/wiki/Average_adjusters",
        "https://en.wikipedia.org/wiki/General_average",

        # Categories
        "https://en.wikipedia.org/wiki/Category:Shipping",
        "https://en.wikipedia.org/wiki/Category:Ship_management",
        "https://en.wikipedia.org/wiki/Category:Maritime_transport",
        "https://en.wikipedia.org/wiki/Category:International_trade",
        "https://en.wikipedia.org/wiki/Category:Freight_transport",
        "https://en.wikipedia.org/wiki/Category:Merchant_ships",
        "https://en.wikipedia.org/wiki/Category:Naval_architecture",
        "https://en.wikipedia.org/wiki/Category:Ship_types",
        "https://en.wikipedia.org/wiki/Category:Admiralty_law",
        "https://en.wikipedia.org/wiki/Category:Port_operations",
        "https://en.wikipedia.org/wiki/Category:Ship_measurement",

        # Ship types
        "https://en.wikipedia.org/wiki/Bulk_carrier",
        "https://en.wikipedia.org/wiki/Oil_tanker",
        "https://en.wikipedia.org/wiki/Container_ship",
        "https://en.wikipedia.org/wiki/Chemical_tanker",
        "https://en.wikipedia.org/wiki/LNG_carrier",
        "https://en.wikipedia.org/wiki/LPG_carrier",
        "https://en.wikipedia.org/wiki/Crude_oil_tanker",
        "https://en.wikipedia.org/wiki/Handysize",
        "https://en.wikipedia.org/wiki/Multi-purpose_vessel",
        "https://en.wikipedia.org/wiki/Reefer_ship",
        "https://en.wikipedia.org/wiki/Roll-on/roll-off",

        # Trade and shipping concepts
        "https://en.wikipedia.org/wiki/Tramp_trade",
        "https://en.wikipedia.org/wiki/Liner_service",
        "https://en.wikipedia.org/wiki/Freight_rate",
        "https://en.wikipedia.org/wiki/Bunker_fuel",
        "https://en.wikipedia.org/wiki/Dead_Freight",
        "https://en.wikipedia.org/wiki/Deadweight_tonnage",
        "https://en.wikipedia.org/wiki/Gross_tonnage",
        "https://en.wikipedia.org/wiki/Net_tonnage",

        # Legal and regulatory
        "https://en.wikipedia.org/wiki/Maritime_law",
        "https://en.wikipedia.org/wiki/Carriage_of_goods_by_sea",
        "https://en.wikipedia.org/wiki/Hague-Visby_Rules",
        "https://en.wikipedia.org/wiki/Hamburg_Rules",
        "https://en.wikipedia.org/wiki/Rotterdam_Rules",
        "https://en.wikipedia.org/wiki/International_Maritime_Organization",
        "https://en.wikipedia.org/wiki/London_Maritime_Arbitrators_Association",

        # General maritime
        "https://en.wikipedia.org/wiki/Maritime_transport",
        "https://en.wikipedia.org/wiki/Shipping_container",
        "https://en.wikipedia.org/wiki/Cargo_ship",
        "https://en.wikipedia.org/wiki/Ship_registration",
        "https://en.wikipedia.org/wiki/Flag_of_convenience",

        # Important routes and chokepoints
        "https://en.wikipedia.org/wiki/Cape_Route",
        "https://en.wikipedia.org/wiki/Strait_of_Malacca",
        "https://en.wikipedia.org/wiki/Suez_Canal",
        "https://en.wikipedia.org/wiki/Panama_Canal",
        "https://en.wikipedia.org/wiki/Panamax",
        "https://en.wikipedia.org/wiki/Suezmax",
        "https://en.wikipedia.org/wiki/Strait_of_Hormuz",

        # Ports and terminals
        "https://en.wikipedia.org/wiki/Port_of_Singapore",
        "https://en.wikipedia.org/wiki/Port_of_Rotterdam",
        "https://en.wikipedia.org/wiki/Port_of_Shanghai",
        "https://en.wikipedia.org/wiki/Container_terminal",
        "https://en.wikipedia.org/wiki/Oil_terminal",

        # Major shipping companies and institutions
        "https://en.wikipedia.org/wiki/Maersk",
        "https://en.wikipedia.org/wiki/CMA_CGM",
        "https://en.wikipedia.org/wiki/Mediterranean_Shipping_Company",
        "https://en.wikipedia.org/wiki/Hapag-Lloyd",
        "https://en.wikipedia.org/wiki/COSCO",
        "https://en.wikipedia.org/wiki/Baltic_Exchange",
        "https://en.wikipedia.org/wiki/BIMCO",
        "https://en.wikipedia.org/wiki/Classification_society",
        "https://en.wikipedia.org/wiki/Lloyd%27s_Register",
        "https://en.wikipedia.org/wiki/Lloyd%27s_of_London",

        # Shipping markets and insurance
        "https://en.wikipedia.org/wiki/Shipping_market",
        "https://en.wikipedia.org/wiki/Protection_and_indemnity_insurance",
        "https://en.wikipedia.org/wiki/Marine_insurance",
        "https://en.wikipedia.org/wiki/Hull_insurance",

        # Maritime incidents
        "https://en.wikipedia.org/wiki/Maritime_piracy",
        "https://en.wikipedia.org/wiki/Shipwreck",
        "https://en.wikipedia.org/wiki/Marine_salvage",

        # Commodities commonly shipped
        "https://en.wikipedia.org/wiki/Iron_ore",
        "https://en.wikipedia.org/wiki/Coal",
        "https://en.wikipedia.org/wiki/Crude_oil",
        "https://en.wikipedia.org/wiki/Liquefied_natural_gas",
        "https://en.wikipedia.org/wiki/Grain",
        "https://en.wikipedia.org/wiki/Bauxite",
        "https://en.wikipedia.org/wiki/Phosphate",

        # Ship crew and operation
        "https://en.wikipedia.org/wiki/Ship%27s_captain",
        "https://en.wikipedia.org/wiki/Deck_department",
        "https://en.wikipedia.org/wiki/Engine_department",
        "https://en.wikipedia.org/wiki/Ship%27s_agent",
        "https://en.wikipedia.org/wiki/Stevedore",

        # Shipbuilding
        "https://en.wikipedia.org/wiki/Shipbuilding",
        "https://en.wikipedia.org/wiki/Naval_architecture",
        "https://en.wikipedia.org/wiki/Ship_design",


        # Chatering Terms and Standard Contract Forms
        "https://en.wikipedia.org/wiki/ASBATANKVOY",
        "https://en.wikipedia.org/wiki/BIMCHEMVOY",
        "https://en.wikipedia.org/wiki/SHELLVOY",
        "https://en.wikipedia.org/wiki/GENCON",
        "https://en.wikipedia.org/wiki/NYPE_93",
        "https://en.wikipedia.org/wiki/BALTIME",
        "https://en.wikipedia.org/wiki/BOXTIME",
        "https://en.wikipedia.org/wiki/SUPPLYTIME",
        "https://en.wikipedia.org/wiki/Notice_of_readiness",
        "https://en.wikipedia.org/wiki/Free_In_Out_(FIO)",

        # Major Ports
        "https://en.wikipedia.org/wiki/Port_of_Hong_Kong",
        "https://en.wikipedia.org/wiki/Port_of_Busan",
        "https://en.wikipedia.org/wiki/Port_of_Los_Angeles",
        "https://en.wikipedia.org/wiki/Port_of_New_York_and_New_Jersey",
        "https://en.wikipedia.org/wiki/Port_of_Antwerp",
        "https://en.wikipedia.org/wiki/Port_of_Hamburg",
        "https://en.wikipedia.org/wiki/Port_of_Dubai",
        "https://en.wikipedia.org/wiki/Port_of_Ningbo-Zhoushan",
        "https://en.wikipedia.org/wiki/Port_of_Guangzhou",
        "https://en.wikipedia.org/wiki/Port_of_Shenzhen",

        # Commodity Trading Terms
        "https://en.wikipedia.org/wiki/Commodity_trading",
        "https://en.wikipedia.org/wiki/Futures_contract",
        "https://en.wikipedia.org/wiki/Forward_contract",
        "https://en.wikipedia.org/wiki/London_Metal_Exchange",
        "https://en.wikipedia.org/wiki/Chicago_Board_of_Trade",
        "https://en.wikipedia.org/wiki/NYMEX",
        "https://en.wikipedia.org/wiki/Commodity_market",
        "https://en.wikipedia.org/wiki/Contango",
        "https://en.wikipedia.org/wiki/Backwardation",
        "https://en.wikipedia.org/wiki/Platts",
        "https://en.wikipedia.org/wiki/Baltic_Dry_Index",
        "https://en.wikipedia.org/wiki/Baltic_Dirty_Tanker_Index",

        # Specific Vessel Types
        "https://en.wikipedia.org/wiki/Very_Large_Crude_Carrier",
        "https://en.wikipedia.org/wiki/Ultra_Large_Crude_Carrier",
        "https://en.wikipedia.org/wiki/Q-Max",
        "https://en.wikipedia.org/wiki/Capesize",
        "https://en.wikipedia.org/wiki/Kamsarmax",
        "https://en.wikipedia.org/wiki/Supramax",
        "https://en.wikipedia.org/wiki/Aframax",
        "https://en.wikipedia.org/wiki/Chinamax",
        "https://en.wikipedia.org/wiki/Post-Panamax",
        "https://en.wikipedia.org/wiki/Floating_storage_and_regasification_unit",
        "https://en.wikipedia.org/wiki/Floating_production_storage_and_offloading",

        # Maritime Regulations and Conventions
        "https://en.wikipedia.org/wiki/MARPOL",
        "https://en.wikipedia.org/wiki/SOLAS_Convention",
        "https://en.wikipedia.org/wiki/International_Ship_and_Port_Facility_Security_Code",
        "https://en.wikipedia.org/wiki/International_Safety_Management_Code",
        "https://en.wikipedia.org/wiki/United_Nations_Convention_on_the_Law_of_the_Sea",
        "https://en.wikipedia.org/wiki/IMO_2020",
        "https://en.wikipedia.org/wiki/International_Labour_Organization_Maritime_Labour_Convention",

        # Trade Routes
        "https://en.wikipedia.org/wiki/Northern_Sea_Route",
        "https://en.wikipedia.org/wiki/Northwest_Passage",
        "https://en.wikipedia.org/wiki/Strait_of_Gibraltar",
        "https://en.wikipedia.org/wiki/Bosporus",
        "https://en.wikipedia.org/wiki/Bab-el-Mandeb",
        "https://en.wikipedia.org/wiki/Kiel_Canal",
        "https://en.wikipedia.org/wiki/St._Lawrence_Seaway",
        "https://en.wikipedia.org/wiki/Maritime_Silk_Road",

        # Maritime Organizations
        "https://en.wikipedia.org/wiki/INTERTANKO",
        "https://en.wikipedia.org/wiki/INTERCARGO",
        "https://en.wikipedia.org/wiki/International_Chamber_of_Shipping",
        "https://en.wikipedia.org/wiki/World_Shipping_Council",
        "https://en.wikipedia.org/wiki/The_International_Association_of_Classification_Societies",
        "https://en.wikipedia.org/wiki/DNV_GL",
        "https://en.wikipedia.org/wiki/American_Bureau_of_Shipping",
        "https://en.wikipedia.org/wiki/Bureau_Veritas",

        # Commodities
        "https://en.wikipedia.org/wiki/Palm_oil",
        "https://en.wikipedia.org/wiki/Soybean",
        "https://en.wikipedia.org/wiki/Wheat",
        "https://en.wikipedia.org/wiki/Rice",
        "https://en.wikipedia.org/wiki/Sugar",
        "https://en.wikipedia.org/wiki/Cotton",
        "https://en.wikipedia.org/wiki/Copper",
        "https://en.wikipedia.org/wiki/Aluminium",
        "https://en.wikipedia.org/wiki/Fertilizer",
        "https://en.wikipedia.org/wiki/Cement",
        "https://en.wikipedia.org/wiki/Timber",

        # Related Categories
        "https://en.wikipedia.org/wiki/Category:Maritime_history",
        "https://en.wikipedia.org/wiki/Category:Ports_and_harbours",
        "https://en.wikipedia.org/wiki/Category:Shipping_routes",
        "https://en.wikipedia.org/wiki/Category:Maritime_law",
        "https://en.wikipedia.org/wiki/Category:Container_terminals",
        "https://en.wikipedia.org/wiki/Category:Energy_shipping",
        "https://en.wikipedia.org/wiki/Category:Tankers",
        "https://en.wikipedia.org/wiki/Category:Bulk_carriers",
        "https://en.wikipedia.org/wiki/Category:Shipping_canals",

        # Container Ships
        "https://de.wikipedia.org/wiki/Ultra_Large_Container_Ship",
        "https://en.wikipedia.org/wiki/Odense_Steel_Shipyard",
        "https://en.wikipedia.org/wiki/Samsung_Heavy_Industries",
        "https://en.wikipedia.org/wiki/Hanwha_Ocean",
        "https://en.wikipedia.org/wiki/List_of_largest_ships_by_gross_tonnage",
        "https://en.wikipedia.org/wiki/Pioneering_Spirit",
        "https://en.wikipedia.org/wiki/Prelude_FLNG",
        "https://en.wikipedia.org/wiki/Bellamya",
        "https://en.wikipedia.org/wiki/Batillus",
        "https://en.wikipedia.org/wiki/Pierre_Guillaumat_(supertanker)",
        "https://en.wikipedia.org/wiki/Prairial_(supertanker)",
        "https://en.wikipedia.org/wiki/Evergreen_A-class_container_ship",
        "https://en.wikipedia.org/wiki/List_of_longest_ships",
        "https://en.wikipedia.org/wiki/Seawise_Giant",
        "https://en.wikipedia.org/wiki/Batillus-class_supertanker",
        "https://en.wikipedia.org/wiki/List_of_Esso_Atlantic-class_supertankers",
        "https://en.wikipedia.org/wiki/Berge_Emperor",
        "https://en.wikipedia.org/wiki/TI-class_supertanker",
        "https://en.wikipedia.org/wiki/Valemax",
        "https://en.wikipedia.org/wiki/MS_Berge_Stahl",
        "https://en.wikipedia.org/wiki/Evergreen_A-class_container_ship",
        "https://en.wikipedia.org/wiki/MV_Barzan",
        "https://en.wikipedia.org/wiki/Icon-class_cruise_ship",
        "https://en.wikipedia.org/wiki/MV_Paul_R._Tregurtha",
        "https://en.wikipedia.org/wiki/Lagoon_450",
        "https://en.wikipedia.org/wiki/List_of_multihulls",
        "https://en.wikipedia.org/wiki/Trade_route",
        "https://en.wikipedia.org/wiki/MARPOL_73/78",
        "https://en.wikipedia.org/wiki/Hanseatic_League",
        "https://en.wikipedia.org/wiki/Bulk_carrier",
        "https://en.wikipedia.org/wiki/List_of_busiest_ports_in_Europe",
        "https://en.wikipedia.org/wiki/List_of_busiest_container_ports",
        "https://en.wikipedia.org/wiki/List_of_busiest_ports_by_cargo_tonnage"

        "https://en.wikipedia.org/wiki/Emission_Control_Area",
        "https://en.wikipedia.org/wiki/Green_shipping",
        "https://en.wikipedia.org/wiki/Ballast_water_management",
        "https://en.wikipedia.org/wiki/Energy_Efficiency_Design_Index",
        "https://en.wikipedia.org/wiki/Ship_Energy_Efficiency_Management_Plan",

        # Digital Transformation in Shipping
        "https://en.wikipedia.org/wiki/Maritime_autonomous_surface_ship",
        "https://en.wikipedia.org/wiki/E-navigation",
        "https://en.wikipedia.org/wiki/Electronic_Chart_Display_and_Information_System",
        "https://en.wikipedia.org/wiki/Blockchain_in_transport",
        "https://en.wikipedia.org/wiki/Port_Community_System",

        # Advanced Chartering Concepts
        "https://en.wikipedia.org/wiki/Contract_of_affreightment",
        "https://en.wikipedia.org/wiki/Consecutive_voyage_charter",
        "https://en.wikipedia.org/wiki/Dispatch_money",
        "https://en.wikipedia.org/wiki/Charterer%27s_liability_insurance",
        "https://en.wikipedia.org/wiki/Worldscale",

        # Specialized Commodity Trading
        "https://en.wikipedia.org/wiki/Carbon_emission_trading",
        "https://en.wikipedia.org/wiki/Weather_derivatives",
        "https://en.wikipedia.org/wiki/Freight_derivatives",
        "https://en.wikipedia.org/wiki/Over-the-counter_(finance)",
        "https://en.wikipedia.org/wiki/Commodity_Exchange_Act",

        # Ship Financing and Economics
        "https://en.wikipedia.org/wiki/Ship_mortgage",
        "https://en.wikipedia.org/wiki/Shipowner",
        "https://en.wikipedia.org/wiki/KG_financing",
        "https://en.wikipedia.org/wiki/Ship_Finance_International",
        "https://en.wikipedia.org/wiki/Sale_and_leaseback",

        # Specialized Vessel Operations
        "https://en.wikipedia.org/wiki/Bunkering",
        "https://en.wikipedia.org/wiki/Ship-to-ship_transfer",
        "https://en.wikipedia.org/wiki/Heavy_lift_ship",
        "https://en.wikipedia.org/wiki/Ship_recycling",
        "https://en.wikipedia.org/wiki/Offshore_vessel",

        # Modern Maritime Developments
        "https://en.wikipedia.org/wiki/Maritime_big_data",
        "https://en.wikipedia.org/wiki/Smart_port",
        "https://en.wikipedia.org/wiki/Maritime_cyber_security",
        "https://en.wikipedia.org/wiki/Internet_of_things_in_shipping",
        "https://en.wikipedia.org/wiki/Automated_container_terminal",

        # Historical Context with Modern Relevance
        "https://en.wikipedia.org/wiki/Spice_trade",
        "https://en.wikipedia.org/wiki/East_India_Company",
        "https://en.wikipedia.org/wiki/Dutch_East_India_Company",
        "https://en.wikipedia.org/wiki/Triangular_trade",
        "https://en.wikipedia.org/wiki/Age_of_Sail"

    ]

    scraper.crawl(seed_urls, max_pages=max_pages)
    corpus_path = scraper.create_training_corpus()

    return corpus_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run large-scale Wikipedia scraping for maritime content')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum number of pages to scrape')

    args = parser.parse_args()
    corpus_path = run_large_scale_scraper(max_pages=args.max_pages)