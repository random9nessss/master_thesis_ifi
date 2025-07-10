import random
from typing import Dict, Any, List
from config.logger import CustomLogger

class CharteringNewsAttributeSampler:
    """
    Samples random attributes for generating maritime chartering news articles.
    This includes advanced attributes such as categories, regions, impact levels,
    tone, etc., giving the LLM a detailed prompt for constructing varied content.
    """

    def __init__(self):
        self.logger = CustomLogger(name="CharteringNewsAttributeSampler")

        self.attribute_dict = {

            # -------------------------------------------------------------------
            # Categories
            # -------------------------------------------------------------------
            "categories": [
                "Market Outlook",
                "Regulatory Changes",
                "Port Congestion",
                "Fleet Developments",
                "Freight Rate Trends",
                "Shipyard News",
                "Maritime Environmental Policies",
                "New Chartering Contracts",
                "Environmental Impact",
                "Maritime Statistics",
                "Safety & Security Updates",
                "Technological Innovations",
                "Crew & Labor Issues",
                "Fuel Market Analysis",
                "Economic Policy Impact",
                "Global Trade Dynamics",
                "Port Infrastructure Investments",
                "Insurance & Claims",
                "Legal & Compliance",
                "Digital Transformation",
                "Sustainability Initiatives",
                "Shipping Logistics",
                "Market Volatility & Trends",
                "Industry Mergers & Acquisitions"
                "Risk Management Strategies"
            ],

            # -------------------------------------------------------------------
            # Regions
            # -------------------------------------------------------------------
            "regions": [
                "Asia-Pacific",
                "Northern Europe",
                "Mediterranean",
                "Middle East",
                "West Africa",
                "Latin America",
                "North America",
                "Global"
            ],

            # -------------------------------------------------------------------
            # Sentiment / Impact Assessement
            # -------------------------------------------------------------------
            "impact_assessment": [
                "Positive",
                "Negative",
                "Neutral",
                "Mixed",
                "Potentially Significant",
            ],

            # -------------------------------------------------------------------
            # Journalist Tone
            # -------------------------------------------------------------------
            "tones": [
                "Urgent",
                "Analytical",
                "Speculative",
                "Official",
                "In-Depth",
                "Brief",
                "Cautiously Optimistic"
            ],

            # -------------------------------------------------------------------
            # (Fictional) News Source
            # -------------------------------------------------------------------
            "sources": [
                "Maritime Daily Bulletin",
                "Global Shipping Times",
                "Chartering Insights Weekly",
                "Port News Network",
                "SeaTrade Review",
                "Oceanic Dispatch",
                "Nautical News Journal",
                "Harbor Herald",
                "Vessel View",
                "Maritime Market Monitor",
                "Shipping Spectrum",
                "Bluewater Bulletin",
                "SeaLog Report",
                "Fleet Focus",
                "Cargo Chronicle",
                "Portside Press",
                "Anchor Advocate",
                "Marine Industry Monitor",
                "Seafarers' Weekly",
                "Maritime Business Digest",
                "Ocean Trade Times",
                "Fleet & Freight News",
                "Shipping Ledger",
                "Blue Horizon Herald",
                "Chartering Chronicle",
                "Maritime Markets Observer",
                "Seaway Sentinel",
                "Global Maritime Journal",
                "Shipboard Insights",
                "Nautical Navigator",
                "Portside Perspective",
                "Cargo Compass",
                "Vessel Ventures",
                "Oceanic Outlook",
                "Maritime Minute",
                "Fleet Frontline",
                "Harbor Happenings",
                "Bluewave Bulletin",
                "SeaLink News",
                "Charter Chat",
                "Maritime Memos",
                "Dockside Daily",
                "Shipping Synopsis",
                "Oceanic Observer",
                "Fleet Forecast",
                "SeaFleet Signal",
                "Maritime Messenger",
                "Seaborne Summary",
                "Harbor Horizon",
                "Marine Market Monitor"
            ],

            # -------------------------------------------------------------------
            # (Fictional) Journalist / Authors
            # -------------------------------------------------------------------
            "journalists": [
                "A. Thompson",
                "C. Lopez",
                "M. Singh",
                "R. Tan",
                "V. Brunetti",
                "J. Roberts",
                "L. Carter",
                "D. O'Brien",
                "S. Kumar",
                "F. Chen",
                "B. Alvarez",
                "E. Johnson",
                "H. Patel",
                "K. Fischer",
                "P. Martins",
                "R. Morgan",
                "T. Nguyen",
                "Z. Li",
                "Q. Patel",
                "G. Stewart",
                "C. Evans",
                "J. Murphy",
                "M. Gallagher",
                "N. Banerjee",
                "S. Romero",
                "D. Delgado",
                "O. Kim",
                "I. Wang",
                "L. Moretti",
                "S. Becker",
                "F. Russo",
                "M. Laurent",
                "A. Costa",
                "P. Silva",
                "C. Davies",
                "R. Wilson",
                "J. Smith",
                "K. Jones",
                "L. Brown",
                "S. Clark",
                "M. Davis",
                "T. Martin",
                "B. Lewis",
                "C. Walker",
                "A. Wright",
                "J. Hall",
                "D. Allen",
                "R. Young",
                "V. King",
                "N. Schmidt"
            ],

            # -------------------------------------------------------------------
            # Article Count
            # -------------------------------------------------------------------
            "article_count_range": (2, 5)
        }

        self.logger.ok("CharteringNewsAttributeSampler initialized")

    # -------------------------------------------------------------------
    # Attribute Sampling
    # -------------------------------------------------------------------
    def sample_random_attributes(self) -> Dict[str, Any]:
        """
        Returns a dictionary of randomly sampled news attributes that can be used
        to guide an LLM in generating maritime chartering news articles.
        """

        articles_to_generate = random.randint(
            self.attribute_dict["article_count_range"][0],
            self.attribute_dict["article_count_range"][1]
        )

        attributes = {
            "category":         random.choice(self.attribute_dict["categories"]),
            "region":           random.choice(self.attribute_dict["regions"]),
            "impact":           random.choice(self.attribute_dict["impact_assessment"]),
            "tone":             random.choice(self.attribute_dict["tones"]),
            "source":           random.choice(self.attribute_dict["sources"]),
            "journalist":       random.choice(self.attribute_dict["journalists"]),
            "article_count":    articles_to_generate
        }

        self.logger.info("Generated random chartering news attributes")
        return attributes