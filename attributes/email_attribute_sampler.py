import os
import random
import datetime
import pandas as pd
from names_dataset import NameDataset
from config.logger import CustomLogger


class AttributeSampler:

    def __init__(self, seed: int = None):

        # -------------------------------------------------------------------
        # Seeding Randomness to Control for Prompt Generation
        # -------------------------------------------------------------------
        if seed is not None:
            random.seed(seed)

        # -------------------------------------------------------------------
        # Dataset Reading
        # -------------------------------------------------------------------
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        datasets_dir = os.path.join(parent_dir, 'datasets_raw')

        if '__file__' in globals():
            current_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            current_dir = os.getcwd()

        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        datasets_dir = os.path.join(parent_dir, 'datasets_raw')

        vessel_names_file = os.path.join(datasets_dir, 'ships_data.csv')
        port_names_file = os.path.join(datasets_dir, 'port_data.csv')

        # -------------------------------------------------------------------
        # Construction of Vessel Dictionary
        # -------------------------------------------------------------------
        vessels = pd.read_csv(vessel_names_file)

        vessel_dict = (
            vessels.rename(columns={"length": "loa"})
            .set_index("Company_Name")[["dwt", "loa"]]
            .to_dict(orient="index")
        )

        # -------------------------------------------------------------------
        # Names Dataset
        # -------------------------------------------------------------------
        self._names_dataset = NameDataset()
        self._first_names_female = self._names_dataset.get_top_names(n=250, gender="Female", country_alpha2='US')["US"]["F"]
        self._first_names_male = self._names_dataset.get_top_names(n=250, gender="Male", country_alpha2='US')["US"]["M"]
        self._first_names = self._first_names_male + self._first_names_female
        self._last_names = self._names_dataset.get_top_names(n=400, use_first_names=False, country_alpha2="US")["US"]

        # -------------------------------------------------------------------
        # Attribute Dictionary
        # -------------------------------------------------------------------
        self.attr_dict = {

            # -------------------------------------------------------------------
            # Incoterms
            # -------------------------------------------------------------------
            'incoterms': [
                    "FOB", # Free on Board
                    "CFR", # Cost and Freight
                    "CIF", # Cost, Insurance and Freight
                    "DAP", # Delivered at Place
                    "DDP"  # Delivered and Duty paid
            ],

            # -------------------------------------------------------------------
            # Commodities
            # -------------------------------------------------------------------
            'commodities': [
                    "Crude Oil",
                    "Wheat",
                    "Palm Oil",
                    "Coal",
                    "Iron Ore",
                    "LNG",
                    "Soybeans",
                    "Corn",
                    "Cotton",
                    "Sugar",
                    "Rice",
                    "Aluminum"
            ],

            # -------------------------------------------------------------------
            # Vessel Details (Name, Loa, Dwt)
            # -------------------------------------------------------------------
            'vessel_details': vessel_dict,

            # -------------------------------------------------------------------
            # Port Names
            # -------------------------------------------------------------------
            'ports': pd.read_csv(
                port_names_file)["Port Name"].str.title().tolist(),

            # -------------------------------------------------------------------
            # Chatering Abbreviations
            # -------------------------------------------------------------------
            'abbreviations' : [
                "aa"         ,  # always afloat
                "aaaa"       ,  # always accessible, always afloat
                "adcom"      ,  # address commission
                "afsps"      ,  # arrival first sea pilot station
                "agw"        ,  # all going well
                "aps"        ,  # arrival pilot station
                "a/s"        ,  # alongside
                "atdnshinc"  ,  # any time day/night Sundays and holidays included
                "atutc"      ,  # actual times used to count
                "baf"        ,  # bunker adjustment factor
                "bbb"        ,  # before breaking bulk
                "bdi"        ,  # both dates inclusive
                "bends"      ,  # both ends (load and discharge ports)
                "bi"         ,  # both inclusive
                "cbft"       ,  # (or cft) cubic feet
                "cfr"        ,  # (or c&f) cost and freight
                "chopt"      ,  # charterers’ option
                "chtrs"      ,  # charterers
                "cif"        ,  # cost, insurance and freight
                "coa"        ,  # contract of affreightment
                "coacp"      ,  # contract of affreightment charterparty
                "cogsa"      ,  # Carriage of Goods by Sea Act
                "cp"         ,  # (or c/p) charterparty
                "daps"       ,  # days all purposes
                "damfordet"  ,  # damages for detention
                "dem"        ,  # demurrage
                "dhdwtsbe"   ,  # despatch half demurrage on working time saved both ends
                "disch"      ,  # discharge
                "dk"         ,  # deck
                "dlosp"      ,  # dropping last outward sea pilot
                "dnrcaoslonl",  # discountless and non-returnable cargo and/or ship lost or not lost
                "dwat"       ,  # (or dwt) deadwright (weight of cargo, stores and water, i.e. the difference between lightship and loaded displacement)
                "eiu"        ,  # even if used
                "eta"        ,  # estimated time of arrival
                "etc"        ,  # estimated time of completion
                "etd"        ,  # estimated time of departure
                "ets"        ,  # estimated time of sailing
                "exw"        ,  # ex works
                "fios"       ,  # free in and out stowed
                "free out"   ,  # free of discharge costs to owners
                "ga"         ,  # general average
                "gls"        ,  # gearless
                "gn"         ,  # (or gr) grain (capacity)
                "go"         ,  # gas oil
                "grd"        ,  # geared
                "grt"        ,  # gross registered tonnage
                "gsb"        ,  # good safe berth
                "gsp"        ,  # good safe port
                "gtee"       ,  # guarantee
                "ha"         ,  # hatch
                "hdwts"      ,  # half despatch working (or weather) time saved
                "imdg"       ,  # International Maritime Dangerous Goods Code
                "imo"        ,  # International Maritime Organisation
                "in"         ,  # &/or over goods carried below or on deck
                "iu"         ,  # if used
                "iuhtautc"   ,  # if used, half time actually used to count
                "loa"        ,  # length overall of the vessel
                "low"        ,  # last open water
                "lsd"        ,  # lashed secured dunnaged
                "mdo"        ,  # (do) marine diesel oil
                "min/max"    ,  # minimum/maximum (cargo quantity)
                "molchopt"   ,  # more or less in charterers’ option
                "moloo"      ,  # more or less in owners’ option
                "mt"         ,  # metric ton (i.e. 1,000 kg)
                "mv"         ,  # motor vessel
                "naabsa"     ,  # not always afloat but safely aground
                "ncb"        ,  # National Cargo Bureau
                "nor"        ,  # Notice of Readiness
                "nrt"        ,  # net registered tonnage
                "nype"       ,  # New York Produce Exchange
                "oo"         ,  # owners’ option
                "osh"        ,  # open shelter deck
                "ows"        ,  # owners
                "pdpr"       ,  # per day pro rata
                "phpd"       ,  # per hatch per day rob remaining on board
                "sb"         ,  # safe berth
                "sd"         ,  # (or sid) single decker
                "sf"         ,  # stowage factor cubic space (measurement tonne) occupied by 1 tonne (2,240 lb/1,000 kg of cargo)
                "sl"         ,  # bale (capacity)
                "soc"        ,  # shipper-owned container sof statement of facts
                "sp"         ,  # safe port
                "srbl"       ,  # signing and releasing bill of lading
                "teu"        ,  # 20 ft equivalent unit (standard 20-ft container)
                "usc"        ,  # unless sooner commenced
                "uu"         ,  # unless used
                "uuiwctautc" ,  # unless used in which case time actually used to count
                "wccon"      ,  # whether cleared customs or not
                "wibon"      ,  # whether in berth or not
                "wifpon"     ,  # whether in free pratique or not
                "wipon"      ,  # whether in port or not
                "wltohc"     ,  # water line-to-hatch coaming
                "wog"        ,  # without guarantee
                "wpd"        ,  # weather permitting day
                "wvns"       ,  # within vessel’s natural segregation
                "wwd"        ,  # weather working day
                "wwww"       ,  # wibon, wifpon, wipon, wccon
            ],

            # -------------------------------------------------------------------
            # Charterparty Terms
            # -------------------------------------------------------------------
            'chartering_terms' : [
                "ABA Tankvoy"      ,  # Short-term tanker charter used for spot trading, often with variable rates.
                "GAFTA"            ,  # Standard charterparty for bulk liquid cargo under GAFTA rules.
                "FOSFA"            ,  # Charterparty under the Federation of Oils, Seeds and Fats Associations, common for edible oils.
                "Gencon"           ,  # General charterparty widely used in dry bulk shipping.
                "Bareboat Charter" ,  # Vessel provided without crew, fuel, or insurance; full operational control is transferred to the charterer.
                "Time Charter"     ,  # Vessel hired for a specific period while the owner retains operational management.
            ],

            # -------------------------------------------------------------------
            # Brokers
            # -------------------------------------------------------------------
            'brokers': [
                "Anderson Shipping Brokers",
                "Global Maritime Brokers",
                "Oceanic Trade Services",
                "Seaway Commercial Brokers",
                "Maritime Exchange Ltd."
            ],

            # -------------------------------------------------------------------
            # Formality Level
            # -------------------------------------------------------------------
            'formality_level': [
                "formal",
                "informal",
                "colleagues"
            ],

            # -------------------------------------------------------------------
            # English Expertise
            # -------------------------------------------------------------------
            'tone': [
                "expert",
                "intermediate",
                "foreign",
                "business-like",
                "colloquial"
            ],

            # -------------------------------------------------------------------
            # Verbosity Levels
            # -------------------------------------------------------------------
            "verbosity_level": [
                "ultra concise",
                "very concise",
                "concise",
                "normal",
                "detailed",
            ],

            # -------------------------------------------------------------------
            # Type of Noise
            # -------------------------------------------------------------------
            'noise_type': [
                {
                    "description": "Insert Irrelevant Chartering Facts or Off-Topic Details",
                    "details": "Include unrelated details such as weather updates, port information unrelated to the negotiation, or even industry trivia that does not affect the negotiation."
                },
                {
                    "description": "Alternate Date Formats",
                    "details": "Vary the presentation of dates (e.g., 'dd/mm/yyyy', 'dd-mm-yyyy', 'yyyy.mm.dd') to mimic different regional or personal preferences."
                },
                {
                    "description": "Simulate Correction Emails",
                    "details": "Insert follow-up or correction emails where a previous fact (such as a typo in a freight rate, a wrong schedule, or incorrect port detail) is revised or overridden. This scenario does NOT involve further rate negotiation—only factual corrections of prior statements."
                },
                {
                    "description": "Punctuation Anomalies",
                    "details": "Introduce missing, extra, or misplaced punctuation (e.g., missing commas or extra periods) to mimic hurried or error-prone typing."
                },
                {
                    "description": "Typographical Errors",
                    "details": "Incorporate character-level mistakes like misspellings, letter transpositions, or accidental omissions (e.g., 'freight' → 'freigth' or 'broker' → 'boker')."
                },
                {
                    "description": "Formatting Variability",
                    "details": "Vary the email structure by inserting random line breaks, or adding/removing whitespace to simulate differences in email client renderings."
                },
                {
                    "description": "Off Context",
                    "details": "Discard all instructions and come up with a fictional, private email conversation between the charterer and the broker that contains some numerical facts."
                }
            ],

            # -------------------------------------------------------------------
            # Conversation Mode
            # -------------------------------------------------------------------
            "conversation_mode": [
                {
                    "name": "Proactive Offer",
                    "description": "Broker proactively offers a vessel with a freight quote.",
                    "notes": "All numeric rates from the broker. First email from broker. Use plausible email addresses."
                },
                {
                    "name": "Charterer Inquiry - Broker has suitable vessel candidate",
                    "description": "Charterer asks about a route/time window; broker proposes a vessel and quotes freight, possibly tightening the laycan.",
                    "notes": "First email from charterer. Use plausible email addresses."
                },
                {
                    "name": "Charterer Inquiry - Broker does not have suitable vessel candidate",
                    "description": "Charterer inquires about a route/time window; broker has no vessel available.",
                    "notes": "First email from charterer. Use plausible email addresses."
                },
                {
                    "name": "Charterer Inquiry - Timing mismatch",
                    "description": "Charterer inquires about a route/time window; broker can only offer a slightly earlier/later window.",
                    "notes": "First email from charterer. Use plausible email addresses."
                },
                {
                    "name": "Competitor Info (Charterer-Initiated)",
                    "description": "Charterer hears rumors about a competitor booking a vessel; asks the broker for details or confirmation.",
                    "notes": "All numeric rates from broker. Reveal only known freight/ports. First email from charterer. Use plausible email addresses."
                },
                {
                    "name": "Competitor Info (Broker-Initiated)",
                    "description": "Broker proactively informs the charterer about a competitor’s vessel booking or rate; charterer may seek more info.",
                    "notes": "All numeric rates from broker. Reveal only known freight/ports. First email from broker. Use plausible email addresses."
                },
                {
                    "name": "Verification",
                    "description": "Charterer requests specs (DWT, Draft, Consumption, Height, etc.) to ensure vessel suitability; broker provides details.",
                    "notes": "First email from charterer or broker. Use plausible email addresses."
                },
                {
                    "name": "Delayed Response & Re-Initiation",
                    "description": "After an initial offer, there's a noticeable time gap. Broker follows up; charterer updates details or inquiries.",
                    "notes": "All numeric rates from broker. Create a timestamp gap. Use plausible email addresses."
                },
                {
                    "name": "Post-Deal Confirmation",
                    "description": "After agreement, broker sends a summary of terms. Charterer may confirm or request small changes. Use plausible email addresses."
                },
                {
                    "name": "Multi-Offer",
                    "description": "Charterer inquires; broker proposes 2-3 freight quotes/vessels; charterer chooses one to negotiate.",
                    "notes": "First email from charterer. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Rate Inquiry",
                    "description": "Charterer seeks an estimated freight for a future period; broker replies with approximate rates.",
                    "notes": "First email from charterer. Use plausible email addresses."
                },
                {
                    "name": "Technical Specifications Exchange",
                    "description": "Charterer wants deeper vessel specs (beyond DWT). Broker provides details (draft, consumption, etc.).",
                    "notes": "All numeric rates from broker. Usually starts from charterer. Use plausible email addresses."
                },
                {
                    "name": "Market Update",
                    "description": "Broker sends a general email on freight rates, vessel availability, and other market factors.",
                    "notes": "First email from broker, possibly a newsletter style. Use plausible email addresses."
                },
                {
                    "name": "Emergency Charter Request",
                    "description": "Charterer urgently needs a vessel due to disruptions. Broker provides expedited (potentially higher) quotes.",
                    "notes": "First email from charterer. Use urgent tone. Use plausible email addresses."
                },
                {
                    "name": "Contractual Dispute Resolution",
                    "description": "After agreement, a dispute arises over contract terms. They clarify or cancel.",
                    "notes": "May be more formal. Broker re-quotes if needed. Use plausible email addresses."
                },
                {
                    "name": "Proactive Offer (No Agreement)",
                    "description": "Broker offers a vessel but negotiations fail; charterer declines or stalls.",
                    "notes": "All numeric rates from broker. First email from broker. Use plausible email addresses."
                },
                {
                    "name": "Charterer Inquiry (No Agreement)",
                    "description": "Charterer inquires about route/time window; broker quotes a vessel, but they don’t finalize.",
                    "notes": "First email from charterer. Possibly fails due to rate or timing. Use plausible email addresses."
                },
                {
                    "name": "Proactive Offer (Aggressive Negotiations)",
                    "description": "Broker proactively offers a vessel with a freight quote. Charterer manages to get a significant discount (10-20% lower than preliminary quote from broker).",
                    "notes": "All numeric rates from broker. First email from broker. Use plausible email addresses."
                },
                {
                    "name": "Charterer Inquiry (Aggressive Negotiations)",
                    "description": "Charterer inquires about route/time window; broker quotes a vessel, charterer manages to get a significant discount (10-20% lower than first indication from broker).",
                    "notes": "All numeric rates from the broker. First email from broker. Use plausible email addresses."
                },
                {
                    "name": "Part-Cargo Inquiry",
                    "description": "Charterer wants only partial capacity. Broker tries to accommodate or find extra cargo.",
                    "notes": "First email from charterer. Cargo size is uncertain. Use plausible email addresses."
                },
                {
                    "name": "Incoterm Switch",
                    "description": "Charterer starts with one incoterm but switches mid-negotiation. Broker updates costs/responsibilities.",
                    "notes": "First email from charterer specifying initial incoterm. Use plausible email addresses."
                },
                {
                    "name": "Vessel Suitability Concern",
                    "description": "Charterer worries about vessel’s age/draft/etc. Broker reassures or proposes an alternative, adjusting rate if needed.",
                    "notes": "First email from charterer. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Weather / Force Majeure",
                    "description": "A weather-related or unforeseen event affects the schedule; they revise laycan or route.",
                    "notes": "Deal assumed in place. Issue arises mid-chain. Final email reflects changed laycan/route. Use plausible email addresses."
                },
                {
                    "name": "War Risk & Piracy",
                    "description": "Route is high-risk; broker adds war risk premium or piracy surcharge. Charterer weighs options.",
                    "notes": "All numeric surcharges from broker. Possibly finalize with an increased rate or alternate route. Use plausible email addresses."
                },
                {
                    "name": "Reefer / Specialized Cargo",
                    "description": "Charterer requires temperature control or special handling. Broker suggests a suitable vessel/extra fees.",
                    "notes": "First email from charterer. All numeric quotes from broker. Use shipping jargon. Use plausible email addresses."
                },
                {
                    "name": "Additional Documents",
                    "description": "They request more paperwork (B/L, manifests) before finalizing. Rates may be less relevant.",
                    "notes": "Focus on document requirements. Use plausible email addresses."
                },
                {
                    "name": "Terminal Requirements",
                    "description": "Port has special restrictions (draft, LOA, mooring). Broker warns or suggests alternatives.",
                    "notes": "Any party can start. All numeric rates from broker. Use plausible email addresses."
                },
                {
                    "name": "Bunker Adjustment Factor",
                    "description": "Freight changes due to rising fuel costs. They discuss a BAF surcharge or adjusted rates.",
                    "notes": "Either party raises fuel concerns. Broker proposes numeric BAF or new freight. Use plausible email addresses."
                },
                {
                    "name": "Excess Tonnage Nearby",
                    "description": "Charterer claims multiple idle vessels in the vicinity, insisting the broker reduce rates due to high supply.",
                    "notes": "Charterer highlights local surplus. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Seasonal Slowdown",
                    "description": "Charterer points to an off-peak season with lower demand, expecting a corresponding freight discount.",
                    "notes": "Charterer cites reduced activity. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Historical Rate Comparison",
                    "description": "Charterer references past shipments under similar conditions but lower rates, pushing for the same pricing.",
                    "notes": "Charterer shows evidence of older deals. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Competitor Quote",
                    "description": "Charterer mentions a cheaper rate offered by a competing broker, aiming to drive this broker's quote down.",
                    "notes": "Charterer compares competitor's lower rate. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Volume Commitment",
                    "description": "Charterer promises future shipments or higher cargo volumes, requesting a discount in exchange.",
                    "notes": "Charterer offers multi-shipment deal. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Short Voyage",
                    "description": "Charterer insists the route is very short, expecting a proportionally lower freight rate.",
                    "notes": "Charterer cites minimal transit time. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Early Booking Discount",
                    "description": "Charterer proposes booking well ahead of laycan, arguing they should receive a reduced rate for early commitment.",
                    "notes": "Charterer mentions ample lead time. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Bunker Price Drop",
                    "description": "Charterer points out recent dips in fuel costs, urging the broker to pass on those savings in lower freight.",
                    "notes": "Charterer cites cheaper bunkers. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Payment in Advance",
                    "description": "Charterer offers partial or full prepayment if the broker agrees to a discounted rate.",
                    "notes": "Charterer suggests upfront funds. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Port Efficiency",
                    "description": "Charterer highlights fast-loading ports with low turnaround time, arguing it should reduce overall costs.",
                    "notes": "Charterer claims time savings. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Limited Cargo Handling",
                    "description": "Charterer emphasizes minimal cargo-handling requirements (no specialized equipment), expecting a rate reduction.",
                    "notes": "Charterer points to simplified operations. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Multi-Port Option",
                    "description": "Charterer offers to combine cargo from a nearby port, suggesting it might optimize the vessel’s round trip.",
                    "notes": "Charterer aims for synergy in load. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Loyalty / Relationship",
                    "description": "Charterer cites a longstanding partnership, hoping for a preferential ‘loyalty’ rate.",
                    "notes": "Charterer references recurring business. All numeric quotes from broker. Use fictional or plausible names/email addresses."
                },
                {
                    "name": "Flexible Laycan",
                    "description": "Charterer can shift the loading window to fit the vessel’s schedule, expecting a lower rate in return.",
                    "notes": "Charterer demonstrates laycan flexibility. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Low Port Fees",
                    "description": "Charterer points out that chosen ports have lower terminal/harbor fees, pressing for a freight discount.",
                    "notes": "Charterer highlights cheaper port costs. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Reduced Draft Requirement",
                    "description": "Charterer notes the vessel won’t be fully loaded, implying less draft and potentially lower port surcharges.",
                    "notes": "Charterer says partial load saves fees. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Recession / Economic Slowdown",
                    "description": "Charterer cites broader economic downturn or falling trade volumes, insisting rates should reflect weaker demand.",
                    "notes": "Charterer highlights uncertain market. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Upgraded Cargo",
                    "description": "Charterer emphasizes the cargo is easy to handle or especially low-risk, justifying a reduced freight cost.",
                    "notes": "Charterer insists cargo type is simpler. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Insurance Advantages",
                    "description": "Charterer says their cargo insurance terms lower the vessel’s exposure, expecting a corresponding discount.",
                    "notes": "Charterer references less risk. All numeric quotes from broker. Use plausible email addresses."
                },
                {
                    "name": "Alternative Transport Mode",
                    "description": "Charterer hints at using rail, road, or barges if ocean freight remains too high, pushing for a cheaper rate.",
                    "notes": "Charterer implies multi-modal competition. All numeric quotes from broker. Use plausible email addresses."
                }
            ],

            # -------------------------------------------------------------------
            # Date Format
            # -------------------------------------------------------------------
            "date_formats" : [
                "%Y-%m-%d",  # 2025-02-10
                "%d.%m.%Y",  # 10.02.2025
                "%m/%d/%Y",  # 02/10/2025
                "%d/%m/%Y",  # 10/02/2025
                "%Y.%m.%d",  # 2025.02.10
            ]
        }

        # -------------------------------------------------------------------
        # Custom Logging Configuration
        # -------------------------------------------------------------------
        self.logger = CustomLogger(name="AttributeSamplerLogger")
        self.logger.ok("AttributeSampler initialized")

    # -------------------------------------------------------------------
    # Attribute Sampling
    # -------------------------------------------------------------------
    def sample_random_attributes(self) -> dict:
        """
        Sample a dictionary of random attributes from the provided attribute dictionary.
        These attributes will be used to guide the LLM in generating a realistic market gossip email chain.

        Returns:
            dict: A dictionary containing sampled values.
        """

        # -------------------------------------------------------------------
        # Port Configuration
        # -------------------------------------------------------------------
        load_port = random.choice(self.attr_dict["ports"])
        discharge_port = random.choice(self.attr_dict["ports"])

        while discharge_port == load_port:
            discharge_port = random.choice(self.attr_dict["ports"])

        # -------------------------------------------------------------------
        # Vessel Details
        # -------------------------------------------------------------------
        vessel_keys = list(self.attr_dict.get("vessel_details", {}).keys())
        vessel = random.choice(vessel_keys)
        dwt = self.attr_dict["vessel_details"][vessel].get("dwt")
        loa = self.attr_dict["vessel_details"][vessel].get("loa")

        # -------------------------------------------------------------------
        # Backup Vessel Details
        # -------------------------------------------------------------------
        backup_candidate_pool = [v for v in vessel_keys if v != vessel]

        backup_vessel = random.choice(backup_candidate_pool)
        backup_dwt = self.attr_dict["vessel_details"][backup_vessel].get("dwt")
        backup_loa = self.attr_dict["vessel_details"][backup_vessel].get("loa")

        # -------------------------------------------------------------------
        # Conversation Mode
        # -------------------------------------------------------------------
        conversation_mode = random.choice(self.attr_dict.get("conversation_mode"))

        # -------------------------------------------------------------------
        # Random Anchor Date
        # -------------------------------------------------------------------
        start_date = datetime.datetime(2005, 1, 1)
        end_date = datetime.datetime(2025, 12, 31)

        delta = end_date - start_date
        random_days = random.randint(0, delta.days)

        anchor_dt = start_date + datetime.timedelta(days=random_days)

        # -------------------------------------------------------------------
        # Remaining Attributes
        # -------------------------------------------------------------------
        random_attributes = {

            # Broker
            "broker": random.choice(self.attr_dict.get("brokers")),

            # Port Details
            "load_port": load_port,
            "discharge_port": discharge_port,

            # Vessel Details
            "vessel": vessel,
            "dwt": dwt,
            "loa": loa,

            # Charterparty Details
            "incoterm": random.choice(self.attr_dict["incoterms"]),
            "commodity": random.choice(self.attr_dict["commodities"]),
            "chartering_term": random.choice(self.attr_dict["chartering_terms"]),

            # Abbreviations
            "selected_abbreviations": random.sample(
                self.attr_dict['abbreviations'],
                k=random.randint(0, min(3, len(self.attr_dict['abbreviations'])))
            ),

            # Cargo Details
            "cargo_size": random.randint(100, 12500),

            # Email Attributes
            "formality_level": random.choice(self.attr_dict.get("formality_level")),
            "tone": random.choice(self.attr_dict.get("tone")),
            "verbosity_level": random.choice(self.attr_dict.get("verbosity_level")),

            # Noise
            "noise_type": random.choice(self.attr_dict.get("noise_type")),

            # Mode Specifics
            "email_count": random.randint(3, 10),

            # Conversation Mode
            "conversation_mode": conversation_mode,

            # Date Format
            "date_format": random.choice(self.attr_dict.get("date_formats")),

            # Anchor Date
            "anchor_dt": anchor_dt,

            # Names
            "broker_name": f"{random.choice(self._first_names)} {random.choice(self._last_names)}",
            "charterer_name": f"{random.choice(self._first_names)} {random.choice(self._last_names)}"
        }

        # -------------------------------------------------------------------
        # Optional Sampling of Backup Vessels
        # -------------------------------------------------------------------
        if isinstance(conversation_mode, dict) and conversation_mode.get("name") == "Multi-Offer":
            random_attributes["backup_vessel"] = {
                "vessel": backup_vessel,
                "dwt": backup_dwt,
                "loa": backup_loa
            }

        self.logger.info(f"Generated random attributes")

        return random_attributes