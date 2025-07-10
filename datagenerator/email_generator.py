import random
from datetime import datetime, timedelta
from quotegenerator.quote_generator import FreightQuoteEngine
from attributes.email_attribute_sampler import AttributeSampler
from config.logger import CustomLogger


class EmailGenerator:

    def __init__(self, sampler: AttributeSampler, freight_quote_engine: FreightQuoteEngine):
        self._sampler = sampler
        self._freight_quote_engine = freight_quote_engine

        self.logger = CustomLogger(name="EmailGenerator")
        self.logger.ok("EmailGenerator initialized")

    def construct_prompt(self, mode: str = "gossip", noise_probability: float = 0.13) -> str:
        """
        Construct a JSON-formatted email chain prompt using the attributes
        sampled from the AttributeSampler and freight quotes from the FreightQuoteEngine.
        This prompt instructs the LLM to generate a diverse email chain conversation in JSON format.

        Parameters:
            mode (str): Mode of sampling (e.g., "gossip" for shorter chains).
            noise_probability (float): Probability of noise injection into email, defaults to 0.133 -> 13%.

        Returns:
            str: A prompt string for the LLM.
        """

        # -------------------------------------------------------------------
        # Sampling Attributes
        # -------------------------------------------------------------------
        attributes = self._sampler.sample_random_attributes(mode)

        broker = attributes["broker"]
        commodity = attributes["commodity"]
        load_port = attributes["load_port"]
        discharge_port = attributes["discharge_port"]
        cargo_size = attributes["cargo_size"]
        incoterm = attributes["incoterm"]

        email_count = attributes["email_count"]

        vessel = attributes["vessel"]
        dwt = attributes["dwt"]
        loa = attributes["loa"]

        # Abbreviation Logic
        abbr_text = ", ".join(attributes["selected_abbreviations"]) if attributes["selected_abbreviations"] else "None"
        selected_abbr_line = (
            f'- **Selected Abbreviations**:           "{abbr_text}" (only use the abbreviations if they make sense in context).'
            if abbr_text is not None else ""
        )

        # Conversation Mode
        conversation_mode = attributes["conversation_mode"]

        # -------------------------------------------------------------------
        # Freight Quote Computation
        # -------------------------------------------------------------------
        freight_quote = self._freight_quote_engine.generate_rate(
            loadport=load_port,
            dischargeport=discharge_port,
            currency=None
        )

        # -------------------------------------------------------------------
        # Broker Attributes
        # -------------------------------------------------------------------
        uncertain_broker = random.choice([True, False])
        broker_demeanor = "Uncertain" if uncertain_broker else "Confident"
        level_of_english = attributes["tone"]
        formality_level = attributes["formality_level"]
        verbosity_level = attributes["verbosity_level"]

        # -------------------------------------------------------------------
        # Probabilistic Noise Generation
        # -------------------------------------------------------------------
        noise_injection = ""
        if random.random() <= noise_probability:
            noise_injection = f"- **Noise Injection**: {attributes['noise_type']['description']} - {attributes['noise_type']['details']}\n"

        # -------------------------------------------------------------------
        # Multi-Offer Instructions
        # -------------------------------------------------------------------
        conversation_mode = attributes["conversation_mode"]

        multi_offer_instructions = ""
        if isinstance(conversation_mode, dict) and conversation_mode.get("name") == "Multi-Offer":
            backup_vessel_info = attributes.get("backup_vessel")
            if backup_vessel_info:
                backup_vessel = backup_vessel_info["vessel"]
                backup_dwt = backup_vessel_info["dwt"]
                backup_loa = backup_vessel_info["loa"]

                multi_offer_instructions = f"""
            - Multi-Offer Specifics:
              - Include an alternative vessel option:
                  * Backup Vessel: "{backup_vessel}" (DWT: {backup_dwt}, LOA: {backup_loa}m)
              - Slightly adjust the starting freight quote for the backup option slightly or include different laycan/demurrage details for the backup vessel to differentiate the offers.
                """

        # -------------------------------------------------------------------
        # Random Anchor Date
        # -------------------------------------------------------------------
        anchor_str = attributes["anchor_dt"].strftime(attributes["date_format"])

        # -------------------------------------------------------------------
        # Name Generator
        # -------------------------------------------------------------------
        broker_name = attributes["broker_name"]
        charterer_name = attributes["charterer_name"]

        # -------------------------------------------------------------------
        # Prompt Generation
        # -------------------------------------------------------------------
        prompt = f"""
            Generate a synthetic email chain in JSON simulating a realistic maritime shipping communication between a broker and a charterer. The chain should reflect authentic back-and-forth exchanges and contain timestamps, short disclaimers (optional), and shipping jargon.
            The communication starts on {anchor_str}.
            
            Essential Attributes:
            - Broker:                           "{broker}"
            - Commodity:                        "{commodity}"
            - Load Port:                        "{load_port}"
            - Discharge Port:                   "{discharge_port}"
            - Cargo Size:                       "{cargo_size}MT"
            - Incoterm:                         "{incoterm}"
            - Vessel:                           "{vessel}" (DWT: {dwt} and LOA: {loa}m may be mentioned naturally in the conversation; however, including these details is optional)
            - Starting Freight Quote:           "{freight_quote}"
            - Broker's Demeanor:                "{broker_demeanor}"
            - Broker's English Level:           "{level_of_english}"
            - Broker's Formality:               "{formality_level}"
            - Verbosity:                        "{verbosity_level}"
            - Selected Abbreviations:           "{abbr_text}" 
              (Incorporate these abbreviations only if they naturally fit into the conversation.)
    
            {noise_injection}
    
            Conversation Guidelines:
            {conversation_mode}
            {multi_offer_instructions}
    
            2. Negotiation Flow:
               - Emails should feel natural and reflect typical shipping negotiations:
                 - The charterer might push back on the rate or ask for clarifications on vessel details.
                 - Mention or highlight important maritime points such as “laycan,” “stem,” “loading window, demurrage rate per day”) even if not part of the essential attributes.
    
            3. Subject Lines & Timestamps:
               - Include a concise but slighlty evolving subject line for each email (e.g., “Re:, "Fwd:", minor additions, etc.).
               - Add a timestamp in chronological order. Gaps between timestamps should be plausible. No mentioning of timezones.
    
            4. Stylistic Variations:
               - If formality is set to “informal” and English level is “foreign” or “intermediate,” introduce mild typos or grammar slips. 
               - If formality is “formal” and English level is “expert,” maintain a more professional and polished tone.
    
            5. JSON Output Format:
               - "email_chain":     an array where each element is an object with the keys "from", "to", "subject", "timestamp", and "body".
               - "labels":          an object containing the following pairs. For any label in "labels" that is not applicable or not mentioned in the conversation, leave its value as an empty string:
                   - `"broker"`
                   - `"commodity"`
                   - `"load_port"`
                   - `"discharge_port"`
                   - `"cargo_size"`
                   - `"incoterm"`
                   - `"vessel"`
                   - `"dwt"`
                   - `"loa"`
                   - '"starting_freight_quote_currency"'
                   - `"starting_freight_quote"`
                   - '"final_freight_quote_currency"'
                   - `"final_freight_quote"`
                   - `"laytime_start_date"`
                   - '"laytime_end_date"'
                   - '"demurrage_currency"'
                   - `"demurrage"`

            6. General Guidelines:
               - Aim for around {email_count} messages, but adjust as needed if more or fewer work better.
               - The final freight quote must always be an actual numerical value, even if the starting freight_quote is indicated as for example mid60s
               - The chain should revolve around one cargo/commodity scenario, not multiple. 
               - Freight quotes may be revised or clarified by the broker, but only after an initial quote has been established and only if the charterer counters with a revised rate inquiry.
               - If a noise injection is triggered, weave it naturally into one of the emails without derailing the conversation.
               - The broker is called {broker_name} and the charterer is called {charterer_name}.
               - Vary the currency formats as preferred (for example: USD, $, usd, us$, etc.).
               - Don't recite information such as vessel names in subsequent emails if they are clear from context, unless the vessel name is revised, just mention it once. 
               - Don't use any markdown formatting. Just plain text.
               
        """.strip()

        self.logger.info("Generated attr. prompt.")

        return prompt