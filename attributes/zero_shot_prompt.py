class ZeroShotPrompt:

    def __init__(self):

        self._prompt = """
        
            You are an expert in maritime shipping communications. Your task is to generate a synthetic email chain in JSON format that reflects a realistic negotiation between a ship broker and a charterer, covering topics such as freight rates, vessel details, laycan, and demurrage. The email chain should feel authentic and professional, but may include small talk or informal nuances.
            
            Requirements:
            1. The conversation must revolve around a single cargo shipment (e.g., crude oil, grains, etc.).
            2. Provide roughly 5 to 7 emails that illustrate a typical back-and-forth negotiation.
            3. Each email in the chain must contain:
               - "from": The sender’s name or identifier.
               - "to": The recipient’s name or identifier.
               - "subject": A concise subject line that may evolve (e.g., Re:, Fwd:, etc.).
               - "timestamp": A plausible date and time in chronological order (no specific timezone needed).
               - "body": The main text content of the email, which can include small disclaimers, shipping jargon, or mention of prior phone calls.
            
            4. The final output must be valid JSON with exactly two top-level keys:
               - "email_chain": an array of emails (each email is an object with the properties listed above).
               - "labels": an object containing the following keys; fill in as many values as make sense based on your narrative. Any unused or irrelevant label should be left as an empty string:
                 - "broker"
                 - "commodity"
                 - "load_port"
                 - "discharge_port"
                 - "cargo_size"
                 - "incoterm"
                 - "vessel"
                 - "dwt"
                 - "loa"
                 - "starting_freight_quote_currency"
                 - "starting_freight_quote"
                 - "final_freight_quote_currency"
                 - "final_freight_quote"
                 - "laytime_start_date"
                 - "laytime_end_date"
                 - "demurrage_currency"
                 - "demurrage"
            
            5. The emails should mention a starting freight rate and potentially evolve to a final negotiated rate. Use realistic values (e.g., “low 50s,” “$50.75,” etc.) if they come up.
            
            6. Avoid referencing any external tools, code, or attributes. Just perform the task as stated.
            
            Important Notes:
            - Maintain a logical and chronological flow.
            - Use plausible names for the broker and charterer (e.g., “Sarah (Broker),” “John (Charterer),” etc.).
            - Keep each email’s content focused on the negotiation and related maritime details without clutter or repetition.
            - Your answer must be *only* the valid JSON output with the structure described above. Do not include any extra commentary or formatting.
            
            Please generate your response now, **only** in valid JSON format, with keys "email_chain" and "labels".
        """