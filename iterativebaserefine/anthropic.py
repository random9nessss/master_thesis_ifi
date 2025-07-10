import pandas as pd
import random
import os
import httpx
import json
import requests
import re
import time
import anthropic

from config.logger import CustomLogger
from attributes.email_attribute_sampler import AttributeSampler

from dotenv import load_dotenv
load_dotenv("ENV.txt")

class IterativeSeededBareAnthropic:
    """
    A class that generates and refines draft emails using news headlines.

    This class loads news headlines from multiple JSON files, generates an initial email draft based on a random
    headline, and then refines that draft. It then iteratively feeds the complete email chain back into the base model
    to generate an answer, refines that answer, and appends it to the chain. The iterative process is repeated a
    random number of times between 2 and 5, so that the full email chain is returned.
    """

    def __init__(self):
        """
        Initialize the IterativeSeededBare instance.

        Sets up the base URL for the starting email generation, the Anthropic API client,
        the attribute sampler for placeholder data, and loads news headlines from specified JSON files.

        Raises:
            EnvironmentError: If the required ANTHROPIC_KEY environment variable is not set.
        """
        self.base_model_url = "http://localhost:8000/generate"

        anthropic_key = os.getenv("ANTHROPIC_KEY")
        if not anthropic_key:
            raise EnvironmentError("ANTHROPIC_KEY environment variable not set.")

        self.client = anthropic.Anthropic(
            api_key=anthropic_key,
            http_client=httpx.Client(verify=False)
        )

        # ------------------------------------------------------------------
        # Directory Setup
        # ------------------------------------------------------------------
        try:
            repo_root = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            repo_root = os.getcwd()

        self._directory = os.path.join(repo_root, "syntheticdata", "iterativebaserefine", "claude")
        self._output_directory = os.path.join(self._directory, "output")

        os.makedirs(self._directory, exist_ok=True)
        os.makedirs(self._output_directory, exist_ok=True)

        self._news = self._load_news_headlines()
        self._attribute_sampler = AttributeSampler()
        self.logger = CustomLogger(name="IterativeBaseRefine")

    def _yield_start_prompt(self, headline: str) -> str:
        """
        Generate the starting prompt for the draft email generation.

        Args:
            headline (str): The news headline to use as context.

        Returns:
            str: The complete prompt to be sent to the starting email generation model.
        """
        return f"""
            You are in the chartering business as a broker, providing your customer with attractive freight rates and intel.
            Draft a negotiation email for your business counterparty. The subject is {headline}.

            [DRAFT EMAIL]
        """

    def _yield_refinement_prompt(self, headline: str, base_text: str) -> str:
        """
        Generate the refinement prompt for improving the draft email.

        Args:
            headline (str): The news headline that contextualizes the email.
            base_text (str): The initial draft email text.

        Returns:
            str: The complete prompt to be sent to the refinement model.
        """
        return f"""
            Please refine the following email chain to make it coherent and realistic without changing the theme,
            then just return the plain, refined email body after the tag [REFINED EMAIL] without any subject, timestamp, etc.

            The email is related to {headline} and is as follows:
            {base_text}
        """

    def _yield_iterative_base_prompt(self, headline: str, previous_message: str) -> str:
        """
        Generate the iterative base prompt for generating an answer email.

        Args:
            headline (str): The news headline to use as context.
            previous_message (str): The complete email chain from previous interactions.

        Returns:
            str: The complete iterative base prompt to be sent to the base model endpoint.
        """
        return f"""
            You are in the chartering business as a charterer, requiring vessels to transport goods.
            The subject of your current discussion is {headline}. Your broker sent you the following email chain:

            {previous_message}

            [ANSWER EMAIL]
        """

    def _yield_iterative_refinement_prompt(self, headline: str, previous_message: str, base_text: str) -> str:
        """
        Generate the iterative refinement prompt for refining the answer email.

        Args:
            headline (str): The news headline that contextualizes the email.
            previous_message (str): The existing email chain.
            base_text (str): The iterative answer email that needs refining.

        Returns:
            str: The complete prompt to be sent to the refinement model.
        """
        return f"""
            Please review the following answer email, which is part of an ongoing email chain, and make only minimal corrections. Correct grammatical, punctuation, or clarity issues without repeating or expanding on content that is already clearly referenced in the previous message.
            Keep the original tone, style, and theme unchanged. Then just return the plain, refined email body after the tag [REFINED EMAIL] without any subject, timestamp, etc.

            The email is related to {headline} and is the answer to the following email chain:
            {previous_message}

            The answer email to be refined is as follows:
            {base_text}

            [REFINED EMAIL]
        """

    def _yield_final_refinement_prompt(self, headline: str, final_email_chain: str) -> str:
        """
        Generate the final refinement prompt to format the complete email chain into a structured JSON output.

        This prompt instructs the model to reformat the email chain according to a strict JSON schema,
        incorporating inferred attribute labels where possible. It uses random attributes sampled from the
        AttributeSampler to fill in placeholders when applicable.

        Args:
            headline (str): The news headline that contextualizes the email chain.
            final_email_chain (str): The unrefined complete email chain.

        Returns:
            str: The prompt to be sent to the refinement model for final formatting.
        """
        random_attributes = self._attribute_sampler.sample_random_attributes(mode="standard")

        return f"""
            Please reformat the following email chain into the JSON structure outlined below. Preserve the original content exactly, only modifying placeholders (such as "[...]") or non-specific terms like the vessel, the cargo, etc.).

            Keep in mind that the conversation centers on {headline}. Use the following attributes only when they are appropriate; otherwise, ignore them:

            - broker_firm:          {random_attributes.get("broker", "")}
            - vessel_name:          {random_attributes.get("vessel_name", "")}
            - deadweight (dwt):     {random_attributes.get("dwt", "")}
            - length overall (loa): {random_attributes.get("loa", "")}
            - incoterm:             {random_attributes.get("incoterm", "")}
            - commodity:            {random_attributes.get("commodity", "")}
            - cargo_size:           {random_attributes.get("cargo_size", "")}

            ---
            Unrefined email chain:
            {final_email_chain}
            ---

            Formatting requirements:
            1. Convert the email chain into the JSON structure provided exactly.
            2. Append a "labels" object at the end, including as many inferred fields as possible. If uncertain about a field's value, leave it empty but still include the key.
            3. Ensure the emails are in chronological order. If necessary, add or estimate a timestamp in the format "YYYY-MM-DD HH:MM" for each email, using the anchor date/time {random_attributes.get('anchor_dt', "")} as a reference.
            4. Generate realistic email addresses based on the names in the base text. If names are missing or incomplete, assume one party is {random_attributes.get('broker_name', "")} and the other is {random_attributes.get('charterer_name', "")}.

            The expected JSON structure is:
            ```json
            {{
              "email_chain": [
                {{
                  "from": "...",
                  "to": "...",
                  "subject": "...",
                  "timestamp": "YYYY-MM-DD HH:MM",
                  "body": "..."
                }},
                ...
              ],
              "labels": {{
                "broker": "",
                "commodity": "",
                "load_port": "",
                "discharge_port": "",
                "cargo_size": "",
                "incoterm": "",
                "vessel": "",
                "dwt": "",
                "loa": "",
                "starting_freight_quote_currency": "",
                "starting_freight_quote": "",
                "final_freight_quote_currency": "",
                "final_freight_quote": "",
                "laytime_start_date": "",
                "laytime_end_date": "",
                "demurrage_currency": "",
                "demurrage": ""
              }}
            }}
            ```
        """

    def _final_refinement(self, headline: str, final_email: str) -> str:
        """
        Apply the final refinement to adjust the format of the complete email chain and extract labels.

        This method sends the final refinement prompt to the Anthropic API endpoint.
        The model is instructed to reformat the complete email chain into a structured JSON format that
        includes both the chronological email chain and an object of inferred labels.

        Args:
            headline (str): The news headline that contextualizes the conversation.
            final_email (str): The unrefined complete email chain.

        Returns:
            str: The final refined email chain in the expected JSON format.

        Raises:
            Exception: If the final refinement API call fails or the response cannot be parsed.
        """
        final_refinement_prompt = self._yield_final_refinement_prompt(headline, final_email)

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=1.0,
                messages=[{"role": "user", "content": final_refinement_prompt}]
            )
        except Exception as e:
            raise Exception(f"Final refinement API call failed: {e}")

        try:
            final_refined_text = response.content[0].text
            return final_refined_text
        except Exception as e:
            raise Exception(f"Failed to parse the final refined email: {e}")

    def _load_news_headlines(self) -> pd.DataFrame:
        """
        Load news headlines from multiple JSON files into a single DataFrame.

        Reads news headlines from four different JSON files. If a file cannot be read or does not contain a
        'headline' column, it will be skipped with an error logged.

        Returns:
            pd.DataFrame: A DataFrame with a single column 'headline' containing all loaded headlines.
        """
        file_paths = {
            "gcaptain": "../data/newsarticles/gcaptain/gcaptain_articles.json",
            "marinetraffic": "../data/newsarticles/marinetraffic/marinetraffic_articles.json",
            "maritimeexecutive": "../data/newsarticles/maritimeexecutive/maritimeexecutive_articles.json",
            "splash": "../data/newsarticles/splash247/splash247_articles.json"
        }

        all_headlines = []

        for source, path in file_paths.items():
            try:
                df = pd.read_json(path)
                if 'headline' in df.columns:
                    headlines = df['headline'].dropna().tolist()
                    all_headlines.extend(headlines)
                else:
                    self.logger.warning(f"'headline' column not found in {source} file at {path}.")
            except Exception as e:
                self.logger.error(f"Failed to load headlines from {source} at {path}: {e}")

        if not all_headlines:
            raise ValueError("No headlines loaded from any source.")

        return pd.DataFrame({'headline': all_headlines})

    def _generate_start_email(self, headline: str) -> str:
        """
        Generate an initial draft email using the provided headline.

        This method sends a prompt to the base model endpoint and retries up to 5 times if the response is invalid.

        Args:
            headline (str): The news headline to include in the prompt.

        Returns:
            str: The generated email draft.

        Raises:
            Exception: If a valid email cannot be generated after 5 attempts.
        """
        start_prompt = self._yield_start_prompt(headline)
        payload = {"prompt": start_prompt}

        for attempt in range(5):
            try:
                response = requests.post(self.base_model_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    result_parsed = result.get("response", "").strip()
                    marker = "[DRAFT EMAIL]"
                    if marker in result_parsed:
                        start_email = result_parsed.split(marker, 1)[1].strip()
                        if start_email:
                            return start_email
                    raise ValueError("Draft email not found in response.")
                else:
                    raise ConnectionError(f"Unexpected status code: {response.status_code}")
            except Exception as e:
                time.sleep(2)

        raise Exception("Failed to generate a start email after 5 attempts.")

    def _refinement(self, headline: str, base_text: str) -> str:
        """
        Refine the draft email to produce a coherent final version.

        This method sends a refinement prompt to the Anthropic API endpoint.

        Args:
            headline (str): The news headline used for context.
            base_text (str): The initial draft email to be refined.

        Returns:
            str: The refined email.

        Raises:
            Exception: If the refinement process does not return a valid refined email.
        """
        refinement_prompt = self._yield_refinement_prompt(headline, base_text)
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=1.0,
                messages=[{"role": "user", "content": refinement_prompt}]
            )
        except Exception as e:
            raise Exception(f"Refinement API call failed: {e}")

        try:
            refined_text = response.content[0].text
            marker = "[REFINED EMAIL]"
            if marker in refined_text:
                refined_email = refined_text.split(marker, 1)[1].strip()
                if refined_email:
                    return refined_email
            raise ValueError("Refined email marker not found or empty response.")
        except Exception as e:
            raise Exception(f"Failed to parse the refined email: {e}")

    def _iterative_base(self, headline: str, previous_message: str) -> str:
        """
        Generate an answer email using an iterative base prompt.

        This method sends an iterative base prompt to the base model endpoint and retrieves the answer email.

        Args:
            headline (str): The news headline used for context.
            previous_message (str): The full email chain so far.

        Returns:
            str: The generated answer email.

        Raises:
            Exception: If a valid answer email cannot be generated after 5 attempts.
        """
        iterative_prompt = self._yield_iterative_base_prompt(headline, previous_message)
        payload = {"prompt": iterative_prompt}

        for attempt in range(5):
            try:
                response = requests.post(self.base_model_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    result_parsed = result.get("response", "").strip()
                    marker = "[ANSWER EMAIL]"
                    if marker in result_parsed:
                        answer_email = result_parsed.split(marker, 1)[1].strip()
                        if answer_email:
                            return answer_email
                    raise ValueError("Answer email not found in response.")
                else:
                    raise ConnectionError(f"Unexpected status code: {response.status_code}")
            except Exception as e:
                time.sleep(2)
        raise Exception("Failed to generate iterative answer email after 5 attempts.")

    def _iterative_refinement(self, headline: str, previous_message: str, base_text: str) -> str:
        """
        Refine the iterative answer email using an iterative refinement prompt.

        This method sends an iterative refinement prompt to the Anthropic API endpoint and retrieves the refined email.

        Args:
            headline (str): The news headline used for context.
            previous_message (str): The full email chain up to this point.
            base_text (str): The iterative answer email to be refined.

        Returns:
            str: The refined answer email.

        Raises:
            Exception: If the iterative refinement process does not return a valid refined email.
        """
        iterative_refinement_prompt = self._yield_iterative_refinement_prompt(headline, previous_message, base_text)
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=1.0,
                messages=[{"role": "user", "content": iterative_refinement_prompt}]
            )
        except Exception as e:
            raise Exception(f"Iterative refinement API call failed: {e}")

        try:
            refined_text = response.content[0].text
            marker = "[REFINED EMAIL]"
            if marker in refined_text:
                refined_email = refined_text.split(marker, 1)[1].strip()
                if refined_email:
                    return refined_email
            raise ValueError("Refined email marker not found or empty response in iterative refinement.")
        except Exception as e:
            raise Exception(f"Failed to parse the iterative refined email: {e}")

    def main(self) -> str:
        """
        Execute the complete email generation and iterative refinement process.

        1. Selects a random headline and generates an initial draft email.
        2. Refines the initial email.
        3. Iteratively feeds the complete email chain back into the base model to generate an answer,
           then refines that answer and appends it to the chain.
        4. Finally, applies a last refinement to adjust the format and extract labels.

        Returns:
            str: The final complete email chain after all refinement processes, formatted as JSON.

        Raises:
            Exception: If any step in the process fails.
        """
        try:
            headline = random.choice(self._news['headline'].tolist())
            self.logger.info(f"Selected Headline: {headline}")
            iterations = random.randint(1, 4)
            total_steps = 2 + iterations * 2

            with self.logger.progress_bar(total=total_steps, desc="Generating Email Chain") as pbar:
                start_email = self._generate_start_email(headline)
                pbar.update(1)
                refined_email = self._refinement(headline, start_email)
                pbar.update(1)

                email_chain = refined_email

                for _ in range(iterations):
                    answer_email = self._iterative_base(headline, email_chain)
                    pbar.update(1)
                    refined_answer = self._iterative_refinement(headline, email_chain, answer_email)
                    pbar.update(1)
                    email_chain += "\n\n" + refined_answer

            final_refined_email = self._final_refinement(headline, email_chain)
            return final_refined_email

        except Exception as e:
            self.logger.error(f"An error occurred during the main process: {e}")
            raise

    def _parse_live_api_response(self, response_text: str) -> dict:
        """
        Parse and clean the live API response string.

        This method removes any Markdown code fences and extraneous formatting from the response text,
        then attempts to parse the cleaned string into a JSON object.

        Args:
            response_text (str): The raw response string from the API.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If the response text cannot be parsed as valid JSON.
        """
        cleaned = response_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
        else:
            self.logger.warning("No valid JSON object found in the API response.")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON from API response: {e}") from e

    def run_aggregation(self) -> None:
        """
        Aggregate all JSON files in the output directory into a single aggregated JSON file.

        If an aggregated file already exists in self._directory (named "aggregated.json"),
        it loads the existing results and appends the new ones. Otherwise, it creates a new aggregated file.
        The final aggregated file is saved in the self._directory.
        """
        aggregated_file_path = os.path.join(self._directory, "aggregated.json")

        if os.path.exists(aggregated_file_path):
            try:
                with open(aggregated_file_path, "r", encoding="utf-8") as f:
                    aggregated_data = json.load(f)
                if not isinstance(aggregated_data, list):
                    self.logger.warning("Existing aggregated file does not contain a list. Overwriting with new aggregation.")
                    aggregated_data = []
            except Exception as e:
                self.logger.error(f"Failed to load existing aggregated file: {e}")
                aggregated_data = []
        else:
            aggregated_data = []

        for filename in os.listdir(self._output_directory):
            if filename.endswith(".json"):
                file_path = os.path.join(self._output_directory, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    aggregated_data.append(file_data)
                except Exception as e:
                    self.logger.error(f"Error loading {filename}: {e}")

        try:
            with open(aggregated_file_path, "w", encoding="utf-8") as f:
                json.dump(aggregated_data, f, indent=2)
            self.logger.ok(f"Aggregated file updated at {aggregated_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated file: {e}")

    def run_live_refinement(self, num_samples: int) -> None:
        """
        Run the live iterative refinement process multiple times and aggregate the results.

        This method runs the complete iterative refinement process (via the main() method)
        'num_samples' times. Each resulting refined email chain (formatted as JSON) is saved into a file
        in a directory named 'iterativebaserefine'. After processing all samples, an aggregated JSON file
        named 'aggregated.json' is created in the same directory containing all the results.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            None

        Raises:
            Exception: Propagates any exceptions raised during the refinement process.
        """
        results = []

        for i in range(num_samples):
            self.logger.info(f"Running live refinement sample {i+1} of {num_samples}...")
            refined_result = self.main()
            try:
                parsed_result = self._parse_live_api_response(refined_result)
            except ValueError as e:
                self.logger.error(f"Sample {i+1}: Failed to parse API response: {e}")
                continue

            with open(os.path.join(self._output_directory, f"sample_{i+1}.json"), "w", encoding="utf-8") as f:
                json.dump(parsed_result, f, indent=2)
            results.append(parsed_result)

        aggregated_filename = os.path.join(self._directory, "aggregated.json")
        try:
            with open(aggregated_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            self.logger.ok(f"Aggregated results saved to {aggregated_filename}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")