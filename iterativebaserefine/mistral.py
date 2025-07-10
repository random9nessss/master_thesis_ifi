import pandas as pd
import random
import os
import json
import requests
import re
import time
import httpx
from mistralai import Mistral

from config.logger import CustomLogger
from attributes.email_attribute_sampler import AttributeSampler

from dotenv import load_dotenv
load_dotenv("ENV.txt")


class IterativeSeededBareMistral:
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

        Sets up the base URL for the starting email generation, the Gemini API client,
        the attribute sampler for placeholder data, and loads news headlines from specified JSON files.

        Raises:
            EnvironmentError: If the required GOOGLE_KEY environment variable is not set.
        """
        self.base_model_url = "http://localhost:8000/generate"

        mistral_key = os.getenv("MISTRAL_KEY")
        if not mistral_key:
            raise EnvironmentError("MISTRAL_KEY environment variable not set.")

        # Initialize Gemini client
        self._client = Mistral(api_key=os.getenv("MISTRAL_KEY"),
                               client=httpx.Client(verify=False))

        # ------------------------------------------------------------------
        # Directory Setup
        # ------------------------------------------------------------------
        try:
            repo_root = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            repo_root = os.getcwd()

        self._directory = os.path.join(repo_root, "syntheticdata", "iterativebaserefine", "mistral")
        self._output_directory = os.path.join(self._directory, "output")

        os.makedirs(self._directory, exist_ok=True)
        os.makedirs(self._output_directory, exist_ok=True)

        self._news = self._load_news_headlines()
        self._attribute_sampler = AttributeSampler()
        self.logger = CustomLogger(name="IterativeBaseRefine")

    def _mistral_submit_prompt(self, prompt: str, model: str = "mistral-large-latest") -> str:
        """
        Submit the given prompt to the Mistral API and return the generated content.
        Implements a retry mechanism for handling server errors.

        Args:
            prompt (str): The prompt to send to the API.
            model (str, optional): The model identifier. Defaults to "mistral-large-latest".

        Returns:
            str: The generated text content from the model.
        """
        response = self._client.chat.complete(
                    model=model,
                    messages=[
                        {
                        "content": prompt,
                        "role": "user"
                        }
                    ]
                )

        return response.choices[0].message.content

    def _yield_start_prompt(self, headline: str) -> str:
        """
        Generate the initial email draft prompt using the provided news headline.

        Args:
            headline (str): The news headline to base the email content on.

        Returns:
            str: The generated start prompt string for the email draft.
        """
        return f"""
            You are in the chartering business as a broker, providing your customer with attractive freight rates and intel.
            Draft a negotiation email for your business counterparty. The subject is {headline}.

            [DRAFT EMAIL]
        """

    def _yield_refinement_prompt(self, headline: str, base_text: str) -> str:
        """
        Generate the prompt used to refine an initial draft email.

        Args:
            headline (str): The news headline associated with the email.
            base_text (str): The initial email draft text to be refined.

        Returns:
            str: The prompt string for refining the email.
        """
        return f"""
            Please refine the following email chain to make it coherent and realistic without changing the theme,
            then just return the plain, refined email body after the tag [REFINED EMAIL] without any subject, timestamp, etc.

            The email is related to {headline} and is as follows:
            {base_text}
        """

    def _yield_iterative_base_prompt(self, headline: str, previous_message: str) -> str:
        """
        Generate the prompt for iterative email generation based on previous email chain.

        Args:
            headline (str): The news headline that contextualizes the email conversation.
            previous_message (str): The current email chain to be extended.

        Returns:
            str: The generated prompt string for iterative base email generation.
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
            Keep the original tone, style, and theme unchanged. Then just return the plain, refined email body after an included tag [REFINED EMAIL] without any subject, timestamp, etc.

            The email is related to {headline} and is the answer to the following email chain:
            {previous_message}

            The answer email to be refined is as follows:
            {base_text}

            [REFINED EMAIL]
        """

    def _yield_final_refinement_prompt(self, headline: str, final_email_chain: str) -> str:
        """
        Generate the prompt to perform final refinement and reformat the complete email chain into JSON.

        This prompt incorporates random attribute sampling to adjust placeholders and non-specific terms.

        Args:
            headline (str): The news headline that contextualizes the email conversation.
            final_email_chain (str): The complete email chain before final refinement.

        Returns:
            str: The final refinement prompt string for reformatting the email chain into JSON.
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

            json
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

        """

    def _final_refinement(self, headline: str, final_email: str) -> str:
        """
        Perform final refinement on the complete email chain using a model prompt.

        Args:
            headline (str): The news headline that contextualizes the email.
            final_email (str): The complete email chain to be refined.

        Returns:
            str: The refined email chain as generated by the model.

        Raises:
            Exception: If the API call for final refinement fails.
        """
        final_refinement_prompt = self._yield_final_refinement_prompt(headline, final_email)
        try:
            response_text = self._mistral_submit_prompt(final_refinement_prompt, model="mistral-large-latest")
            return response_text
        except Exception as e:
            raise Exception(f"Final refinement API call failed: {e}")

    def _load_news_headlines(self) -> pd.DataFrame:
        """
        Load news headlines from predefined JSON files and aggregate them into a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing a single column 'headline' with all loaded news headlines.

        Raises:
            ValueError: If no headlines are loaded from any source.
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
        Generate the initial draft email for a given headline by sending a prompt to the base model.

        The method attempts up to 5 times to generate a valid draft email containing the designated marker.

        Args:
            headline (str): The news headline to base the email on.

        Returns:
            str: The generated draft email text.

        Raises:
            Exception: If a valid draft email cannot be generated after 5 attempts.
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
        Refine the initial draft email by submitting a refinement prompt to the model.

        Args:
            headline (str): The news headline associated with the email.
            base_text (str): The initial draft email text to be refined.

        Returns:
            str: The refined email text.

        Raises:
            Exception: If the refinement API call fails or the refined email cannot be parsed.
        """
        refinement_prompt = self._yield_refinement_prompt(headline, base_text)
        try:
            response_text = self._mistral_submit_prompt(refinement_prompt, model="mistral-large-latest")
        except Exception as e:
            raise Exception(f"Refinement API call failed: {e}")

        try:
            marker = "[REFINED EMAIL]"
            if marker in response_text:
                refined_email = response_text.split(marker, 1)[1].strip()
                if refined_email:
                    return refined_email
            raise ValueError("Refined email marker not found or empty response.")
        except Exception as e:
            raise Exception(f"Failed to parse the refined email: {e}")

    def _iterative_base(self, headline: str, previous_message: str) -> str:
        """
        Generate an iterative answer email based on the current email chain by sending a prompt to the base model.

        The method attempts up to 5 times to generate a valid answer email containing the designated marker.

        Args:
            headline (str): The news headline that contextualizes the email conversation.
            previous_message (str): The current email chain that the answer will extend.

        Returns:
            str: The generated answer email text.

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
        Refine the iteratively generated answer email by submitting a refinement prompt to the model.

        Args:
            headline (str): The news headline associated with the email.
            previous_message (str): The existing email chain.
            base_text (str): The iteratively generated answer email to be refined.

        Returns:
            str: The refined answer email text, or the original text if no specific marker is found.

        Raises:
            Exception: If the iterative refinement API call fails or the refined email cannot be parsed.
        """
        iterative_refinement_prompt = self._yield_iterative_refinement_prompt(headline, previous_message, base_text)
        try:
            response_text = self._mistral_submit_prompt(iterative_refinement_prompt, model="mistral-large-latest")
        except Exception as e:
            raise Exception(f"Iterative refinement API call failed: {e}")

        try:
            marker = "[REFINED EMAIL]"
            if marker in response_text:
                refined_email = response_text.split(marker, 1)[1].strip()
                if refined_email:
                    return refined_email
            else:
                return response_text
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
        cleaned = re.sub(r"^json\s*", "", cleaned)
        cleaned = re.sub(r"\s*$", "", cleaned)
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

        The method loads existing aggregated data if available, appends new data from output files,
        and then writes the complete aggregation back to disk.
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

        For each sample, the process generates and refines an email chain, parses the API response,
        and saves the individual sample as a JSON file. Finally, all results are aggregated and saved.

        Args:
            num_samples (int): The number of live refinement samples to run.
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

            with open(os.path.join(self._output_directory, f"sample_{int(time.time())}.json"), "w", encoding="utf-8") as f:
                json.dump(parsed_result, f, indent=2)
            results.append(parsed_result)

        aggregated_filename = os.path.join(self._directory, "aggregated.json")
        try:
            with open(aggregated_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            self.logger.ok(f"Aggregated results saved to {aggregated_filename}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")