import os
import re
import json
import shutil
import pandas as pd
from datetime import datetime
import requests

from config.logger import CustomLogger


class DeepseekLocalGenerator:
    """
    A generator class that interacts with the DeepSeek‑R1 local LM Studio API to produce email samples.

    **DeepSeek‑R1 Recommendations**:
        - Temperature: 0.5–0.7 (default: 0.6).
        - No system prompt (all instructions should be part of the user prompt).
        - Prepend '<think>\\n' to the user prompt to encourage the model to perform reasoning.
        - Omit max_tokens to allow for unbounded output.
    """

    def __init__(
        self,
        mail_gen,
        model: str = "deepseek-r1-distill-qwen-32b",
        temperature: float = 0.6,
        base_url: str = "http://localhost:1234/v1"
    ):
        """
        Initialize a DeepseekLocalGenerator instance.

        This constructor:
          - Stores references to an email prompt generator (mail_gen).
          - Initializes a custom logger.
          - Sets up configuration for the local LM Studio API (model, temperature, etc.).
          - Attempts to locate a distance matrix file relative to this script.
          - Creates necessary directories for storing raw, processed, and aggregated outputs.

        Args:
            mail_gen (EmailGenerator): An instance providing a construct_prompt() method to generate email creation prompts.
            model (str): The DeepSeek‑R1 model name to use. Default: "deepseek-r1-distill-qwen-32b".
            temperature (float): Sampling temperature, recommended 0.6 for DeepSeek‑R1. Default: 0.6.
            base_url (str): The base URL of the local LM Studio API endpoint. Default: "http://localhost:1234/v1".
        """
        self.mail_gen = mail_gen
        self.logger = CustomLogger("DeepseekLocalGenerator")
        self.logger.ok("DeepseekLocalGenerator initialized")
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.endpoint = f"{self.base_url}/chat/completions"

        # ------------------------------------------------------------------
        # Relative path for distance_matrix.xlsx
        # ------------------------------------------------------------------
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        self.distance_matrix_path = os.path.normpath(
            os.path.join(script_dir, "..", "datasets_processed", "distance_matrix.xlsx")
        )

        # ------------------------------------------------------------------
        # Creation of Subdirectories
        # ------------------------------------------------------------------
        self.localemails_dir = os.path.normpath(
            os.path.join(script_dir, "..", "localemails")
        )
        os.makedirs(self.localemails_dir, exist_ok=True)

        self.raw_dir = os.path.join(self.localemails_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.processed_dir = os.path.join(self.localemails_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        self.aggregated_dir = os.path.join(self.localemails_dir, "aggregated")
        os.makedirs(self.aggregated_dir, exist_ok=True)

    def _clean_sample(self, sample_str: str) -> dict:
        """
        Clean and parse a single generated sample string into JSON.

        This method:
          1. Strips leading/trailing whitespace and quotes.
          2. Removes any <think>...</think> blocks (chain-of-thought).
          3. Strips triple-backtick fences (```json) if present.
          4. Attempts to parse the resulting content as JSON.

        Args:
            sample_str (str): The raw generated text that may contain chain-of-thought or markdown fences.

        Raises:
            ValueError: If the cleaned content is not valid JSON (json.JSONDecodeError internally).

        Returns:
            dict: A Python dictionary parsed from the cleaned JSON string.
        """

        text = sample_str.strip()

        # ------------------------------------------------------------------
        # Repeated Stripping of Quotes
        # ------------------------------------------------------------------
        def is_wrapped_in_quotes(s):
            return (
                    (s.startswith('"') and s.endswith('"')) or
                    (s.startswith("'") and s.endswith("'"))
            )

        while is_wrapped_in_quotes(text):
            text = text[1:-1].strip()

        # ------------------------------------------------------------------
        # Removal of <think/> tag
        # ------------------------------------------------------------------
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # ------------------------------------------------------------------
        # Removal of triple-backticks
        # ------------------------------------------------------------------
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```", "", text)

        # ------------------------------------------------------------------
        # Removal of Chinese characters
        # ------------------------------------------------------------------
        text = re.sub(r"[\u4e00-\u9fff]+", "", text).strip()

        # ------------------------------------------------------------------
        # Parsing Attempt
        # ------------------------------------------------------------------
        try:
            return json.loads(text)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed. Final cleaned text: {text}")
            raise ValueError(f"Error parsing JSON from sample")

    def _aggregate_results(self) -> None:
        """
        Aggregate and standardize all raw samples into one aggregated JSON file.

        This method:
          - Iterates over .json files in the 'raw' directory.
          - Reads each file as a DataFrame, extracts text or dict samples from the first column.
          - Cleans and converts each sample to valid JSON (via _clean_sample).
          - Appends all valid JSON samples to an aggregated list.
          - Moves processed raw files into the 'processed' subdirectory.
          - Loads any existing aggregated JSON from 'aggregated.json' (if present),
            extends it with new results, and saves it back to disk.
        """
        new_results = []


        for filename in os.listdir(self.raw_dir):
            file_path = os.path.join(self.raw_dir, filename)

            if not filename.lower().endswith(".json"):
                continue

            try:
                df = pd.read_json(file_path)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

            if df.empty:
                self.logger.warning(f"No data found in {file_path}")
                continue

            raw_samples = df.iloc[:, 0].tolist()
            for sample in raw_samples:
                if isinstance(sample, dict):
                    new_results.append(sample)
                elif isinstance(sample, str):
                    try:
                        cleaned_sample = self._clean_sample(sample)
                        new_results.append(cleaned_sample)
                    except Exception as e:
                        self.logger.error(f"Error cleaning sample in {filename}: {e}")
                else:
                    self.logger.warning(
                        f"Skipping non-string, non-dict sample in {filename}: {sample}"
                    )

            dest_path = os.path.join(self.processed_dir, filename)
            try:
                shutil.move(file_path, dest_path)
                self.logger.info(f"Moved {filename} to processed directory.")
            except Exception as e:
                self.logger.error(f"Error moving file {filename} to processed: {e}")

        aggregated_file = os.path.join(self.aggregated_dir, "aggregated.json")
        aggregated_results = []

        if os.path.exists(aggregated_file):
            try:
                with open(aggregated_file, "r", encoding="utf-8") as f:
                    aggregated_results = json.load(f)

            except Exception as e:
                self.logger.error(f"Error loading existing aggregated file: {e}")

        aggregated_results.extend(new_results)

        try:
            with open(aggregated_file, "w", encoding="utf-8") as f:
                json.dump(aggregated_results, f, indent=2)
            self.logger.info(f"Aggregated results saved to {aggregated_file}")

        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")

    def _generate_response(self, prompt: str):
        """
        Send a prompt to the local LM Studio API and return the model's raw output string.

        This method:
          - Optionally prepends '<think>\\n' if not present (DeepSeek recommendation).
          - Constructs the payload with the user prompt, model name, and temperature.
          - Sends a POST request to /chat/completions on the local API.
          - Returns the content of choices[0].message.content if present, else an empty string or None.

        Args:
            prompt (str): The textual prompt to be sent to the local model.

        Returns:
            str or None: The raw text output from the model if successful, otherwise None.
        """
        # ------------------------------------------------------------------
        # Prepending of <think/> tag, recommended by Deepseek
        # ------------------------------------------------------------------
        if not prompt.startswith("<think>\n"):
            prompt = "<think>\n" + prompt

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.endpoint, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.logger.error(f"HTTP Request failed: {err}")
            return None

        try:
            result = response.json()
        except json.JSONDecodeError:
            self.logger.error(f"Response not valid JSON. Text:\n{response.text}")
            return None

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content

    def run_local_gen(self, number_of_samples: int) -> None:
        """
        Perform a complete local generation run, from prompt creation to result aggregation.

        Steps:
          1. Generate prompts by calling construct_prompt() on mail_gen, the specified number of times.
          2. For each prompt, call the local LM Studio API via _generate_response.
          3. Save all raw results as a JSON list to a timestamped file in 'raw'.
          4. Call _aggregate_results to clean and merge new samples into an aggregated JSON file.

        Args:
            number_of_samples (int): Number of prompts/samples to generate and retrieve from the local model.

        Returns:
            None
        """
        self.logger.info("Starting local DeepSeek generation...")

        samples = []
        for i in range(number_of_samples):
            prompt = self.mail_gen.construct_prompt()
            self.logger.info(f"Generating sample {i+1}")
            content = self._generate_response(prompt)
            if content is not None:
                samples.append(content)

        if not samples:
            self.logger.warning("No samples generated.")
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        raw_output_path = os.path.join(self.raw_dir, f"{timestamp}.json")
        try:
            with open(raw_output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved raw results to {raw_output_path}")
        except Exception as e:
            self.logger.error(f"Error saving raw results: {e}")
            return

        self._aggregate_results()
        self.logger.info("Local DeepSeek generation completed.")