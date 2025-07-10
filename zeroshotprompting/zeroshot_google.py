import os
import re
import httpx
import shutil
import json
import time
import pandas as pd
from google import genai
from datetime import datetime

from attributes.few_shot_prompt import FewShotPrompt
from attributes.zero_shot_prompt import ZeroShotPrompt
from config.logger import CustomLogger
from dotenv import load_dotenv


class GoogleZeroShotClient:

    def __init__(self, mode: str = "zeroshot"):

        assert mode in ["zeroshot", "fewshot"], f"Mode provided <{mode}> is invalid"

        self.logger = CustomLogger(name="GoogleZeroShotClient")
        self.logger.ok("GoogleZeroShotClient initialized")

        # ------------------------------------------------------------------
        # Mode Loading
        # ------------------------------------------------------------------
        self._mode = mode
        self._zero_shot_module = ZeroShotPrompt()
        self._few_shot_module = FewShotPrompt()

        # ------------------------------------------------------------------
        # Loading API Key and Initializing API Client
        # ------------------------------------------------------------------
        load_dotenv("ENV.txt")

        if not os.getenv("GOOGLE_KEY"):
            self.logger.error("Google API Key not found. Set GOOGLE_KEY in ENV.txt")
            exit(1)

        self._client = genai.Client(api_key=os.getenv("GOOGLE_KEY"))

        # ------------------------------------------------------------------
        # Directory Setup
        # ------------------------------------------------------------------

        # .py files
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        # .ipynb files
        except NameError:
            current_dir = os.getcwd()

        repo_root = self.find_masterthesis_dev_path(current_dir)
        self._directory = os.path.join(repo_root, "syntheticdata", self._mode ,"gemini")
        self._input_directory = os.path.join(self._directory, "input")
        self._output_directory = os.path.join(self._directory, "output")

        try:
            os.makedirs(self._directory, exist_ok=True)
            self.logger.info(f"Directory {self._directory} created")

            os.makedirs(self._input_directory, exist_ok=True)
            self.logger.info(f"Input directory {self._input_directory} ready")

            os.makedirs(self._output_directory, exist_ok=True)
            self.logger.info(f"Output directory {self._output_directory} ready")

        except OSError as e:
            self.logger.error(f"Error creating directories: {str(e)}")
            exit(1)

        # ------------------------------------------------------------------
        # Task and Index Setup
        # ------------------------------------------------------------------
        self._index = 0
        self._tasks = []
        self._responses = []

    def find_masterthesis_dev_path(self, start_path=None):
        """
        Traverse upward from the given start_path until a directory named 'Masterthesis-dev' is found.

        Parameters:
            start_path (str, optional): The starting path to begin the search. Defaults to the current working directory.

        Returns:
            str: The absolute path to the 'Masterthesis-dev' directory.

        Raises:
            FileNotFoundError: If no directory named 'Masterthesis-dev' is found.
        """
        if start_path is None:
            start_path = os.getcwd()

        current_path = os.path.abspath(start_path)

        while True:
            if os.path.basename(current_path) == "Masterthesis-dev":
                return current_path

            parent_path = os.path.dirname(current_path)

            if parent_path == current_path:
                raise FileNotFoundError(
                    "Could not locate a folder named 'Masterthesis-dev' above {}".format(start_path))

            current_path = parent_path

    def _increment_index(self):
        """
        Increment the internal index counter by 1.
        """
        self._index += 1

    def _get_index(self):
        """
        Retrieve the current value of the internal index counter.

        Returns:
            int: The current index value.
        """
        return self._index

    def _clean_sample(self, sample_str: str) -> dict:
        """
        Attempt to parse sample_str as JSON directly.
        If that fails, remove Markdown formatting and try again.

        Parameters:
            sample_str (str): The raw sample string, potentially wrapped in Markdown fences.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If no valid JSON object can be parsed.
        """
        try:
            return json.loads(sample_str)
        except json.JSONDecodeError:
            cleaned = sample_str.strip()
            cleaned = re.sub(r"^```json\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1:
                self.logger.warning("No valid JSON object found in sample.")
                raise ValueError("No valid JSON object found in sample.")
            cleaned = cleaned[start:end + 1]
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON still invalid after cleaning: {e}")
                raise ValueError("JSON still invalid after cleaning.") from e

    def _aggregate_results(self) -> None:
        """
        Aggregate results from the output directory by reading each JSON file,
        cleaning each sample (if needed), and merging them into a single aggregated JSON file.
        Processed files are moved to a processed directory.

        This version ensures that only valid, nonempty JSON objects are aggregated.
        It also attempts to change the aggregated file's permissions to avoid read-only issues.
        """
        new_results = []

        aggregated_dir = os.path.join(self._directory, "aggregated")
        os.makedirs(aggregated_dir, exist_ok=True)
        processed_dir = os.path.join(self._directory, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        for filename in os.listdir(self._output_directory):
            file_path = os.path.join(self._output_directory, filename)
            if not filename.lower().endswith(".json"):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

            samples = data if isinstance(data, list) else [data]

            for sample in samples:
                if isinstance(sample, dict):
                    if sample:
                        new_results.append(sample)
                    else:
                        self.logger.warning(f"Empty dictionary found in {filename}.")
                elif isinstance(sample, str):
                    try:
                        cleaned_sample = self._clean_sample(sample)
                        if cleaned_sample and isinstance(cleaned_sample, dict):
                            new_results.append(cleaned_sample)
                        else:
                            self.logger.warning(f"Cleaned sample from {filename} is empty or invalid.")
                    except Exception as e:
                        self.logger.error(f"Error cleaning sample from {filename}: {e}")
                else:
                    self.logger.warning(f"Unexpected sample type in {filename}: {type(sample)}")

            dest_path = os.path.join(processed_dir, filename)
            try:
                shutil.move(file_path, dest_path)
                self.logger.info(f"Moved {filename} to processed directory.")
            except Exception as e:
                self.logger.error(f"Error moving file {filename} to processed: {e}")

        aggregated_file = os.path.join(aggregated_dir, "aggregated.json")
        aggregated_results = []
        if os.path.exists(aggregated_file):
            try:
                with open(aggregated_file, "r", encoding="utf-8") as f:
                    aggregated_results = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading existing aggregated file: {e}")

        aggregated_results.extend(new_results)

        if os.path.exists(aggregated_file):
            try:
                os.chmod(aggregated_file, 0o666)
            except Exception as e:
                self.logger.error(f"Error changing permissions of aggregated file: {e}")

        try:
            with open(aggregated_file, "w", encoding="utf-8") as f:
                json.dump(aggregated_results, f, indent=2)
            self.logger.ok(f"Aggregated results saved to {aggregated_file}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")

    def _submit_prompt(self, prompt: str, model: str = "gemini-2.0-flash") -> str:
        """
        Submit the given prompt to the Google API and retrieve the generated content.

        Parameters:
            prompt (str): The prompt to send to the API.
            model (str, optional): The model to use for generation. Defaults to "gemini-2.0-flash".

        Returns:
            str: The generated content as a string.
        """
        max_retries = 5
        delay = 2
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if hasattr(e, "status_code") and e.status_code == 503:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed with 503 UNAVAILABLE. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    self.logger.error(f"API error: {e}")
                    time.sleep(delay)
                    delay *= 2

        self.logger.error("Max retries reached. Service unavailable.")
        raise Exception("Max retries reached for generating content.")

    def run_inference(self, number_samples: int = 10):
        """
        Run inference on a given number of samples using the email generator and Google API.

        This method performs the following steps for each sample:
            1. Generates a task prompt using the email generator.
            2. Appends the task prompt to the internal task list.
            3. Serializes the task prompt to a JSONL file in the input directory.
            4. Submits the prompt to the Google API.
            5. Cleans and parses the raw response.
            6. Prints the raw and cleaned responses.
            7. Serializes the cleaned response to a JSONL file in the output directory.

        Note:
            Adheres to free tier API limits:
              - Maximum 15 requests per minute.
              - Maximum 1500 requests per day.

        Args:
            number_samples (int, optional): The number of samples to process. Defaults to 10.

        Returns:
            None
        """
        tasks_file_path = os.path.join(self._input_directory, f"tasks_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        outputs_file_path = os.path.join(self._output_directory,
                                         f"outputs_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")

        for idx in range(number_samples):
            # ------------------------------------------------------------------
            # Task Generation and Serialization
            # ------------------------------------------------------------------
            if self._mode == "zeroshot":
                task = self._zero_shot_module._prompt
                self._tasks.append(task)

            elif self._mode == "fewshot":
                task = self._few_shot_module.generate_few_shot_prompt()
                self._tasks.append(task)

            else:
                raise NotImplemented(f"Currently only <zeroshot> and <fewshot> are supported.")

            try:
                with open(tasks_file_path, "a", encoding="utf-8") as tf:
                    tf.write(json.dumps(task) + "\n")
            except Exception as e:
                self.logger.error(f"Error writing task to {tasks_file_path}: {e}")

            # ------------------------------------------------------------------
            # Submit Prompt and Retrieve Response
            # ------------------------------------------------------------------
            try:
                response_raw = self._submit_prompt(task)
                response_clean = self._clean_sample(response_raw)
                self._responses.append(response_clean)
            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {e}")
                continue

            # ------------------------------------------------------------------
            # Increment Index and Throttle if Needed
            # ------------------------------------------------------------------
            self._increment_index()

            # ------------------------------------------------------------------
            # Increment Index and Throttle if Needed
            # ------------------------------------------------------------------
            if (idx + 1) % 15 == 0 and (idx + 1) < number_samples:
                self.logger.info("Rate limit reached. Sleeping for 60 seconds.")
                time.sleep(60)

        # ------------------------------------------------------------------
        # Result Serialization
        # ------------------------------------------------------------------
        try:
            with open(outputs_file_path, "w", encoding="utf-8") as of:
                json.dump(self._responses, of, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing output to {outputs_file_path}: {e}")

        # ------------------------------------------------------------------
        # Result Aggregation
        # ------------------------------------------------------------------
        self._aggregate_results()

        # ------------------------------------------------------------------
        # Clearing Tasks
        # ------------------------------------------------------------------
        self._tasks.clear()