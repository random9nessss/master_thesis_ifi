import os
import re
import json
import time
import shutil
import pandas as pd
import httpx
from datetime import datetime
from google import genai
from dotenv import load_dotenv
from config.logger import CustomLogger


class GoogleRefine:
    """
    Handles the refinement process of synthetic email chains using Google's Gemini API.

    This class reads a JSONL file containing base chains (each with a "chain" field),
    constructs a detailed refinement prompt for each entry, submits them to the Gemini API,
    cleans and aggregates the responses, and saves the outputs.
    """

    def __init__(self, base_model: str, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the GeminiRefine processor by setting up directories, loading the API key,
        and initializing the Gemini API client.

        Args:
            model_name (str, optional): The name of the Gemini model to use. Defaults to "gemini-2.0-flash".
        """
        self.logger = CustomLogger(name="GeminiRefine")
        self.logger.ok("GeminiRefine initialized")

        # -------------------------------------------------------------------
        # Load Environment Variables and Initialize Google Client
        # -------------------------------------------------------------------
        load_dotenv("ENV.txt")
        if not os.getenv("GOOGLE_KEY"):
            self.logger.error("Google API Key not found. Set GOOGLE_KEY in ENV.txt")
            exit(1)

        self._client = genai.Client(api_key=os.getenv("GOOGLE_KEY"))
        self.model_name = model_name

        # -------------------------------------------------------------------
        # Directory Setup
        # -------------------------------------------------------------------
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()

        repo_root = self.find_masterthesis_dev_path(current_dir)
        self._directory = os.path.join(repo_root, "syntheticdata", "baserefine", "refine", base_model, "gemini")
        self._input_directory = os.path.join(self._directory, "input")
        self._output_directory = os.path.join(self._directory, "output")
        self._processed_directory = os.path.join(self._directory, "processed")
        self._aggregated_directory = os.path.join(self._directory, "aggregated")

        for d in [self._directory, self._input_directory, self._output_directory,
                  self._processed_directory, self._aggregated_directory]:
            os.makedirs(d, exist_ok=True)
            self.logger.info(f"Directory {d} ready")

        # -------------------------------------------------------------------
        # Task and Index Setup
        # -------------------------------------------------------------------
        self._tasks = []
        self._responses = []
        self._index = 0

    def find_masterthesis_dev_path(self, start_path=None):
        """
        Traverse upward from the given start path until a directory named 'Masterthesis-dev' is found.

        Args:
            start_path (str, optional): The starting path to begin the search. Defaults to the current working directory.

        Returns:
            str: The absolute path to the 'Masterthesis-dev' directory.

        Raises:
            FileNotFoundError: If no such directory is found.
        """
        if start_path is None:
            start_path = os.getcwd()
        current_path = os.path.abspath(start_path)

        while True:
            if os.path.basename(current_path) == "Masterthesis-dev":
                return current_path
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                raise FileNotFoundError(f"Could not locate a folder named 'Masterthesis-dev' above {start_path}")
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

    def _retrieve_refinement_prompt(self, base_text: str) -> str:
        """
        Constructs the refinement prompt for Gemini's API given a base email chain text.

        Args:
            base_text (str): The unrefined email chain text.

        Returns:
            str: The formatted refinement prompt.
        """
        return f"""
            Please refine the following email chain to make it coherent and realistic without changing the theme, then convert it to well-formed JSON as specified.
            
            ---
            Unrefined email chain:
            {base_text}
            ---
            
            Formatting requirements:
            1. Follow the JSON structure below exactly.
            2. Ensure each email in the chain is refined, logically consistent, and preserves its meaning.
            3. Include a `labels` object at the end with as many fields as can be reasonably inferred (leave blank if uncertain).
            4. Maintain the chronological order of emails. If necessary, add or estimate timestamps in the format YYYY-MM-DD HH:MM.
            5. Use realistic email addresses based on names in the text.
            6. Do not add content not present in the base text.
            
            JSON structure:
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
        """

    def create_refinement_tasks_from_jsonl(self, input_jsonl_file: str):
        """
        Reads a JSON Lines file and constructs a refinement task for each entry.

        Expects each line to be a JSON object with a "chain" field.

        Args:
            input_jsonl_file (str): Path to the JSONL file containing base email chains.

        Returns:
            None
        """
        if not os.path.exists(input_jsonl_file):
            self.logger.error(f"Input file does not exist: {input_jsonl_file}")
            return

        with open(input_jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.logger.info(f"Loaded {len(lines)} lines from '{input_jsonl_file}' for refinement.")

        for i, line in enumerate(lines, start=1):
            try:
                data = json.loads(line.strip())
                chain_text = data.get("chain", "")
                if not chain_text:
                    self.logger.warning(f"No chain text found for line {i}. Skipping.")
                    continue

                prompt = self._retrieve_refinement_prompt(chain_text)
                self.append_custom_prompt_task(prompt)
                self.logger.info(f"Refinement prompt for chain {i} queued.")
            except Exception as e:
                self.logger.error(f"Error parsing line {i}: {e}")

    def append_custom_prompt_task(self, prompt: str):
        """
        Appends a custom prompt task to the internal task list.

        Args:
            prompt (str): The refinement prompt to be added as a task.

        Returns:
            None
        """
        task = {
            "custom_id": str(self._get_index()),
            "prompt": prompt
        }

        self._increment_index()
        self._tasks.append(task)
        self.logger.info(f"Task #{self._index} added with custom prompt.")


    def _submit_prompt(self, prompt: str) -> str:
        """
        Submits a prompt to the Gemini API and returns the generated text response.

        This method attempts to generate content using the specified Gemini API model.
        If a 503 (UNAVAILABLE) error occurs, it will retry the request up to a maximum of 5 attempts,
        with an exponential backoff starting at 2 seconds. For other server errors or unexpected exceptions,
        it logs the error and re-raises the exception.

        Parameters:
            prompt (str): The text prompt to submit to the Gemini API.

        Returns:
            str: The generated text response from the API.

        Raises:
            Exception: If the maximum number of retries is reached or a non-retriable error occurs.
        """
        max_retries = 5
        delay = 2
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            except genai.errors.ServerError as e:
                if hasattr(e, "status_code") and e.status_code == 503:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed with 503 UNAVAILABLE. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    self.logger.error(f"API error: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error during prompt submission: {e}")
                raise

        self.logger.error("Max retries reached. Service unavailable.")
        raise Exception("Max retries reached for generating content.")


    def _clean_sample(self, sample_str: str) -> dict:
        """
        Cleans and parses a sample string that is expected to contain JSON content.

        This method first attempts to directly parse the provided string as JSON.
        If parsing fails, it removes Markdown formatting (such as code block markers) and extracts
        the JSON content by identifying the first '{' and the last '}' characters.
        If a valid JSON object cannot be obtained, a ValueError is raised.

        Parameters:
            sample_str (str): The string containing the JSON sample, possibly with Markdown formatting.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If a valid JSON object cannot be obtained after cleaning.
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


    def serialize_response(self, response, filename: str = None) -> str:
        """
        Serializes the given response to a JSON file in the output directory.

        The response is appended as a new line in JSON format.
        If no filename is provided, a timestamped filename is generated.
        Logs the outcome of the file operation.

        Parameters:
            response: The response object (typically a dictionary) to be serialized.
            filename (str, optional): The name of the file to which the response will be serialized.
                                      If not provided, a timestamped filename is generated.

        Returns:
            str: The full file path of the serialized file if successful, or an empty string if an error occurred.
        """
        if not filename:
            filename = f"tasks_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        filepath = os.path.join(self._output_directory, filename)
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps({"response": response}) + "\n")
            self.logger.ok(f"Response appended to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error serializing response: {e}")
            return ""


    def aggregate_results(self) -> None:
        """
        Aggregates individual response files from the output directory into a single JSON file.

        This method reads all JSON and JSONL files from the output directory, cleans and merges their content,
        and appends them to an aggregated JSON file located in an 'aggregated' subdirectory.
        After processing each file, the original file is moved to a 'processed' subdirectory to avoid reprocessing.
        Any errors during reading or moving files are logged, and processing continues with the next file.
        """
        new_results = []
        aggregated_dir = os.path.join(self._directory, "aggregated")
        os.makedirs(aggregated_dir, exist_ok=True)
        processed_dir = os.path.join(self._directory, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        for filename in os.listdir(self._output_directory):
            if not filename.lower().endswith(".json") and not filename.lower().endswith(".jsonl"):
                continue

            file_path = os.path.join(self._output_directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if filename.lower().endswith(".jsonl"):
                        for line in f:
                            data = json.loads(line)
                            new_results.append(data)
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            new_results.extend(data)
                        else:
                            new_results.append(data)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

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
        try:
            with open(aggregated_file, "w", encoding="utf-8") as f:
                json.dump(aggregated_results, f, indent=2)
            self.logger.ok(f"Aggregated results saved {aggregated_file}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")


    def run_refinement(self, input_jsonl_file: str, output_filename: str, number_samples: int = None):
        """
        Executes the full refinement process for processing a dataset of prompts.

        The process involves:
          1. Loading a base dataset from a JSON Lines (JSONL) file.
          2. Optionally reducing the dataset to a specified number of samples.
          3. Processing the dataset in batches of up to 50 rows.
          4. For each row in a batch, constructing a refinement prompt using the base chain,
             and queuing the prompt as a task.
          5. For each task:
             - Submitting the prompt to the Gemini API.
             - Cleaning the returned response.
             - Immediately serializing the cleaned response to disk.
             - Aggregating the response into a single aggregated file.
          6. Incorporating rate limiting by pausing between tasks and batches as necessary.

        Parameters:
            input_jsonl_file (str): Path to the JSONL file containing the base dataset.
            output_filename (str): Base name for output files generated during the process.
            number_samples (int, optional): If provided and less than the total number of samples,
                                            only the last `number_samples` from the dataset are processed.

        Raises:
            Exception: If critical errors occur during file I/O or API interactions.
        """
        df_base = pd.read_json(input_jsonl_file, lines=True)
        if number_samples is not None and number_samples < len(df_base):
            df_base = df_base.tail(number_samples)

        batch_size = 50
        for batch_start in range(0, len(df_base), batch_size):
            df_batch = df_base.iloc[batch_start: batch_start + batch_size]

            for idx, row in df_batch.iterrows():
                base_chain = row.get("chain", "")
                if not base_chain:
                    self.logger.warning(f"Row {idx} doesn't have a 'chain' field. Skipping.")
                    continue
                prompt = self._retrieve_refinement_prompt(base_chain)
                self.append_custom_prompt_task(prompt)

            total_tasks_in_batch = len(self._tasks)
            for i, task in enumerate(self._tasks):
                prompt = task.get("prompt", "")
                if prompt:
                    response_raw = self._submit_prompt(prompt)
                    try:
                        response_clean = self._clean_sample(response_raw)
                    except ValueError as e:
                        self.logger.error(f"Error cleaning sample for task #{task['custom_id']}: {e}. Using raw response.")
                        response_clean = {"raw": response_raw}

                    self.serialize_response(response_clean)
                    self.aggregate_results()
                    self.logger.info(f"Processed task #{task['custom_id']}")
                else:
                    self.logger.warning(f"No prompt found for task #{task['custom_id']}")

                if (i + 1) % 15 == 0 and (i + 1) < total_tasks_in_batch:
                    self.logger.info("Rate limit reached. Sleeping for 60 seconds.")
                    time.sleep(60)

                time.sleep(1)

            self._tasks.clear()