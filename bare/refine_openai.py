import os
import json
import re
import shutil
import time
import httpx
import openai
import pandas as pd
from dotenv import load_dotenv
from config.logger import CustomLogger

logger = CustomLogger("Client Monitoring")


def wait_for_all_jobs(client):
    """
    Wait for all active batch jobs to complete using an exponential backoff strategy.

    This function continuously monitors the status of active batch jobs by invoking
    the client's `monitor_batch_jobs()` method and checking the contents of the
    'active_batch_jobs.json' file located in the client's directory. It employs an
    exponential backoff approach—starting with a wait time of 200 seconds and doubling
    the wait time (up to a maximum of 2000 seconds) after each check—until no active jobs
    remain.

    Args:
        client: An object representing the client interfacing with the batch job API.
            It must have the following attributes:
                - _directory (str): The directory where batch job files (e.g.,
                  'active_batch_jobs.json') are stored.
                - monitor_batch_jobs() (callable): A method that updates the status of batch jobs.
                - logger: A logger instance with an `info` method for logging messages.

    Returns:
        None
    """
    wait_time = 200
    while True:
        client.monitor_batch_jobs()
        active_jobs_file = os.path.join(client._directory, "active_batch_jobs.json")
        active_jobs = []
        if os.path.exists(active_jobs_file):
            try:
                with open(active_jobs_file, "r", encoding="utf-8") as f:
                    active_jobs = json.load(f)
            except json.JSONDecodeError:
                active_jobs = []
        if not active_jobs:
            break

        logger.info(f"Active jobs still running: {active_jobs}. Waiting for {wait_time} seconds...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 2000)


class OpenAIRefine:

    def __init__(self, model_name: str):
        self.logger = CustomLogger(name="OpenAIClient")
        self.logger.ok("OpenAIClient initialized")

        # ------------------------------------------------------------------
        # Loading API Key and Initializing API Client
        # ------------------------------------------------------------------
        load_dotenv("ENV.txt")
        openai.api_key = os.getenv("OPENAI_KEY")

        if not openai.api_key:
            self.logger.error("OpenAI API Key not found. Set OPENAI_KEY in ENV.txt")
            exit(1)

        self._client = openai.OpenAI(api_key=openai.api_key, http_client=httpx.Client(verify=False))

        # ------------------------------------------------------------------
        # Batch File Directory Setup
        # ------------------------------------------------------------------

        # .py files
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        # .ipynb files
        except NameError:
            current_dir = os.getcwd()

        repo_root = self.find_masterthesis_dev_path(current_dir)
        self._directory = os.path.join(os.getcwd(), "syntheticdata", "baserefine", "refine", model_name, "gpt-4-turbo")
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

    def find_masterthesis_dev_path(self, start_path=None):
        """
        Traverse upward from start_path until a directory named 'Masterthesis-dev' is found.
        Returns the absolute path to 'Masterthesis-dev' if found, otherwise raises an error.
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

    def _retrieve_refinement_prompt(self, base_text: str) -> str:
        """
        Constructs the refinement prompt for OpenAI's API given a base email chain text.

        Args:
            base_text (str): The unrefined email chain text.

        Returns:
            str: A formatted prompt instructing the model on how to refine the chain,
                 including JSON formatting requirements.
        """
        return f"""
            Please refine the following email chain to make it coherent and realistic without changing the theme, 
            then convert it to well-formed JSON as specified.

            ---
            Unrefined email chain:
            {base_text}
            ---

            Formatting requirements:
            1. Follow the JSON structure below exactly.
            2. Make sure each email in the chain is refined, logically consistent, and preserves the same meaning.
            3. Include a `labels` object at the end with as many fields filled in as can reasonably be inferred. If uncertain, leave blank.
            4. Maintain chronological sequence of emails. If needed, add or estimate a timestamp in the format YYYY-MM-DD HH:MM to each email.
            5. Use realistic email addresses derived from the names given in the base text.
            6. Do not invent extra content not implied by the base text.

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
              }}
            }}
        """

    def _append_refinement_task(self, prompt: str, model: str = "gpt-4-turbo", temperature: float = 1.0):
        """
        Append a custom prompt task to the internal tasks list for the OpenAI batch API.
        """
        task = {
            "custom_id": str(self._index),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
            }
        }

        self._increment_index()
        self._tasks.append(task)
        self.logger.info("Refinement task added with custom prompt.")

    def _create_batch_input_file(self, filename: str = "batch_input.jsonl") -> str:
        """
        Create a batch input file in JSON Lines format containing all generated tasks.

        Each task is serialized as a JSON object on a separate line. The file is saved
        in the designated input subdirectory.

        Parameters:
            filename (str): The name of the batch input file (default is "batch_input.jsonl").

        Returns:
            str: The filename of the created batch input file.
        """
        filepath = os.path.join(self._input_directory, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for item in self._tasks:
                f.write(json.dumps(item) + "\n")

        self.logger.info(f"Batch file created at {filepath}")
        return filename

    def _serialize_batch_jobs(self, batch_job_id: str) -> None:
        """
        Appends the provided batch job ID to a tracking file for active jobs.

        Args:
            batch_job_id (str): The batch job identifier to serialize.
        """
        active_jobs_file = os.path.join(self._directory, "active_batch_jobs.json")

        if os.path.exists(active_jobs_file):
            try:
                with open(active_jobs_file, "r", encoding="utf-8") as f:
                    active_jobs = json.load(f)
            except json.JSONDecodeError:
                active_jobs = []
        else:
            active_jobs = []

        active_jobs.append(batch_job_id)

        with open(active_jobs_file, "w", encoding="utf-8") as f:
            json.dump(active_jobs, f, indent=2)

        self.logger.info(f"Serialized batch job id {batch_job_id} to {active_jobs_file}")

    def _submit_batch_job(self, filename: str) -> None:
        """
        Submit a batch job to the OpenAI API using the specified batch input file.

        This method uploads the batch file to OpenAI, creates a new batch job, renames
        the input file using the batch job ID, and then serializes the batch job ID for future monitoring.

        Parameters:
            filename (str): The name of the batch input file (located in the input directory) to be submitted.

        Returns:
            None
        """
        input_filepath = os.path.join(self._input_directory, filename)

        # ------------------------------------------------------------------
        # Upload Batch File
        # ------------------------------------------------------------------
        with open(input_filepath, "rb") as infile:
            batch_file = self._client.files.create(
                file=infile,
                purpose="batch"
            )

        self.logger.ok("Uploaded batch job to OpenAI")

        # ------------------------------------------------------------------
        # Start Batch Job
        # ------------------------------------------------------------------
        batch_job = self._client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        self.logger.ok(f"Batch job for file_id {batch_file.id} started with batch job id {batch_job.id}")

        # ------------------------------------------------------------------
        # Rename Batch Job File
        # ------------------------------------------------------------------
        new_filename = f"{batch_job.id}.jsonl"
        new_filepath = os.path.join(self._input_directory, new_filename)
        os.rename(input_filepath, new_filepath)
        self.logger.info(f"Renamed input file to {new_filepath}")

        self._serialize_batch_jobs(batch_job.id)

    def _clean_sample(self, sample_str: str) -> dict:
        """
        Strips away Markdown-style formatting (e.g., ```json) and loads the remaining string as JSON.

        Args:
            sample_str (str): The raw string containing JSON data with optional Markdown fences.

        Returns:
            dict: A dictionary parsed from the cleaned JSON string.

        Raises:
            ValueError: If the JSON string cannot be parsed.
        """
        cleaned = sample_str.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

        else:
            self.logger.warning("No valid JSON object found in sample.")

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON from sample: {e}") from e

    def _aggregate_results(self) -> None:
        """
        Aggregates results from the output directory, standardizes them by cleaning markdown formatting,
        merges them into a single JSON file (appending to existing records if any), and moves the processed files
        to a processed directory.
        """
        new_results = []

        # ------------------------------------------------------------------
        # Aggregation Directory
        # ------------------------------------------------------------------
        aggregated_dir = os.path.join(self._directory, "aggregated")
        os.makedirs(aggregated_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Processed Directory
        # ------------------------------------------------------------------
        processed_dir = os.path.join(self._directory, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Iteration over Output Directory
        # ------------------------------------------------------------------
        for filename in os.listdir(self._output_directory):
            file_path = os.path.join(self._output_directory, filename)

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
                try:
                    cleaned_sample = self._clean_sample(sample)
                    new_results.append(cleaned_sample)

                except Exception as e:
                    self.logger.error(f"Error cleaning sample from {filename}: {e}")
                    continue

            # ------------------------------------------------------------------
            # Moving Files to Processed Directory
            # ------------------------------------------------------------------
            dest_path = os.path.join(processed_dir, filename)
            try:
                shutil.move(file_path, dest_path)
                self.logger.info(f"Moved {filename} to processed directory.")

            except Exception as e:
                self.logger.error(f"Error moving file {filename} to processed: {e}")

        # ------------------------------------------------------------------
        # Append to Existing Aggregated JSON File if Present
        # ------------------------------------------------------------------
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
            self.logger.ok(f"Aggregated results saved to {aggregated_file}")

        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")

    def monitor_batch_jobs(self) -> None:
        """
        Monitor active batch jobs and process completed ones.

        This method checks the status of each active batch job by reading the active jobs file.
        For each job:
          - If the job is completed, its results are retrieved and saved, and the job ID is moved
            to the completed jobs file.
          - If the job is still in progress, its ID remains in the active jobs file for future monitoring.

        Returns:
            None
        """
        active_jobs_file = os.path.join(self._directory, "active_batch_jobs.json")
        completed_jobs_file = os.path.join(self._directory, "completed_batch_jobs.json")

        # ------------------------------------------------------------------
        # Loading Active Batch Jobs
        # ------------------------------------------------------------------
        if os.path.exists(active_jobs_file):
            try:
                with open(active_jobs_file, "r", encoding="utf-8") as f:
                    active_jobs = json.load(f)
            except json.JSONDecodeError:
                active_jobs = []
        else:
            self.logger.info("No active batch jobs to monitor.")
            return

        # ------------------------------------------------------------------
        # Loading Completed Batch Jobs
        # ------------------------------------------------------------------
        if os.path.exists(completed_jobs_file):
            try:
                with open(completed_jobs_file, "r", encoding="utf-8") as f:
                    completed_jobs = json.load(f)
            except json.JSONDecodeError:
                completed_jobs = []
        else:
            completed_jobs = []

        still_active_jobs = []
        for job_id in active_jobs:
            try:
                batch_job = self._client.batches.retrieve(job_id)
            except Exception as e:
                self.logger.error(f"Error retrieving batch job {job_id}: {str(e)}")
                still_active_jobs.append(job_id)
                continue

            # ------------------------------------------------------------------
            # Result Retrieval
            # ------------------------------------------------------------------
            if batch_job.status == "completed":
                self.logger.ok(f"Batch job {job_id} completed")
                result_file_name = self._retrieve_results(batch_job.output_file_id)
                self._aggregate_results()

                completed_jobs.append({
                    "batch_id": job_id,
                    "output": result_file_name
                })
            else:
                self.logger.info(f"Batch job {job_id} status: {batch_job.status}")
                still_active_jobs.append(job_id)

        # ------------------------------------------------------------------
        # Update of Active and Completed Batch Jobs
        # ------------------------------------------------------------------
        with open(active_jobs_file, "w", encoding="utf-8") as f:
            json.dump(still_active_jobs, f, indent=2)

        with open(completed_jobs_file, "w", encoding="utf-8") as f:
            json.dump(completed_jobs, f, indent=2)

    def _retrieve_results(self, batch_job_result_id: str) -> str:
        """
        Retrieve and save the results of a completed batch job.

        This method downloads the output file associated with the given batch job result ID,
        decodes and parses its contents (assumed to be in JSON Lines format), and saves the
        consolidated results to a JSON file in the output subdirectory.

        Parameters:
            batch_job_result_id (str): The ID of the output file containing the batch job results.

        Returns:
            str: The filename of the result file.
        """
        results_list = []
        try:
            result_file = self._client.files.content(batch_job_result_id)
            file_content = result_file.content
            if isinstance(file_content, bytes):
                file_content = file_content.decode("utf-8")

            for line in file_content.strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    try:
                        if "response" in data and "body" in data["response"]:
                            content = data["response"]["body"]["choices"][0]["message"]["content"]
                        else:
                            content = data["choices"][0]["message"]["content"]

                        content = content.strip()

                        if content.startswith("[") and content.endswith("]"):
                            try:
                                conversation = json.loads(content)
                                if isinstance(conversation, list):
                                    results_list.extend(conversation)
                                else:
                                    results_list.append(conversation)
                            except Exception as e:
                                self.logger.error(f"Error parsing conversation JSON: {str(e)}. Using raw content.")
                                results_list.append(content)
                        else:
                            results_list.append(content)
                    except (KeyError, IndexError) as e:
                        self.logger.error(f"Could not extract content from line: {line} - {str(e)}")

            results_filename = os.path.join(self._output_directory, f"{batch_job_result_id}.json")

            with open(results_filename, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2)

            self.logger.ok(f"Results for batch job {batch_job_result_id} saved to {results_filename}")
            return os.path.basename(results_filename)

        except Exception as e:
            self.logger.error(f"Error retrieving results for batch job {batch_job_result_id}: {str(e)}")
            return ""

    def run_batch_refinement(self, base_file_path: str, batch_filename_prefix: str, number_samples: int = None):
        """
        Run the full batch inference process by processing the base JSON Lines file in batches.

        This method performs the following steps:
          1. Reads the base file containing email chain samples.
          2. Optionally limits the total number of samples to process.
          3. Splits the samples into batches of 50.
          4. For each batch:
             - Iterates over the rows.
             - For each row with a valid "chain" field, generates a refinement prompt and appends a task.
             - Creates a batch input file with all the tasks.
             - Submits the batch job to the OpenAI API.
             - Clears the internal task list for the next batch.

        Args:
            base_file_path (str): Path to the JSON Lines file with base email chains.
            batch_filename_prefix (str): A prefix to use for naming each batch input file.
            number_samples (int, optional): Limit the number of samples processed from the file.
                                            If None, all samples are processed.

        Returns:
            None
        """
        df_base = pd.read_json(base_file_path, lines=True)

        if number_samples is not None and number_samples < len(df_base):
            df_base = df_base.tail(number_samples)

        batch_size = 50

        # ------------------------------------------------------------------
        # Processing the DataFrame in Batches of 50 rows
        # ------------------------------------------------------------------
        for batch_start in range(0, len(df_base), batch_size):
            df_batch = df_base.iloc[batch_start:batch_start + batch_size]

            for idx, row in df_batch.iterrows():
                base_chain = row.get("chain", "")
                if not base_chain:
                    self.logger.warning(f"Row {idx} doesn't have a 'chain' field. Skipping.")
                    continue

                prompt = self._retrieve_refinement_prompt(base_chain)
                self._append_refinement_task(prompt, model="gpt-4-turbo", temperature=1.0)

            batch_filename = f"{batch_filename_prefix}_{batch_start}.jsonl"

            # ------------------------------------------------------------------
            # Batch File Creation
            # ------------------------------------------------------------------
            self._create_batch_input_file(batch_filename)
            self._submit_batch_job(batch_filename)
            self._tasks.clear()

            self.logger.ok(f"Submitted batch {(batch_start // batch_size) + 1} "
                           f"(rows {batch_start} to {min(batch_start + batch_size - 1, len(df_base) - 1)})")

            wait_for_all_jobs(self)