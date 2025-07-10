import os
import re
import json
import time
import httpx
import shutil
from dotenv import load_dotenv
from config.logger import CustomLogger
import pandas as pd

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

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


class AnthropicRefine:
    """
    Handles the refinement process using the Anthropic API.

    This class is responsible for constructing refinement tasks, submitting batch jobs,
    monitoring job statuses, retrieving and aggregating refined results, and organizing
    output files.
    """

    def __init__(self, model_name: str):
        """
        Initialize the AnthropicRefine client with the specified model name.

        Loads API keys from the environment, initializes the Anthropic API client, sets up
        required directories, and initializes internal task management counters.

        Args:
            model_name (str): The name of the model to be used (e.g., "claude").
        """
        self.logger = CustomLogger(name="AnthropicClient")
        self.logger.ok("AnthropicClient initialized")

        # ------------------------------------------------------------------
        # Loading API Key and Initializing API Client
        # ------------------------------------------------------------------
        load_dotenv("ENV.txt")
        if not os.getenv("ANTHROPIC_KEY"):
            self.logger.error("Anthropic API Key not found. Set ANTHROPIC_KEY in ENV.txt")
            exit(1)

        self._client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_KEY"),
            http_client=httpx.Client(verify=False)
        )

        # ------------------------------------------------------------------
        # Batch File Directory Setup
        # ------------------------------------------------------------------
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()

        repo_root = self.find_masterthesis_dev_path(current_dir)
        self._directory = os.path.join(repo_root, "syntheticdata", "baserefine", "refine", model_name, "claude")
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
                    "Could not locate a folder named 'Masterthesis-dev' above {}".format(start_path)
                )
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
        Construct the refinement prompt for the Anthropic API given a base email chain text.

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
               - `"broker"`,
               - `"commodity"`,
               - `"load_port"`,
               - `"discharge_port"`,
               - `"cargo_size"`,
               - `"incoterm"`,
               - `"vessel"`,
               - `"dwt"`,
               - `"loa"`,
               - `"starting_freight_quote_currency"`,
               - `"starting_freight_quote"`,
               - `"final_freight_quote_currency"`,
               - `"final_freight_quote"`,
               - `"laytime_start_date"`,
               - `"laytime_end_date"`,
               - `"demurrage_currency"`,
               - `"demurrage"`
              }}
            }}
            ```
        """

    def _append_refinement_task(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 1.0):
        """
        Append a custom prompt task to the internal tasks list.

        Constructs a Request object using the provided prompt and parameters, then increments the index.

        Parameters:
            prompt (str): The custom prompt content.
            model (str): The model to be used for the completion (default: "claude-3-5-sonnet-20241022").
            temperature (float): The sampling temperature for the model (default: 1.0).

        Returns:
            None
        """
        task = Request(
            custom_id=str(self._index),
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=5000,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt,
                }]
            )
        )
        self._increment_index()
        self._tasks.append(task)
        self.logger.info("Refinement task added.")

    def _submit_batch_job(self) -> None:
        """
        Submit the queued tasks as a batch job to the Anthropic API.

        This method performs the following steps:
          1. Creates a new batch job with the queued tasks.
          2. Serializes the tasks to a JSON Lines file.
          3. Serializes the batch job ID for future monitoring.

        Returns:
            None
        """
        # ------------------------------------------------------------------
        # Starting Batch Job
        # ------------------------------------------------------------------
        message_batch = self._client.messages.batches.create(
            requests=self._tasks
        )
        self.logger.ok(f"Uploaded batch job to Anthropic with ID {message_batch.id}")

        # ------------------------------------------------------------------
        # Input File Serialization
        # ------------------------------------------------------------------
        filepath = os.path.join(self._input_directory, f"{message_batch.id}.jsonl")
        self._serialize_tasks_to_file(filepath)
        self._serialize_batch_jobs(message_batch.id)

    def _serialize_tasks_to_file(self, filename: str = "batch_input.jsonl") -> str:
        """
        Serialize the queued tasks to a JSON Lines file.

        Each task is converted to a dictionary containing its custom_id and parameters,
        then written as a separate JSON object on a new line.

        Parameters:
            filename (str): The filename or absolute path to store the JSON Lines data.

        Returns:
            str: The filename used for serialization.
        """
        filepath = filename if os.path.isabs(filename) else os.path.join(self._input_directory, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for task in self._tasks:
                    if isinstance(task, dict):
                        custom_id = task.get("custom_id")
                        params = task.get("params", {})
                    else:
                        custom_id = task.custom_id
                        params = task.params

                    if isinstance(params, dict):
                        model = params.get("model")
                        max_tokens = params.get("max_tokens")
                        temperature = params.get("temperature")
                        messages = params.get("messages")
                    else:
                        model = params.model
                        max_tokens = params.max_tokens
                        temperature = params.temperature
                        messages = params.messages

                    task_dict = {
                        "custom_id": custom_id,
                        "params": {
                            "model": model,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "messages": messages
                        }
                    }
                    f.write(json.dumps(task_dict) + "\n")
            self.logger.info(f"Batch file created at {filepath}")
        except Exception as e:
            self.logger.error(f"Error serializing tasks to {filepath}: {str(e)}")
        return filename

    def _serialize_batch_jobs(self, batch_job_id: str) -> None:
        """
        Append the provided batch job ID to the active jobs JSON file for monitoring.

        Parameters:
            batch_job_id (str): The ID of the batch job to be serialized.

        Returns:
            None
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

    def _clean_sample(self, sample_str: str) -> dict:
        """
        Remove Markdown-style formatting from a sample string and parse it as JSON.

        Parameters:
            sample_str (str): The raw string containing JSON data with optional Markdown code fences.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If the string cannot be parsed as valid JSON.
        """
        cleaned = str(sample_str).strip()
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
        Aggregate results from the output directory.

        This method reads all JSON files from the output directory, cleans and merges their contents
        into a single aggregated JSON file, and then moves processed files to a designated directory.

        Returns:
            None
        """
        new_results = []

        aggregated_dir = os.path.join(self._directory, "aggregated")
        os.makedirs(aggregated_dir, exist_ok=True)

        processed_dir = os.path.join(self._directory, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Iterate over Files in Output Directory
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
            # Move File to Processed Directory
            # ------------------------------------------------------------------
            dest_path = os.path.join(processed_dir, filename)
            try:
                shutil.move(file_path, dest_path)
                self.logger.info(f"Moved {filename} to processed directory.")
            except Exception as e:
                self.logger.error(f"Error moving file {filename} to processed: {e}")

        # ------------------------------------------------------------------
        # Result Aggregation
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
        Monitor active batch jobs and process those that have completed.

        This method performs the following:
          - Loads the active batch jobs from a JSON file.
          - Checks each job's status via the Anthropic API.
          - For completed jobs, retrieves and aggregates the results, then updates the completed jobs file.
          - For jobs still in progress, retains them in the active jobs file for future monitoring.

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
                batch_job = self._client.messages.batches.retrieve(job_id)
            except Exception as e:
                self.logger.error(f"Error retrieving batch job {job_id}: {str(e)}")
                still_active_jobs.append(job_id)
                continue

            # ------------------------------------------------------------------
            # Result Retrieval if Job Ended
            # ------------------------------------------------------------------
            if batch_job.processing_status == "ended":
                self.logger.ok(f"Batch job {job_id} completed")
                result_file_name = self._retrieve_results(batch_job.id)
                self._aggregate_results()
                completed_jobs.append({
                    "batch_id": job_id,
                    "output": result_file_name
                })
            else:
                self.logger.info(f"Batch job {job_id} status: {batch_job.processing_status}")
                still_active_jobs.append(job_id)

        # ------------------------------------------------------------------
        # Active and Completed Batch Job File Update
        # ------------------------------------------------------------------
        with open(active_jobs_file, "w", encoding="utf-8") as f:
            json.dump(still_active_jobs, f, indent=2)
        with open(completed_jobs_file, "w", encoding="utf-8") as f:
            json.dump(completed_jobs, f, indent=2)

    def _retrieve_results(self, batch_job_id: str) -> str:
        """
        Retrieve and save the results of a completed batch job.

        This method streams results from the Anthropic API for the given batch job ID,
        processes each result using pattern matching to differentiate between succeeded, errored,
        and expired results, and saves the consolidated results to a JSON file.

        Parameters:
            batch_job_id (str): The ID of the batch job to retrieve results for.

        Returns:
            str: The basename of the JSON file containing the consolidated results.
        """
        results_list = []
        try:

            # ------------------------------------------------------------------
            # Result Streaming from Anthropic API
            # ------------------------------------------------------------------
            result_stream = self._client.messages.batches.results(batch_job_id)
            for result in result_stream:
                match result.result.type:
                    case "succeeded":
                        content_parts = result.result.message.content
                        content = " ".join(part.text for part in content_parts)
                        results_list.append(content)
                    case "errored":
                        if result.result.error.type == "invalid_request":
                            self.logger.error(f"Validation error for task {result.custom_id}")
                        else:
                            self.logger.error(f"Server error for task {result.custom_id}")
                        results_list.append({
                            "custom_id": result.custom_id,
                            "content": f"Error: {result.result.error.type}",
                            "result_type": result.result.type
                        })
                    case "expired":
                        self.logger.info(f"Request expired for task {result.custom_id}")
                        results_list.append({
                            "custom_id": result.custom_id,
                            "content": "Request expired",
                            "result_type": result.result.type
                        })
            results_filename = os.path.join(self._output_directory, f"{batch_job_id}.json")
            with open(results_filename, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2)
            self.logger.ok(f"Results for batch job {batch_job_id} saved to {results_filename}")
            return os.path.basename(results_filename)
        except Exception as e:
            self.logger.error(f"Error retrieving results for batch job {batch_job_id}: {str(e)}")
            return ""

    def run_refinement(self, base_file_path: str, batch_filename_prefix: str, number_samples: int = None):
        """
        Run the full batch refinement process by processing the base JSON Lines file in batches.

        This method performs the following steps:
          1. Loads the base dataset (JSON Lines format) into a DataFrame.
          2. Optionally limits the total number of samples (from the tail of the DataFrame).
          3. Processes the DataFrame in batches of 50 rows:
             - For each row with a valid 'chain' field, constructs a refinement prompt and appends a task.
             - Submits the queued tasks as a batch job to the Anthropic API.
             - Clears the internal task list.
             - Waits for the active batch job(s) to complete before processing the next batch.

        Parameters:
            base_file_path (str): Path to the input dataset file (JSON Lines format).
            batch_filename_prefix (str): A prefix for naming batch-related files.
            number_samples (int, optional): If provided, only the last 'number_samples' rows will be processed.

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
                self._append_refinement_task(prompt)

            # ------------------------------------------------------------------
            # Submission of Batch Job
            # ------------------------------------------------------------------
            self._submit_batch_job()
            self._tasks.clear()
            self.logger.ok(f"Submitted batch {(batch_start // batch_size) + 1} "
                           f"(rows {batch_start} to {min(batch_start + batch_size - 1, len(df_base) - 1)})")

            wait_for_all_jobs(self)