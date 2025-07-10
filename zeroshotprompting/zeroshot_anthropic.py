import os
import re
import json
import httpx
import shutil
from dotenv import load_dotenv
from config.logger import CustomLogger
from attributes.zero_shot_prompt import ZeroShotPrompt
from attributes.few_shot_prompt import FewShotPrompt
import pandas as pd

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


class AnthropicZeroShotClient:

    def __init__(self, mode="zeroshot"):

        assert mode in ["zeroshot", "fewshot"], f"Mode provided <{mode}> is invalid"

        self.logger = CustomLogger(name="AnthropicZeroShotClient")
        self.logger.ok("AnthropicZeroShotClient initialized")

        # ------------------------------------------------------------------
        # Mode Loading
        # ------------------------------------------------------------------
        self._mode = mode
        self._zero_shot_module = ZeroShotPrompt()
        self._few_shot_module =  FewShotPrompt()

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

        # .py files
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        # .ipynb files
        except NameError:
            current_dir = os.getcwd()

        repo_root = self.find_masterthesis_dev_path(current_dir)
        self._directory = os.path.join(repo_root, "syntheticdata", self._mode, "claude")
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

    def _append_custom_prompt_task(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 1.0):
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
        self.logger.info("Task added with custom prompt.")

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
        # Start Batch Job
        # ------------------------------------------------------------------

        message_batch = self._client.messages.batches.create(
            requests=self._tasks
        )

        self.logger.ok(f"Uploaded batch job to Anthropic {message_batch.id}")

        # ------------------------------------------------------------------
        # Serialize task file to json
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
            cleaned = cleaned[start:end+1]

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
            # Result Retrieval
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
        # Update of Active and Completed Batch Jobs
        # ------------------------------------------------------------------
        with open(active_jobs_file, "w", encoding="utf-8") as f:
            json.dump(still_active_jobs, f, indent=2)

        with open(completed_jobs_file, "w", encoding="utf-8") as f:
            json.dump(completed_jobs, f, indent=2)

    def _retrieve_results(self, batch_job_id: str) -> str:
        """
        Retrieve and save the results of a completed batch job.

        This method streams results from the Anthropic API for the given batch job ID,
        processes each result using match-case to differentiate between succeeded, errored, and expired results,
        and saves the consolidated results to a JSON file.

        Parameters:
            batch_job_id (str): The ID of the batch job to retrieve results for.

        Returns:
            str: The basename of the JSON file containing the consolidated results.
        """
        results_list = []
        try:

            # ------------------------------------------------------------------
            # Result Streaming
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
                            self.logger.error(f"Validation error {result.custom_id}")
                        else:
                            self.logger.error(f"Server error {result.custom_id}")
                        results_list.append({
                            "custom_id": result.custom_id,
                            "content": f"Error: {result.result.error.type}",
                            "result_type": result.result.type
                        })

                    case "expired":
                        self.logger.info(f"Request expired {result.custom_id}")
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

    def run_batch_inference(self, number_samples: int = 10):
        """
        Run the full batch inference process.

        This method performs the following steps:
          1. Generates a specified number of prompt tasks.
          2. Submits the queued tasks as a batch job to the Anthropic API.
          3. Clears the internal task list for future runs.

        Parameters:
            number_samples (int): The number of prompt tasks to generate (default: 10).

        Returns:
            None
        """
        # ------------------------------------------------------------------
        # Creation of Tasks
        # ------------------------------------------------------------------
        for i in range(number_samples):

            if self._mode == "zeroshot":
                self._append_custom_prompt_task(prompt=self._zero_shot_module._prompt)

            elif self._mode == "fewshot":
                self._append_custom_prompt_task(prompt=self._few_shot_module.generate_few_shot_prompt())

            else:
                raise NotImplemented(f"Currently only <zeroshot> and <fewshot> are supported.")

        # ------------------------------------------------------------------
        # Starting Batch Job
        # ------------------------------------------------------------------
        self._submit_batch_job()

        # ------------------------------------------------------------------
        # Clearing Tasks
        # ------------------------------------------------------------------
        self._tasks.clear()