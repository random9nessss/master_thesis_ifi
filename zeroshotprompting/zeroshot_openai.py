import os
import re
import shutil
import httpx
import json
import openai
import pandas as pd
from dotenv import load_dotenv
from attributes.zero_shot_prompt import ZeroShotPrompt
from attributes.few_shot_prompt import FewShotPrompt

from config.logger import CustomLogger


class OpenAIZeroShotClient:

    def __init__(self, mode="zeroshot"):
        """
        Initialize an OpenAIClient instance.

        This method performs the following tasks:
          - Sets up a custom logger.
          - Loads the OpenAI API key from the ENV.txt file.
          - Initializes the OpenAI client with a custom HTTP client (with certificate verification disabled).
          - Creates directories for storing batch files if they do not already exist.
          - Initializes an internal task list and index counter.
          - Loads the zero-shot prompt module for generating prompts.

        Args:
            mode (str, optional): The operating mode, e.g. "zero_shot" or "fewshot". Defaults to "zero_shot".

        Raises:
            SystemExit: If the OPENAI_KEY is not found in the environment.
        """
        self.logger = CustomLogger(name="OpenAIZeroShotClient")
        self.logger.ok("OpenAIZeroShotClient initialized")

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
        self._directory = os.path.join(repo_root, "syntheticdata", self._mode, "gpt-4-turbo")
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

    def _append_custom_prompt_task(self, prompt: str, model: str = "gpt-4-turbo", temperature: float = 1.0):
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
        Serialize and store a batch job ID in a JSON file for later monitoring.

        This method appends the provided batch job ID to the active jobs file so that
        the job's status can be monitored over time.

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
            cleaned = cleaned[start:end+1]

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

    def run_batch_inference(self, batch_filename: str, number_samples: int = 10):
        """
         Run the full batch inference process.

         This method performs the following steps:
           1. Generates a specified number of prompt tasks.
           2. Creates a batch input file with all generated tasks.
           3. Submits the batch job to the OpenAI API.
           4. Clears the internal task list to prepare for future runs.

         Args:
             batch_filename (str): The name of the batch input file to be created.
             number_samples (int, optional): The number of prompt tasks to generate. Defaults to 10.

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
        # Creation of Batch File
        # ------------------------------------------------------------------
        file = self._create_batch_input_file(batch_filename)

        # ------------------------------------------------------------------
        # Starting Batch Job
        # ------------------------------------------------------------------
        self._submit_batch_job(file)

        # ------------------------------------------------------------------
        # Clearing Tasks
        # ------------------------------------------------------------------
        self._tasks.clear()