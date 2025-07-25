import os
import json
import re
import shutil
import time
import httpx
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from config.logger import CustomLogger


class MistralRefine:
    """
    Handles the refinement process of synthetic email chains using Mistral's API.

    This version reads a JSONL file containing base chains (with a "chain" field),
    constructs a detailed refinement prompt for each entry, and submits them in batch.
    """

    def __init__(self, model_name : str = "llama8b"):
        """
        Initializes the MistralRefine processor by setting up directories, loading the API key,
        and initializing the Mistral client along with internal task management.

        Raises:
            SystemExit: If the MISTRAL_KEY environment variable is not set.
        """
        self.logger = CustomLogger(name="MistralRefine")
        self.logger.ok("MistralRefine initialized")

        # -------------------------------------------------------------------
        # Load Environment Variables and Initialize Mistral Client
        # -------------------------------------------------------------------
        load_dotenv("ENV.txt")
        if not os.getenv("MISTRAL_KEY"):
            self.logger.error("MISTRAL API Key not found. Set MISTRAL_KEY in ENV.txt")
            exit(1)

        self._client = Mistral(api_key=os.getenv("MISTRAL_KEY"), client=httpx.Client(verify=False))

        # -------------------------------------------------------------------
        # Directory Setup
        # -------------------------------------------------------------------
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()

        repo_root = self.find_masterthesis_dev_path(current_dir)
        self._directory = os.path.join(repo_root, "syntheticdata", "mistral")
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
        self._index = 0

    def find_masterthesis_dev_path(self, start_path=None):
        """
        Traverse upward from the given start path until a directory named 'Masterthesis-dev' is found.

        Args:
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
        Constructs the refinement prompt for Mistral's API given a base email chain text.

        Args:
            base_text (str): The unrefined email chain text.

        Returns:
            str: The formatted refinement prompt.
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

    def create_refinement_tasks_from_jsonl(self, input_jsonl_file: str, model: str = "mistral-large-latest", temperature: float = 1.0):
        """
        Reads a JSON Lines file and constructs a refinement task for each entry.

        Expects each line to be a JSON object with a "chain" field.

        Args:
            input_jsonl_file (str): Path to the JSONL file containing base email chains.
            model (str, optional): The model identifier to use (not utilized in this function).
            temperature (float, optional): Sampling temperature (not utilized in this function).

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
        Appends a custom prompt task to the internal tasks list for Mistral's batch API.

        Args:
            prompt (str): The refinement prompt to be added as a task.

        Returns:
            None
        """
        task = {
            "custom_id": str(self._get_index()),
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        }
        self._increment_index()
        self._tasks.append(task)
        self.logger.info(f"Task #{self._index} added with custom prompt.")

    def _create_batch_input_file(self, filename: str = "batch_input.jsonl") -> str:
        """
        Creates a JSON Lines file containing all generated tasks.

        Args:
            filename (str, optional): The name of the file to be created in the input directory.

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
            batch_job_id (str): The identifier of the batch job to be tracked.

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
        Submits the batch input file to Mistral's Batch API.

        This method uploads the file, creates a batch job using Mistral's API,
        renames the input file with the job ID, and tracks the batch job ID.

        Args:
            filename (str): The name of the batch input file to be submitted.

        Returns:
            None
        """
        input_filepath = os.path.join(self._input_directory, filename)
        with open(input_filepath, "rb") as f:
            batch_data = self._client.files.upload(
                file={
                    "file_name": filename,
                    "content": f
                },
                purpose="batch"
            )
        self.logger.ok("Uploaded batch job to MistralAI")
        created_job = self._client.batch.jobs.create(
            input_files=[batch_data.id],
            model="mistral-large-latest",
            endpoint="/v1/chat/completions"
        )
        self.logger.ok(f"Batch job started with ID: {created_job.id}")
        new_filename = f"{created_job.id}.jsonl"
        new_filepath = os.path.join(self._input_directory, new_filename)
        time.sleep(5)
        os.rename(input_filepath, new_filepath)
        self.logger.info(f"Renamed input file to {new_filepath}")
        self._serialize_batch_jobs(created_job.id)

    def monitor_batch_jobs(self) -> None:
        """
        Monitors active batch jobs and processes completed ones.

        For each active job, if the job is completed, the results are retrieved and aggregated;
        otherwise, the job remains in the active tracking file.

        Returns:
            None
        """
        active_jobs_file = os.path.join(self._directory, "active_batch_jobs.json")
        completed_jobs_file = os.path.join(self._directory, "completed_batch_jobs.json")

        if os.path.exists(active_jobs_file):
            try:
                with open(active_jobs_file, "r", encoding="utf-8") as f:
                    active_jobs = json.load(f)
            except json.JSONDecodeError:
                active_jobs = []
        else:
            self.logger.info("No active batch jobs to monitor.")
            return

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
                batch_job = self._client.batch.jobs.get(job_id=job_id)
            except Exception as e:
                self.logger.error(f"Error retrieving batch job {job_id}: {str(e)}")
                still_active_jobs.append(job_id)
                continue

            if batch_job.status == "SUCCESS":
                self.logger.ok(f"Batch job {job_id} completed")
                result_file_name = self._retrieve_results(batch_job.output_file)
                self._aggregate_results()
                completed_jobs.append({"batch_id": job_id, "output": result_file_name})
            else:
                self.logger.info(f"Batch job {job_id} status: {batch_job.status}")
                still_active_jobs.append(job_id)

        with open(active_jobs_file, "w", encoding="utf-8") as f:
            json.dump(still_active_jobs, f, indent=2)
        with open(completed_jobs_file, "w", encoding="utf-8") as f:
            json.dump(completed_jobs, f, indent=2)

    def _retrieve_results(self, batch_job_result_id: str) -> str:
        """
        Retrieves and saves the results of a completed batch job from Mistral's API.

        The results are processed from a JSON Lines file and consolidated into a single JSON file.

        Args:
            batch_job_result_id (str): The identifier of the output file containing the batch job results.

        Returns:
            str: The filename of the saved results file. Returns an empty string if retrieval fails.
        """
        results_list = []
        try:
            result_file = self._client.files.download(file_id=batch_job_result_id)
            temp_file_path = os.path.join(self._output_directory, "temp.jsonl")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                for chunk in result_file.stream:
                    f.write(chunk.decode("utf-8"))
            with open(temp_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
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
                        self.logger.error(f"Could not extract content: {e}")
            results_filename = os.path.join(self._output_directory, f"{batch_job_result_id}.json")
            with open(results_filename, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2)
            self.logger.ok(f"Results for batch job {batch_job_result_id} saved to {results_filename}")
            return os.path.basename(results_filename)
        except Exception as e:
            self.logger.error(f"Error retrieving results for batch job {batch_job_result_id}: {str(e)}")
            return ""

    def _aggregate_results(self) -> None:
        """
        Aggregates refined results from the output directory into a consolidated JSON file.

        Processed output files are moved to a separate directory after aggregation.

        Returns:
            None
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
            self.logger.ok(f"Aggregated refined results saved to {aggregated_file}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated refined results: {e}")

    def _clean_sample(self, sample_str: str) -> dict:
        """
        Cleans a sample string by removing markdown formatting and parsing the JSON.

        Args:
            sample_str (str): The raw string containing the JSON data with optional markdown fences.

        Returns:
            dict: The parsed JSON object if successful, otherwise None.
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
        except json.JSONDecodeError:
            self.logger.warning(f"Could not parse JSON from sample: {cleaned[:20]}...")
            return None

    def run_batch_inference(self, input_jsonl_file: str, batch_filename: str):
        """
        Runs the full batch inference process using Mistral's API:
          1. Reads the JSONL file and creates a refinement task for each "chain" entry.
          2. Creates a batch input file from the accumulated tasks.
          3. Submits the batch job.
          4. Clears the internal task list for future runs.

        Args:
            input_jsonl_file (str): The path to the JSONL file containing base email chains.
            batch_filename (str): The name of the batch input file to be created.

        Returns:
            None
        """
        self.create_refinement_tasks_from_jsonl(input_jsonl_file)
        file = self._create_batch_input_file(batch_filename)
        self._submit_batch_job(file)
        self._tasks.clear()