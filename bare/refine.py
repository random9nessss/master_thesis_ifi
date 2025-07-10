import os
import json
import re
import shutil
import httpx
import openai
import pandas as pd
from dotenv import load_dotenv
from config.logger import CustomLogger


class Refine:
    """
    Handles the refinement process of synthetic email chains.

    Responsibilities include:
      - Reading synthetic email chains from a JSON Lines file.
      - Constructing refinement prompts and submitting them to OpenAI's Batch API.
      - Monitoring job statuses, retrieving and aggregating refined results.
      - Organizing output files into designated directories.
    """

    def __init__(self):
        """
         Initializes the Refine processor by setting up directories, the OpenAI client,
         and internal counters for task management.
         """
        self.logger = CustomLogger(name="RefineGenerator")

        # -------------------------------------------------------------------
        # Load Environment Variables
        # -------------------------------------------------------------------
        load_dotenv("ENV.txt")
        openai.api_key = os.getenv("OPENAI_KEY")

        if not openai.api_key:
            self.logger.error("OpenAI API Key not found. Set OPENAI_KEY in ENV.txt.")
            exit(1)

        # -------------------------------------------------------------------
        # OpenAI Client
        # -------------------------------------------------------------------
        self._client = openai.OpenAI(
            api_key=openai.api_key,
            http_client=httpx.Client(verify=False)
        )

        # -------------------------------------------------------------------
        # Directory Setup
        # -------------------------------------------------------------------
        self._batch_dir = os.path.join(os.getcwd(), "syntheticdata", "baserefine")
        self._input_dir = os.path.join(self._batch_dir, "input")
        self._input_base_dir = os.path.join(self._batch_dir, "input_base")
        self._output_dir = os.path.join(self._batch_dir, "output")
        self._processed_dir = os.path.join(self._batch_dir, "processed")
        self._aggregated_dir = os.path.join(self._batch_dir, "aggregated")

        try:
            os.makedirs(self._input_base_dir, exist_ok=True)
            self.logger.info(f"Directory {self._input_base_dir} ready")

            os.makedirs(self._input_dir, exist_ok=True)
            self.logger.info(f"Directory {self._input_dir} ready")

            os.makedirs(self._output_dir, exist_ok=True)
            self.logger.info(f"Directory {self._output_dir} ready")

            os.makedirs(self._processed_dir, exist_ok=True)
            self.logger.info(f"Directory {self._processed_dir} ready")

            os.makedirs(self._aggregated_dir, exist_ok=True)
            self.logger.info(f"Directory {self._aggregated_dir} ready")

        except OSError as e:
            self.logger.error(f"Error creating directories: {str(e)}")

        # ------------------------------------------------------------------
        # Task and Index Setup
        # ------------------------------------------------------------------
        self._tasks = []
        self._index = 0

        self.logger.ok("Refine generator initialized.")

    def _increment_index(self):
        """
        Increments the internal task index counter by 1.
        """
        self._index += 1

    def _get_index(self):
        """
        Retrieves the current value of the internal task index counter.

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

    def create_refinement_tasks_from_jsonl(self,
                                           input_jsonl_file: str,
                                           model: str = "gpt-4-turbo",
                                           temperature: float = 1.0):
        """
        Reads synthetic email chains from a JSON Lines file and constructs a refinement task
        for each valid chain.

        For each line in the file, a refinement prompt is created and appended to the internal
        task list for batch processing.

        Args:
            input_jsonl_file (str): Path to the JSON Lines file containing base email chains.
            model (str): The model identifier to use for refinement (default "gpt-4-turbo").
            temperature (float): Sampling temperature for generation. Defaults to 1.0.
        """
        if not os.path.exists(input_jsonl_file):
            self.logger.error(f"Input file does not exist: {input_jsonl_file}")
            return

        with open(input_jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ------------------------------------------------------------------
        # Loading Base File
        # ------------------------------------------------------------------
        self.logger.info(f"Loaded {len(lines)} lines from '{input_jsonl_file}' for refinement.")
        for i, line in enumerate(lines, start=1):
            try:
                data = json.loads(line.strip())
                chain_text = data.get("chain", "")
                if not chain_text:
                    self.logger.warning(f"No chain text found for line {i}. Skipping.")
                    continue

                # ------------------------------------------------------------------
                # Generating Refine File
                # ------------------------------------------------------------------
                prompt = self._retrieve_refinement_prompt(chain_text)
                self.append_custom_prompt_task(prompt=prompt, model=model, temperature=temperature)
                self.logger.info(f"Refinement prompt for chain {i} queued.")

            except Exception as e:
                self.logger.error(f"Error parsing line {i}: {e}")

    def append_custom_prompt_task(self, prompt: str,
                                  model: str = "gpt-4-turbo",
                                  temperature: float = 1.0):
        """
         Appends a custom prompt task to the internal task list for later batch submission.

         The task includes details such as HTTP method, API endpoint, and message payload.

         Args:
             prompt (str): The custom prompt to be used for refinement.
             model (str): The model identifier to use. Defaults to "gpt-4-turbo".
             temperature (float): Sampling temperature. Defaults to 1.0.
         """
        task = {
            "custom_id": str(self._get_index()),
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
        self.logger.info(f"Task #{self._index} added with custom prompt.")

    def create_and_submit_batch(self, output_batch_filename: str = "refinement_batch.jsonl"):
        """
        Creates a batch input file from the current tasks and submits it to OpenAI's Batch API.

        After submission, the internal task list is cleared.

        Args:
            output_batch_filename (str): The filename for the batch input file. Defaults to "refinement_batch.jsonl".
        """
        input_file_created = self._create_batch_input_file(output_batch_filename)
        self._submit_batch_job(input_file_created)
        self._tasks.clear()
        self.logger.ok("All refinement tasks submitted. Task list cleared.")

    def _create_batch_input_file(self, filename: str) -> str:
        """
        Writes the internal task list to a JSON Lines file in the input directory.

        Args:
            filename (str): The name of the file to create.

        Returns:
            str: The filename of the created batch input file.
        """
        filepath = os.path.join(self._input_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in self._tasks:
                f.write(json.dumps(item) + "\n")

        self.logger.info(f"Batch file created at {filepath}")
        return filename

    def _submit_batch_job(self, filename: str) -> None:
        """
        Submits the batch input file to OpenAI by uploading it, starting the batch job,
        renaming the file with the job ID, and serializing the job ID for tracking.

        Args:
            filename (str): The name of the batch input file to submit.
        """
        input_filepath = os.path.join(self._input_dir, filename)

        with open(input_filepath, "rb") as infile:
            batch_file = self._client.files.create(
                file=infile,
                purpose="batch"
            )

        # ------------------------------------------------------------------
        # Invoking Batch-Job
        # ------------------------------------------------------------------
        self.logger.ok("Uploaded batch job to OpenAI.")
        batch_job = self._client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        self.logger.ok(f"Batch job started with ID: {batch_job.id}")

        # ------------------------------------------------------------------
        # Renaming with JobID
        # ------------------------------------------------------------------
        new_filename = f"{batch_job.id}.jsonl"
        new_filepath = os.path.join(self._input_dir, new_filename)
        os.rename(input_filepath, new_filepath)
        self.logger.info(f"Renamed input file to {new_filepath}")

        self._serialize_batch_jobs(batch_job.id)

    def _serialize_batch_jobs(self, batch_job_id: str) -> None:
        """
        Appends the provided batch job ID to a tracking file for active jobs.

        Args:
            batch_job_id (str): The batch job identifier to serialize.
        """
        active_jobs_file = os.path.join(self._batch_dir, "active_batch_jobs.json")

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

    def monitor_batch_jobs(self):
        """
        Monitors the status of active batch jobs.

        For each active job:
          - If completed, retrieves results, aggregates them, and records the job in the completed jobs file.
          - If still active or retrieval fails, the job remains in the active list.

        Updates the active and completed job tracking files accordingly.
        """
        active_jobs_file = os.path.join(self._batch_dir, "active_batch_jobs.json")
        completed_jobs_file = os.path.join(self._batch_dir, "completed_batch_jobs.json")

        # ------------------------------------------------------------------
        # Loading active Jobs
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
        # Loading completed Jobs
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

            if batch_job.status == "completed":
                self.logger.ok(f"Batch job {job_id} completed.")
                result_file_name = self._retrieve_results(batch_job.output_file_id)
                self.aggregate_results()
                completed_jobs.append({"batch_id": job_id, "output": result_file_name})

            else:
                self.logger.info(f"Batch job {job_id} status: {batch_job.status}")
                still_active_jobs.append(job_id)

        # ------------------------------------------------------------------
        # Updating Active Jobs
        # ------------------------------------------------------------------
        with open(active_jobs_file, "w", encoding="utf-8") as f:
            json.dump(still_active_jobs, f, indent=2)

        # ------------------------------------------------------------------
        # Updating Completed Jobs
        # ------------------------------------------------------------------
        with open(completed_jobs_file, "w", encoding="utf-8") as f:
            json.dump(completed_jobs, f, indent=2)

    def _retrieve_results(self, batch_job_result_id: str) -> str:
        """
        Retrieves the results of a completed batch job, processes the content, and saves
        the aggregated output to the output directory.

        Args:
            batch_job_result_id (str): The file ID for the batch job's results.

        Returns:
            str: The filename of the saved results.
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
                                self.logger.error(f"Error parsing JSON: {e}. Using raw content.")
                                results_list.append(content)
                        else:
                            results_list.append(content)

                    except (KeyError, IndexError) as e:
                        self.logger.error(f"Could not extract content: {e}")

            results_filename = os.path.join(self._output_dir, f"{batch_job_result_id}.json")
            with open(results_filename, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2)

            self.logger.ok(f"Results saved to {results_filename}")
            return os.path.basename(results_filename)

        except Exception as e:
            self.logger.error(f"Error retrieving results for batch job {batch_job_result_id}: {str(e)}")
            return ""

    def aggregate_results(self):
        """
        Aggregates refined results from the output directory, writes a consolidated JSON file
        to the aggregated directory, and moves processed JSON files to the processed directory.
        """
        new_results = []
        for filename in os.listdir(self._output_dir):
            if not filename.lower().endswith(".json"):
                continue
            file_path = os.path.join(self._output_dir, filename)
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
                cleaned_data = self._clean_sample(sample)
                if cleaned_data:
                    new_results.append(cleaned_data)

            dest_path = os.path.join(self._processed_dir, filename)
            try:
                shutil.move(file_path, dest_path)
                self.logger.info(f"Moved {filename} -> {dest_path}")
            except Exception as e:
                self.logger.error(f"Error moving file {filename}: {e}")

        aggregated_file = os.path.join(self._aggregated_dir, "aggregated.json")
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

    def _clean_sample(self, sample_str: str) -> str:
        """
        Cleans a sample string by removing markdown formatting (e.g., triple backticks)
        and attempts to parse the contained JSON.

        Args:
            sample_str (str): The raw string sample.

        Returns:
            dict or None: The parsed JSON object if successful; otherwise, None.
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