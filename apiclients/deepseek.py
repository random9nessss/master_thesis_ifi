import os
import openai
import httpx
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from config.logger import CustomLogger
import shutil  # For moving files

class DeepseekClient:
    def __init__(self, email_generator):
        """
        Initialize a DeepseekClient instance.

        This version:
          - Sets up a custom logger.
          - Loads the API key from the ENV.txt file.
          - Sets the base URL to Deepseek's API endpoint.
          - Initializes the Deepseek client for regular chat completions.
          - Sets up directories for input/output.
          - Stores the provided email generator to construct user prompts.

        Parameters:
            email_generator (EmailGenerator): An instance used to generate email prompts.
        """
        self.logger = CustomLogger(name="DeepseekClient")
        self.logger.ok("DeepseekClient initialized")

        # ------------------------------------------------------------------
        # Load API Key and Initialize API Client
        # ------------------------------------------------------------------
        load_dotenv("ENV.txt")
        api_key = os.getenv("DEEPSEEK_KEY")
        if not api_key:
            self.logger.error("Deepseek API Key not found. Set DEEPSEEK_KEY in ENV.txt")
            exit(1)

        # ------------------------------------------------------------------
        # Directory Setup
        # ------------------------------------------------------------------
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()

        repo_root = current_dir
        repo_root = r"/Users/kevinbrundler/Desktop/Master Thesis" #Manual Overwrite
        self._directory = os.path.join(repo_root, "syntheticdata", "deepseek")
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
        # Deepseek Client
        # ------------------------------------------------------------------
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._email_generator = email_generator

    def get_chat_completion(
            self,
            model: str = "deepseek-chat",
            max_tokens: int = 2048,
            temperature: float = 1.0,
            system_prompt: str = "You are a helpful assistant"
    ) -> str:
        """
        Make a single chat completion request using the Deepseek API.

        This method constructs a conversation with a system prompt and a user prompt
        generated via the email generator, then returns the assistant's response.

        Parameters:
            model (str): The model to be used (default is "deepseek-chat").
            max_tokens (int): Maximum tokens in the response (default is 2048).
            temperature (float): Sampling temperature (default is 1.0).
            system_prompt (str): The system message for context.

        Returns:
            str: The content of the assistant's reply.
        """
        user_prompt = self._email_generator.construct_prompt()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            result = response.choices[0].message.content
            self.logger.ok("Chat completion obtained successfully.")
            return result
        except Exception as e:
            self.logger.error(f"Error obtaining chat completion: {str(e)}")
            return ""

    def serialize_response(self, response, filename: str = None) -> str:
        """
        Serialize the given response to a JSON file in an output directory.
        If no filename is provided, a timestamped filename is generated.

        Parameters:
            response (str or list): The response content to be saved.
            filename (str, optional): The desired filename for the serialized response.

        Returns:
            str: The full path to the serialized file.
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

    def _clean_sample(self, sample_str: str) -> dict:
        """
        Remove Markdown-style formatting from a sample string and parse it as JSON.

        Parameters:
            sample_str (str): The raw string containing JSON data, possibly enclosed in Markdown fences.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If the string cannot be parsed as JSON.
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
            self.logger.error(f"Error parsing JSON from sample: {e}")
            return {}

    def aggregate_results(self) -> None:
        """
        Aggregate results from the output directory by cleaning each response,
        merging them into a single aggregated JSON file with a unique timestamp-dependent name,
        and moving processed files to a 'processed' subdirectory.
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
            self.logger.ok(f"Aggregated results saved to {aggregated_file}")
        except Exception as e:
            self.logger.error(f"Error saving aggregated results: {e}")