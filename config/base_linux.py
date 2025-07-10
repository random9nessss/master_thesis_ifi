import os
import json
import random
import requests
import warnings
import datetime
from dotenv import load_dotenv
from config.logger import CustomLogger
from attributes.few_shot_candidates import FewShotCandidates

warnings.filterwarnings("ignore")

class Base:
    """
    The 'Base' class is responsible for generating synthetic email chains by querying
    a local text generation API. Each chain is stored as a JSON object in a .jsonl file.
    """

    def __init__(self, api_url: str = None, seed: int = None):
        if seed is not None:
            random.seed(seed)

        load_dotenv("ENV.txt")
        model_name = os.getenv("DEFAULT_MODEL", "llama8b")

        # Directory setup for output
        self._batch_dir = os.path.join(os.getcwd(), "syntheticdata", "baserefine", model_name)
        self._input_dir = os.path.join(self._batch_dir, "input_base")
        os.makedirs(self._input_dir, exist_ok=True)

        self._starting_candidates = FewShotCandidates().few_shot_candidates
        self.api_url = api_url or os.getenv("API_URL", "http://127.0.0.1:8000/generate")
        self.logger = CustomLogger(name="BaseGenerator")
        self.logger.ok("BaseGenerator initialized")

    def _call_generation_api(self, prompt: str) -> str:
        payload = {"prompt": prompt}
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return ""

    def _few_shot_base_prompt(self) -> str:
        email_a = random.choice(self._starting_candidates)
        email_b = random.choice([e for e in self._starting_candidates if e != email_a])

        prompt = f"""            
                    These are sample shipping emails:

                    Email A:
                    {email_a}
                    <END EMAIL A>

                    Email B:
                    {email_b}
                    <END EMAIL B>

                    Now write a new email in the same style.

                    Email C:
                """

        return prompt.replace("\n", "").strip()

    def _generate_email(self, prompt: str) -> str:
        return self._call_generation_api(prompt)

    def _generate_starting_candidate(self) -> str:
        """
        Generates and extracts a starting email candidate using a few-shot prompt.
        It looks for the text between "Email C:" and "<END EMAIL C>" and returns it.
        If the markers aren't found, it returns the full raw output.
        """
        prompt = self._few_shot_base_prompt()
        raw_output = self._call_generation_api(prompt)

        start_marker = "Email C:"
        start_index = raw_output.find(start_marker)
        if start_index == -1:
            self.logger.warning("Start marker 'Email C:' not found. Returning full output as candidate.")
            return raw_output.strip()

        start_index += len(start_marker)
        if "<END EMAIL C>" in raw_output:
            end_index = raw_output.find("<END EMAIL C>", start_index)
            candidate = raw_output[start_index:end_index].strip()
        else:
            candidate = raw_output[start_index:].strip()

        if not candidate:
            self.logger.warning("Extracted start email is empty. Returning full raw output.")
            candidate = raw_output.strip()

        return candidate

    def _iterative_email_generation(self,
                                    min_chain_length: int = 2,
                                    max_chain_length: int = 5) -> str:
        """
        Uses _generate_starting_candidate to get a valid starting email candidate,
        retrying if necessary, then builds the email chain.
        """
        max_attempts = 5
        attempt = 0
        start_email = ""

        while attempt < max_attempts and not start_email:
            attempt += 1
            candidate = self._generate_starting_candidate()
            if candidate:
                start_email = candidate
            else:
                self.logger.warning(f"Attempt {attempt}: Generated candidate is empty. Retrying...")

        if not start_email:
            self.logger.error(f"Failed to generate a valid start email after {max_attempts} attempts. Using fallback candidate.")
            start_email = random.choice(self._starting_candidates)
        else:
            self.logger.ok(f"Successfully generated starting candidate:")

        # Initialize the email chain with the valid starting candidate
        chain = f"Email 1:\n{start_email}\n"

        total_emails = random.randint(min_chain_length, max_chain_length)
        self.logger.info(f"Generating chain of length {total_emails}.")

        for i in range(2, total_emails + 1):
            prompt = chain + f"\n<Email {i}>:\n"
            generated = self._generate_email(prompt)
            email_split = generated.split(f"<Email {i}>:\n", 1)
            if len(email_split) > 1:
                chunk = email_split[1]
                next_email_marker = f"Email {i + 1}:"
                if next_email_marker in chunk:
                    chunk = chunk.split(next_email_marker, 1)[0]
                chain += f"Email {i}:\n{chunk.strip()}\n"
            else:
                chain += f"Email {i}:\n{generated.strip()}\n"

        return chain

    def generate_email_chains_to_file(self,
                                      num_chains: int,
                                      file_name: str = None):
        if file_name is None:
            file_name = f"base_chains_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        output_file_path = os.path.join(self._input_dir, file_name)
        chains = []

        self.logger.info(f"Generating {num_chains} email chains. Saving to: {output_file_path}")

        with open(output_file_path, "w", encoding="utf-8") as f:
            for i in range(num_chains):
                chain_text = self._iterative_email_generation()
                record = {"id": i, "chain": chain_text}
                f.write(json.dumps(record) + "\n")
                chains.append(chain_text)

        self.logger.ok(f"Generated and saved {num_chains} chains to {output_file_path}")
        return output_file_path
