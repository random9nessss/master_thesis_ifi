import os
import json
import random
import torch
import warnings
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from config.logger import CustomLogger
from attributes.few_shot_candidates import FewShotCandidates

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class Base:
    """
    The 'Base' class is responsible for generating synthetic email chains using
    a local (or HF-hosted) LLaMA-like model, then storing each chain in a .jsonl file
    in 'baserefine/input/' for subsequent refinement steps.
    """

    _model_cache = {}

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", cuda: bool = True, seed: int = None):
        """
        Initializes the Base generator, including model and tokenizer loading.
        """

        # -------------------------------------------------------------------
        # Seeding Randomness to Control for Prompt Generation
        # -------------------------------------------------------------------
        if seed is not None:
            random.seed(seed)

        # -------------------------------------------------------------------
        # Loading Environment Variables
        # -------------------------------------------------------------------
        load_dotenv("ENV.txt")

        # -------------------------------------------------------------------
        # Directory Setup
        # -------------------------------------------------------------------
        self._batch_dir = os.path.join(os.getcwd(), "syntheticdata", "baserefine")
        self._input_dir = os.path.join(self._batch_dir, "input_base")
        os.makedirs(self._input_dir, exist_ok=True)

        # -------------------------------------------------------------------
        # Starting Candidates
        # -------------------------------------------------------------------
        self._starting_candidates = FewShotCandidates().few_shot_candidates

        # -------------------------------------------------------------------
        # Model Setup
        # -------------------------------------------------------------------
        self._tokenizer, self._model = self._load_model_and_tokenizer(model_name, cuda)
        self._cuda = cuda

        # -------------------------------------------------------------------
        # Logger
        # -------------------------------------------------------------------
        self.logger = CustomLogger(name="BaseGenerator")
        self.logger.ok("BaseGenerator initialized")

    def _load_model_and_tokenizer(self, model_name: str, cuda: bool = True):
        """
         Loads the tokenizer and model from Hugging Face, utilizing a cache to avoid repeated downloads.

         Args:
             model_name (str): The Hugging Face model identifier to load.
             cuda (bool): Whether to move the model to GPU. Defaults to True.

         Returns:
             tuple: A tuple containing the tokenizer and model.

         Raises:
             ValueError: If the 'HF_TOKEN' environment variable is not set.
         """
        # -------------------------------------------------------------------
        # Model Cache Retrieval
        # -------------------------------------------------------------------
        if model_name in Base._model_cache:
            self.logger.info(f"Model '{model_name}' cached. Using existing instance.")
            return Base._model_cache[model_name]

        # -------------------------------------------------------------------
        # Loading environment variables
        # -------------------------------------------------------------------
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in ENV.txt")

        # -------------------------------------------------------------------
        # Model Loading
        # -------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

        if cuda:
            torch.cuda.empty_cache()
            model.half().cuda()

        Base._model_cache[model_name] = (tokenizer, model)
        return tokenizer, model

    def _generate_email(self,
                        prompt: str,
                        max_new_tokens: int = 180,
                        do_sample: bool = True,
                        temperature: float = 1.0,
                        top_p: float = 0.9) -> str:
        """
        Generates a single email by producing text from the language model based on the provided prompt.

        Args:
            prompt (str): The text prompt to feed into the model.
            max_new_tokens (int): Maximum number of tokens to generate. Defaults to 180.
            do_sample (bool): Whether to use sampling; if False, uses greedy decoding. Defaults to True.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_p (float): Nucleus sampling probability threshold. Defaults to 0.9.

        Returns:
            str: The generated email text.
        """
        # -------------------------------------------------------------------
        # Device Setup
        # -------------------------------------------------------------------
        if self._cuda:
            inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")

        # -------------------------------------------------------------------
        # Output Generation
        # -------------------------------------------------------------------
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )

        # -------------------------------------------------------------------
        # Output Decoding
        # -------------------------------------------------------------------
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _few_shot_base_prompt(self) -> str:
        """
        Constructs a few-shot prompt using two randomly selected starting email candidates.

        Returns:
            str: A formatted few-shot prompt for generating a new email in a similar style.
        """
        # -------------------------------------------------------------------
        # Few-Shot Sampling
        # -------------------------------------------------------------------
        email_a = random.choice(self._starting_candidates)
        email_b = random.choice([e for e in self._starting_candidates if e != email_a])

        # -------------------------------------------------------------------
        # Few-Shot Prompt
        # -------------------------------------------------------------------
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

    def _generate_starting_candidate(self) -> str:
        """
         Generates a starting email candidate using a few-shot prompt.

         This method queries the language model with a few-shot prompt to generate a new email,
         and then returns the raw generated text.

         Returns:
             str: The raw email text output from the language model.
         """
        # -------------------------------------------------------------------
        # Few-Shot Prompt Generation
        # -------------------------------------------------------------------
        prompt = self._few_shot_base_prompt()

        # -------------------------------------------------------------------
        # Device Setup
        # -------------------------------------------------------------------
        if torch.cuda.is_available() and self._cuda:
            inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")

        # -------------------------------------------------------------------
        # Output Generation
        # -------------------------------------------------------------------
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=1,
                top_p=0.9
            )

        # -------------------------------------------------------------------
        # Output Decoding
        # -------------------------------------------------------------------
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _iterative_email_generation(self,
                                   min_chain_length: int = 2,
                                   max_chain_length: int = 5,
                                   max_new_tokens: int = 180,
                                   temperature: float = 1.0,
                                   top_p: float = 0.9) -> str:
        """
        Iteratively generates a chain of emails by prompting the model for each subsequent email.

        The process involves:
            1. Generating a starting email candidate using a few-shot prompt.
            2. Determining the total number of emails in the chain (randomly chosen between min_chain_length and max_chain_length).
            3. Iteratively generating subsequent emails based on the previous context.

        Args:
            min_chain_length (int): The minimum number of emails in the chain. Defaults to 2.
            max_chain_length (int): The maximum number of emails in the chain. Defaults to 5.
            max_new_tokens (int): Maximum new tokens to generate for each email. Defaults to 180.
            temperature (float): Sampling temperature for generation. Defaults to 1.0.
            top_p (float): Nucleus sampling probability threshold. Defaults to 0.9.

        Returns:
            str: The complete email chain as a single formatted string.
        """
        # -------------------------------------------------------------------
        # Start Email Generation
        # -------------------------------------------------------------------
        max_attempts = 3
        attempt = 0
        start_email = ""

        while attempt < max_attempts and not start_email:
            attempt += 1
            raw_output = self._generate_starting_candidate()
            start_marker = "Email C:"
            start_index = raw_output.find(start_marker)

            if start_index == -1:
                self.logger.warning(f"Attempt {attempt}: Start marker 'Email C:' not found in output. Retrying...")
                continue

            start_index += len(start_marker)

            if "<END EMAIL C>" in raw_output:
                end_index = raw_output.find("<END EMAIL C>", start_index)
                start_email = raw_output[start_index:end_index].strip()

            else:
                start_email = raw_output[start_index:].strip()

            if not start_email:
                self.logger.warning(f"Attempt {attempt}: Extracted start email is empty. Retrying...")

        # -------------------------------------------------------------------
        # Fallback: Random Starting Candidate Sampling
        # -------------------------------------------------------------------
        if not start_email:
            self.logger.error(f"Failed to generate a valid start email after {max_attempts} attempts.")
            start_email = random.choice(self._starting_candidates)

        # -------------------------------------------------------------------
        # Initialize the email chain
        # -------------------------------------------------------------------
        chain = f"Email 1:\n{start_email}\n"

        # -------------------------------------------------------------------
        # Determine total chain length
        # -------------------------------------------------------------------
        total_emails = random.randint(min_chain_length, max_chain_length)
        self.logger.info(f"Generating chain of length {total_emails}.")

        for i in range(2, total_emails + 1):
            prompt = chain + f"\n<Email {i}>:\n"

            # -------------------------------------------------------------------
            # Generation of next Email
            # -------------------------------------------------------------------
            generated = self._generate_email(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )

            email_i_text = generated.split(f"<Email {i}>:\n", 1)
            if len(email_i_text) > 1:
                next_email_marker = f"Email {i + 1}:"
                chunk = email_i_text[1]

                if next_email_marker in chunk:
                    chunk = chunk.split(next_email_marker, 1)[0]
                chain += f"Email {i}:\n{chunk.strip()}\n"

            else:
                chain += f"Email {i}:\n{generated.strip()}\n"

        return chain

    def generate_email_chains_to_file(self,
                                      num_chains: int,
                                      file_name: str = "base_chains.jsonl"):
        """
        Generates a specified number of synthetic email chains and writes them to a file.

        Each chain is saved as a JSON object on a separate line in the format:
            {"id": <int>, "chain": "<email chain text>"}

        Args:
            num_chains (int): The number of email chains to generate.
            file_name (str): The name of the file to write the chains to. Defaults to "base_chains.jsonl".

        Returns:
            str: The full path to the output file containing the generated email chains.
        """
        output_file_path = os.path.join(self._input_dir, file_name)
        chains = []

        self.logger.info(f"Generating {num_chains} email chains. Saving to: {output_file_path}")

        # -------------------------------------------------------------------
        # Generation and Serialization
        # -------------------------------------------------------------------
        with open(output_file_path, "w", encoding="utf-8") as f:
            for i in range(num_chains):
                chain_text = self._iterative_email_generation()
                record = {
                    "id": i,
                    "chain": chain_text
                }
                f.write(json.dumps(record) + "\n")
                chains.append(chain_text)

        self.logger.ok(f"Generated and saved {num_chains} chains to {output_file_path}")
        return output_file_path
