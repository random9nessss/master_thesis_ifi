import os
import re
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

from config.logger import CustomLogger
from emailprocessor import EmailPreprocessor

import requests
from huggingface_hub import configure_http_backend

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("llm_engine").setLevel(logging.ERROR)
logging.getLogger("cuda").setLevel(logging.ERROR)
logging.getLogger("multiproc_worker_utils").setLevel(logging.ERROR)
logging.getLogger("custom_all_reduce").setLevel(logging.ERROR)
logging.getLogger("custom_cache_manager").setLevel(logging.ERROR)

def backend_factory() -> requests.Session:
    """
     Create and configure a Requests session that disables SSL verification.

     Returns:
         requests.Session: A configured Requests session.
     """
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)


# -------------------------------
# Device Configuration
# -------------------------------
DEVICE_GPT = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_T5 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# -------------------------------
# Load Models
# -------------------------------

# Scoring Model
gpt2_model_name = "gpt2-xl"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(DEVICE_GPT)
gpt2_model.eval()

# Paraphrasing Model
t5_model_name = "google/t5-v1_1-xl"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(DEVICE_T5)
t5_model.eval()


# -------------------------------
# Helper Functions
# -------------------------------
def compute_log_likelihood(text, model, tokenizer):
    """
    Computes the negative log likelihood (loss) for the given text using a causal LM.
    Lower loss means higher likelihood.

    If the input text is empty or tokenization yields no tokens, returns np.nan.

    Args:
        text (str): Input text.
        model: The language model.
        tokenizer: The corresponding tokenizer.

    Returns:
        float: The loss value (average cross entropy per token), or np.nan if input is empty.
    """
    text = truncate_long_text(text)
    text = text.strip()
    if not text:
        return np.nan

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    if inputs["input_ids"].numel() == 0:
        return np.nan

    inputs = {k: v.to(DEVICE_GPT) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def process_human_emails(human_emails, n_perturbations, span_length, pct_mask, use_log_ratio=True, logger=None):
    """Process human emails once and return the detailed results and scores.

    Args:
        human_emails (list): List of human email content to analyze
        n_perturbations (int): Number of perturbations per email
        span_length (int): Length of each masked span in words
        pct_mask (float): Percentage of words to mask
        use_log_ratio (bool): Whether to use log ratio for scoring
        logger: Optional logger for warnings

    Returns:
        dict: Dictionary with "scores" and "details" keys
    """
    if logger is None:
        logger = CustomLogger(name="HumanEmailProcessor")

    human_scores = []
    human_details = []

    logger.info(f"Running one-time perturbations on {len(human_emails)} human emails...")
    logger.info(f"Using perturbation parameters: span_length={span_length}, pct_mask={pct_mask}")
    logger.info(f"Scoring method: {'Log-ratio' if use_log_ratio else 'Difference'}")

    for i, email in enumerate(human_emails):
        if not email.strip():
            logger.warning(f"Skipping empty human email at index {i}")
            continue

        orig_ll = compute_log_likelihood(email, gpt2_model, gpt2_tokenizer)
        if np.isnan(orig_ll):
            logger.warning(f"Skipping human email at index {i} due to NaN log likelihood")
            continue

        perturbed_texts = run_perturbations(email,
                                            n_perturbations=n_perturbations,
                                            span_length=span_length,
                                            pct_mask=pct_mask)

        perturbed_lls = [compute_log_likelihood(pt, gpt2_model, gpt2_tokenizer) for pt in perturbed_texts]
        valid_perturbed_lls = [ll for ll in perturbed_lls if not np.isnan(ll)]

        if not valid_perturbed_lls:
            logger.warning(f"Skipping human email at index {i} due to no valid perturbed log likelihoods")
            continue

        avg_perturbed_ll = np.mean(valid_perturbed_lls)
        std_perturbed_ll = np.std(valid_perturbed_lls)

        if use_log_ratio:
            epsilon = 1e-10
            score = np.log(orig_ll + epsilon) - np.log(avg_perturbed_ll + epsilon)
        else:
            if std_perturbed_ll > 0:
                score = (orig_ll - avg_perturbed_ll) / std_perturbed_ll
            else:
                score = orig_ll - avg_perturbed_ll

        human_scores.append(score)

        human_details.append({
            "email": email,
            "orig_ll": orig_ll,
            "avg_perturbed_ll": avg_perturbed_ll,
            "std_perturbed_ll": std_perturbed_ll,
            "score": score,
            "perturbed_lls": valid_perturbed_lls,
            "perturbed_samples": perturbed_texts
        })

        logger.info(
            f"Human email [{i + 1}/{len(human_emails)}]: orig_ll={orig_ll:.3f}, avg_perturbed_ll={avg_perturbed_ll:.3f}, score={score:.3f}")

    return {
        "scores": human_scores,
        "details": human_details
    }

def truncate_long_text(text, max_words=150):
    """Simple function to truncate very long texts to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text

    perturbation_logger.warning(f"Truncating text from {len(words)} to {max_words} words")
    return " ".join(words[:max_words])

perturbation_logger = CustomLogger(name="PerturbationLogger")


def perturb_text(text, span_length=4, pct_mask=0.3, force_perturb=True):
    """
    Perturbs text by masking and regenerating a specified percentage of the words.

    Args:
        text (str): Text to perturb
        span_length (int): Length of each masked span in words
        pct_mask (float): Percentage of words to mask (0.0-1.0)
        force_perturb (bool): Whether to force at least one perturbation

    Returns:
        str: The perturbed text
    """
    text = truncate_long_text(text)
    words = text.split()
    n_words = len(words)

    if n_words < span_length + 2:
        return text

    num_spans_to_mask = max(1 if force_perturb else 0, int(n_words * pct_mask / span_length))

    if not force_perturb and random.random() > pct_mask:
        return text

    if num_spans_to_mask == 0:
        return text

    perturbed_text = text

    perturbed_positions = set()

    spans_perturbed = 0
    max_attempts = num_spans_to_mask * 3

    attempt = 0
    while spans_perturbed < num_spans_to_mask and attempt < max_attempts:
        attempt += 1

        current_words = perturbed_text.split()
        current_n_words = len(current_words)

        if current_n_words < span_length + 2:
            break

        valid_starts = []
        for i in range(current_n_words - span_length + 1):
            if not any(i + j in perturbed_positions for j in range(span_length)):
                valid_starts.append(i)

        if not valid_starts:
            break

        start = random.choice(valid_starts)
        end = start + span_length

        for i in range(start, end):
            perturbed_positions.add(i)

        span_text = ' '.join(current_words[start:end])
        perturbation_logger.warning(f"Masking words {start} to {end}: '{span_text}'")

        mask_token = f"<extra_id_{spans_perturbed % 100}>"
        masked_words = current_words.copy()
        masked_words[start:end] = [mask_token]
        masked_text = " ".join(masked_words)

        input_ids = t5_tokenizer.encode(
            f"fill: {masked_text}",
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(DEVICE_T5)

        outputs = t5_model.generate(
            input_ids,
            max_length=min(len(input_ids[0]) + span_length * 2, 448),
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )

        generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=False)

        if mask_token in generated:
            fill = generated.split(mask_token)[1].split("<extra_id_")[0].strip()
        else:
            fill = generated.replace("fill:", "").strip()

        special_tokens = ["<pad>", "</s>"] + [f"<extra_id_{i}>" for i in range(100)]
        for token in special_tokens:
            fill = fill.replace(token, "")

        fill = fill.strip()

        new_perturbed_text = masked_text.replace(mask_token, fill)

        for token in special_tokens:
            new_perturbed_text = new_perturbed_text.replace(token, "")

        if new_perturbed_text != perturbed_text:
            perturbed_text = new_perturbed_text
            spans_perturbed += 1

    perturbation_logger.warning(f"Successfully perturbed {spans_perturbed}/{num_spans_to_mask} spans "
                                f"({spans_perturbed * span_length / n_words:.1%} of words)")

    return perturbed_text


def run_perturbations(text, n_perturbations=50, span_length=3, pct_mask=0.2):
    """
    Runs a number of perturbations on a given text.

    Args:
        text (str): The original text.
        n_perturbations (int): The number of perturbations to generate.
        span_length (int): The number of words to mask in each span.
        pct_mask (float): The percentage of total words to mask.

    Returns:
        List[str]: A list of perturbed texts.
    """
    batch_size = 2
    perturbed_texts = []

    for i in range(0, n_perturbations, batch_size):
        batch_count = min(batch_size, n_perturbations - i)
        batch = [perturb_text(text, span_length=span_length, pct_mask=pct_mask)
                 for _ in range(batch_count)]
        perturbed_texts.extend(batch)

        torch.cuda.empty_cache()

    return perturbed_texts


def default_ai_email_paths():
    """
    Returns a default dictionary of AI email JSON file paths.

    Returns:
        dict: Mapping of model keys to JSON file paths.
    """

    return {
        # ------------------------------------------------------------------
        # Attribute Prompting
        # ------------------------------------------------------------------
        "attr_prompting_claude": "../data/email_datasets/synthetic/attrprompting/claude/aggregated/aggregated.json",
        "attr_prompting_deepseek": "../data/email_datasets/synthetic/attrprompting/deepseek/aggregated/aggregated.json",
        "attr_prompting_gemini": "../data/email_datasets/synthetic/attrprompting/gemini/aggregated/aggregated.json",
        "attr_prompting_gpt4": "../data/email_datasets/synthetic/attrprompting/gpt-4-turbo/aggregated/aggregated.json",
        "attr_prompting_mistral": "../data/email_datasets/synthetic/attrprompting/mistral/aggregated/aggregated.json",

        # ------------------------------------------------------------------
        # BARE Llama3B
        # ------------------------------------------------------------------
        "bare_llama3b_claude": "../data/email_datasets/synthetic/baserefine/refine/llama3b/claude/aggregated/aggregated.json",
        "bare_llama3b_deepseek": "../data/email_datasets/synthetic/baserefine/refine/llama3b/deepseek/aggregated/aggregated.json",
        "bare_llama3b_gemini": "../data/email_datasets/synthetic/baserefine/refine/llama3b/gemini/aggregated/aggregated.json",
        "bare_llama3b_gpt4": "../data/email_datasets/synthetic/baserefine/refine/llama3b/gpt-4-turbo/aggregated/aggregated.json",
        "bare_llama3b_mistral": "../data/email_datasets/synthetic/baserefine/refine/llama3b/mistral/aggregated/aggregated.json",

        # ------------------------------------------------------------------
        # BARE Llama8B
        # ------------------------------------------------------------------
        "bare_llama8b_claude": "../data/email_datasets/synthetic/baserefine/refine/llama8b/claude/aggregated/aggregated.json",
        "bare_llama8b_deepseek": "../data/email_datasets/synthetic/baserefine/refine/llama8b/deepseek/aggregated/aggregated.json",
        "bare_llama8b_gemini": "../data/email_datasets/synthetic/baserefine/refine/llama8b/gemini/aggregated/aggregated.json",
        "bare_llama8b_gpt4": "../data/email_datasets/synthetic/baserefine/refine/llama8b/gpt-4-turbo/aggregated/aggregated.json",
        "bare_llama8b_mistral": "../data/email_datasets/synthetic/baserefine/refine/llama8b/mistral/aggregated/aggregated.json",

        # ------------------------------------------------------------------
        # Few Shot
        # ------------------------------------------------------------------
        "fewshot_claude": "../data/email_datasets/synthetic/fewshot/claude/aggregated/aggregated.json",
        "fewshot_deepseek": "../data/email_datasets/synthetic/fewshot/deepseek/aggregated/aggregated.json",
        "fewshot_gemini": "../data/email_datasets/synthetic/fewshot/gemini/aggregated/aggregated.json",
        "fewshot_gpt4": "../data/email_datasets/synthetic/fewshot/gpt-4-turbo/aggregated/aggregated.json",
        "fewshot_mistral": "../data/email_datasets/synthetic/fewshot/mistral/aggregated/aggregated.json",

        # ------------------------------------------------------------------
        # Zero Shot
        # ------------------------------------------------------------------
        "zeroshot_claude": "../data/email_datasets/synthetic/zeroshot/claude/aggregated/aggregated.json",
        "zeroshot_deepseek": "../data/email_datasets/synthetic/zeroshot/deepseek/aggregated/aggregated.json",
        "zeroshot_gemini": "../data/email_datasets/synthetic/zeroshot/gemini/aggregated/aggregated.json",
        "zeroshot_gpt4": "../data/email_datasets/synthetic/zeroshot/gpt-4-turbo/aggregated/aggregated.json",
        "zeroshot_mistral": "../data/email_datasets/synthetic/zeroshot/mistral/aggregated/aggregated.json",
    }


def sample_ai_emails(model_paths: dict, n: int = 50, random_state: int = 42, logger=None) -> dict:
    """
    Samples AI email bodies from JSON files specified in model_paths.

    Args:
        model_paths (dict): Dictionary mapping model keys to JSON file paths.
        n (int): Number of samples to extract per model.
        random_state (int): Random seed.
        logger: Optional logger for warnings.

    Returns:
        dict: Mapping of model keys to lists of sampled email bodies.
    """
    random.seed(random_state)
    sampled_emails = {}
    for key, file_path in model_paths.items():
        try:
            df = pd.read_json(file_path)
            chains = df["email_chain"].dropna().tolist()
        except Exception as e:
            if logger:
                logger.warning(f"Couldn't load data for {key}: {e}")
            continue
        sampled_bodies = []
        while len(sampled_bodies) < n:
            chain = random.choice(chains)
            if chain:
                email = random.choice(chain)
                body = email.get("body", "").strip()
                if body:
                    sampled_bodies.append(body)
        sampled_emails[key] = sampled_bodies
    return sampled_emails


def load_local_email_data(number_of_samples=50, buffer_size=10, logger=None) -> dict:
    """
    Loads human emails from a local Enron CSV and samples AI emails from JSON files.

    Args:
        number_of_samples (int): Number of human emails to sample.
        buffer_size (int): Additional samples to include for buffering.
        logger: Optional logger.

    Returns:
        dict: Dictionary containing keys "original" (human emails),
              "sampled" (AI emails by model key), and "buffer" for additional data.
    """
    human_csv = r"/home/ANYACCESS.NET/brk.ch/src/thesis/Masterthesis/datasets_processed/enron_parsed.csv"
    df_enron = pd.read_csv(human_csv)
    total_samples = number_of_samples + buffer_size
    sampled_emails = df_enron["parsed_content"].dropna().sample(total_samples, random_state=42).tolist()

    # -------------------------------
    # Preprocessing Human and Ai Emails
    # -------------------------------
    preprocessor = EmailPreprocessor(debug_mode=False)
    cleaned_emails = [preprocessor.preprocess(email) for email in sampled_emails]

    human_emails = cleaned_emails[:number_of_samples]
    ai_emails = sample_ai_emails(default_ai_email_paths(), n=total_samples, random_state=42, logger=logger)

    for model in ai_emails:
        ai_emails[model] = [preprocessor.preprocess(email) for email in ai_emails[model]][:number_of_samples]

    return {
        "original": human_emails,
        "sampled": ai_emails,
        "buffer": {
            "original": cleaned_emails[number_of_samples:],
            "sampled": {model: emails[number_of_samples:] for model, emails in ai_emails.items()}
        }
    }


# -------------------------------
# DetectGPT Experiment Class
# -------------------------------
class DetectGPT:
    """
    A class to run the DetectGPT experiment.

    For each email (from human and AI sets), it computes the log likelihood using GPT-2,
    generates a number of perturbed versions using T5, and computes a score based on:
       score = log(original_ll) - log(average(perturbed_ll))
    or using the original formula:
       score = original_ll - average(perturbed_ll)

    Results are logged, and a histogram of scores is saved to:
        output/detectgpt/{ai_model_key}.png

    The data used in the experiment is also saved to:
        output/detectgpt/data/{ai_model_key}.json
    """

    def __init__(self,
                 n_samples: int = 50,
                 n_perturbations: int = 50,
                 ai_model_key: str = None,
                 span_length: int = 3,
                 pct_mask: float = 0.2,
                 use_log_ratio: bool = True,
                 human_results: dict = None):
        """
        Initialize the DetectGPT experiment.

        Args:
            n_samples (int): Number of emails to sample for both human and AI.
            n_perturbations (int): Number of perturbations per email.
            ai_model_key (str, optional): The key (from default_ai_email_paths) to select AI emails.
                                          Defaults to "attr_prompting_claude" if not provided.
            span_length (int): Number of words to mask in each perturbation.
            pct_mask (float): Probability that the text will be perturbed.
            use_log_ratio (bool): Whether to use log ratio for scoring.
        """
        self.n_samples = n_samples
        self.n_perturbations = n_perturbations
        self.span_length = span_length
        self.pct_mask = pct_mask
        self.use_log_ratio = use_log_ratio
        self.human_results = human_results
        self.logger = CustomLogger(name="DetectGPT")

        self.data = load_local_email_data(number_of_samples=n_samples, buffer_size=10, logger=self.logger)
        self.human_emails = self.data["original"]
        ai_dict = self.data["sampled"]

        if ai_model_key is None:
            self.ai_model_key = "attr_prompting_claude"
        else:
            self.ai_model_key = ai_model_key

        if self.ai_model_key in ai_dict:
            self.ai_emails = ai_dict[self.ai_model_key]
        else:
            self.logger.warning(f"Model key {self.ai_model_key} not found in sampled AI emails. Using empty list.")
            self.ai_emails = []

        # -------------------------------
        # Output Saving
        # -------------------------------
        self.output_dir = os.path.join(os.getcwd(), "output", "detectgpt")
        self.data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def run_experiment(self):
        """
        Runs the perturbation experiment for both human and AI emails.

        For each email, computes:
          - orig_ll: Log likelihood of the original email.
          - perturbed_ll: Average log likelihood over n_perturbations.

        If using log-ratio:
          - score = log(orig_ll) - log(avg(perturbed_ll))
        Else:
          - score = orig_ll - avg(perturbed_ll) (normalized by std if non-zero)

        Returns:
            Tuple[List[float], List[float]]: Scores for human and AI emails.
        """
        if self.human_results:
            self.logger.info("Using pre-computed human email results")
            human_scores = self.human_results["scores"]
            human_details = self.human_results["details"]
        else:
            self.logger.info(f"Using perturbation parameters: span_length={self.span_length}, pct_mask={self.pct_mask}")
            self.logger.info(f"Scoring method: {'Log-ratio' if self.use_log_ratio else 'Difference'}")
            self.logger.info("Running perturbations on human emails...")
            human_scores = []
            human_details = []

            for i, email in enumerate(self.human_emails):
                if not email.strip():
                    self.logger.warning(f"Skipping empty human email at index {i}")
                    continue

                orig_ll = compute_log_likelihood(email, gpt2_model, gpt2_tokenizer)
                if np.isnan(orig_ll):
                    self.logger.warning(f"Skipping human email at index {i} due to NaN log likelihood")
                    continue

                perturbed_texts = run_perturbations(email,
                                                    n_perturbations=self.n_perturbations,
                                                    span_length=self.span_length,
                                                    pct_mask=self.pct_mask)

                perturbed_lls = [compute_log_likelihood(pt, gpt2_model, gpt2_tokenizer) for pt in perturbed_texts]
                valid_perturbed_lls = [ll for ll in perturbed_lls if not np.isnan(ll)]

                if not valid_perturbed_lls:
                    self.logger.warning(f"Skipping human email at index {i} due to no valid perturbed log likelihoods")
                    continue

                avg_perturbed_ll = np.mean(valid_perturbed_lls)
                std_perturbed_ll = np.std(valid_perturbed_lls)

                if self.use_log_ratio:
                    epsilon = 1e-10
                    score = np.log(orig_ll + epsilon) - np.log(avg_perturbed_ll + epsilon)
                else:
                    if std_perturbed_ll > 0:
                        score = (orig_ll - avg_perturbed_ll) / std_perturbed_ll
                    else:
                        score = orig_ll - avg_perturbed_ll

                human_scores.append(score)

                # -------------------------------
                # Storing Experiment Details
                # -------------------------------
                human_details.append({
                    "email": email,
                    "orig_ll": orig_ll,
                    "avg_perturbed_ll": avg_perturbed_ll,
                    "std_perturbed_ll": std_perturbed_ll,
                    "score": score,
                    "perturbed_lls": valid_perturbed_lls,
                    "perturbed_samples": perturbed_texts
                })

                self.logger.info(
                    f"Human email [{i + 1}/{len(self.human_emails)}]: orig_ll={orig_ll:.3f}, avg_perturbed_ll={avg_perturbed_ll:.3f}, score={score:.3f}")

        ai_scores = []
        ai_details = []

        self.logger.info("Running perturbations on AI emails...")
        for i, email in enumerate(self.ai_emails):
            if not email.strip():
                self.logger.warning(f"Skipping empty AI email at index {i}")
                continue

            orig_ll = compute_log_likelihood(email, gpt2_model, gpt2_tokenizer)
            if np.isnan(orig_ll):
                self.logger.warning(f"Skipping AI email at index {i} due to NaN log likelihood")
                continue

            perturbed_texts = run_perturbations(email,
                                                n_perturbations=self.n_perturbations,
                                                span_length=self.span_length,
                                                pct_mask=self.pct_mask)

            perturbed_lls = [compute_log_likelihood(pt, gpt2_model, gpt2_tokenizer) for pt in perturbed_texts]
            valid_perturbed_lls = [ll for ll in perturbed_lls if not np.isnan(ll)]

            if not valid_perturbed_lls:
                self.logger.warning(f"Skipping AI email at index {i} due to no valid perturbed log likelihoods")
                continue

            avg_perturbed_ll = np.mean(valid_perturbed_lls)
            std_perturbed_ll = np.std(valid_perturbed_lls)

            if self.use_log_ratio:
                epsilon = 1e-10
                score = np.log(orig_ll + epsilon) - np.log(avg_perturbed_ll + epsilon)
            else:
                if std_perturbed_ll > 0:
                    score = (orig_ll - avg_perturbed_ll) / std_perturbed_ll
                else:
                    score = orig_ll - avg_perturbed_ll

            ai_scores.append(score)

            # -------------------------------
            # Storing Experiment Details
            # -------------------------------
            ai_details.append({
                "email": email,
                "orig_ll": orig_ll,
                "avg_perturbed_ll": avg_perturbed_ll,
                "std_perturbed_ll": std_perturbed_ll,
                "score": score,
                "perturbed_lls": valid_perturbed_lls,
                "perturbed_samples": perturbed_texts
            })

            self.logger.info(
                f"AI email [{i + 1}/{len(self.ai_emails)}]: orig_ll={orig_ll:.3f}, avg_perturbed_ll={avg_perturbed_ll:.3f}, score={score:.3f}")

        self.save_experiment_data(human_details, ai_details)

        self.human_scores = human_scores
        self.ai_scores = ai_scores
        return human_scores, ai_scores

    def save_experiment_data(self, human_details, ai_details):
        """
        Saves the experiment data to a JSON file.

        Args:
            human_details (list): Details of human email processing
            ai_details (list): Details of AI email processing
        """
        experiment_data = {
            "experiment_settings": {
                "n_samples": self.n_samples,
                "n_perturbations": self.n_perturbations,
                "span_length": self.span_length,
                "pct_mask": self.pct_mask,
                "ai_model_key": self.ai_model_key,
                "scoring_method": "log_ratio" if self.use_log_ratio else "difference",
                "gpt2_model": gpt2_model_name,
                "t5_model": t5_model_name
            },
            "human_emails": human_details,
            "ai_emails": ai_details
        }

        score_method = "logratio" if self.use_log_ratio else "diff"
        filename = f"{self.ai_model_key}_s{self.span_length}_p{int(self.pct_mask * 100)}_{score_method}.json"
        output_path = os.path.join(self.data_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Experiment data saved to {output_path}")

    def plot_paper_style_results(self, models_data=None, grid_shape=(2, 2), figsize=(10, 8)):
        """
        Plots histograms in a paper-like style showing perturbation discrepancy distributions
        for multiple models in a grid layout.

        Parameters:
        -----------
        models_data : dict or None
            A dictionary where keys are model names and values are dicts with 'human_scores' and 'ai_scores'.
            If None, uses the current instance's scores with its model name.
        grid_shape : tuple
            The shape of the grid (rows, cols) for subplot arrangement.
        figsize : tuple
            Figure size (width, height) in inches.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import matplotlib as mpl
        from matplotlib.ticker import MaxNLocator

        # Use a clean, publication-style plot
        plt.style.use('seaborn-v0_8-whitegrid')

        # Define colors to match the paper's style
        human_color = '#4b86b4'  # Blue
        model_color = '#f58a42'  # Orange

        # Create the figure with the specified grid
        fig, axes = plt.subplots(*grid_shape, figsize=figsize, constrained_layout=True)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Set up the data
        if models_data is None:
            # Use current instance data
            models_data = {
                self.ai_model_key: {
                    'human_scores': self.human_scores,
                    'ai_scores': self.ai_scores
                }
            }

        # Plot each model in its subplot
        for i, (model_name, data) in enumerate(models_data.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get the scores
            human_scores = data['human_scores']
            ai_scores = data['ai_scores']

            # Calculate appropriate bin settings
            all_scores = human_scores + ai_scores
            min_score, max_score = min(all_scores), max(all_scores)
            bin_width = (max_score - min_score) / 15  # Aim for about 15 bins
            bins = np.arange(min_score, max_score + bin_width, bin_width)

            # Plot histograms
            sns.histplot(human_scores, bins=bins, alpha=0.7, label="Human", color=human_color, ax=ax)
            sns.histplot(ai_scores, bins=bins, alpha=0.7, label="Model", color=model_color, ax=ax)

            # Clean up the plot
            ax.set_title(model_name, fontsize=12)

            # Apply integer frequency ticks for y-axis
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if i % grid_shape[1] == 0:  # First column
                ax.set_ylabel("Frequency", fontsize=11)
            else:
                ax.set_ylabel("")

            if i >= len(axes) - grid_shape[1]:  # Last row
                ax.set_xlabel("Log Likelihood Drop (Perturbation Discrepancy)", fontsize=11)
            else:
                ax.set_xlabel("")

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add legend only to the first plot
            if i == 0:
                ax.legend(frameon=True, loc='upper right', fontsize=10)
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Save the figure
        method = "Log-Ratio" if hasattr(self, 'use_log_ratio') and self.use_log_ratio else "Difference"
        filename = f"detectgpt_comparison_{'_'.join(models_data.keys())}_{method}.png"
        output_path = os.path.join(self.output_dir, filename) if hasattr(self, 'output_dir') else filename

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if hasattr(self, 'logger'):
            self.logger.info(f"Plot saved to {output_path}")
        plt.close()

        return output_path

    def plot_results(self):
        """
        Plots histograms of scores for human and AI emails and saves the plot.

        The plot is saved to output/detectgpt/{ai_model_key}_{parameters}.png.
        """
        if not hasattr(self, 'human_scores') or not hasattr(self, 'ai_scores'):
            self.logger.warning("No scores available for plotting. Run experiment first.")
            return

        all_scores = self.human_scores + self.ai_scores
        all_labels = [0] * len(self.human_scores) + [1] * len(self.ai_scores)

        thresholds = sorted(all_scores)
        best_accuracy = 0
        best_threshold = 0

        for threshold in thresholds:
            predictions = [1 if score >= threshold else 0 for score in all_scores]
            accuracy = sum(pred == label for pred, label in zip(predictions, all_labels)) / len(all_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        predictions = [1 if score >= best_threshold else 0 for score in all_scores]
        human_correct = sum(pred == label for pred, label in zip(predictions[:len(self.human_scores)],
                                                                 all_labels[:len(self.human_scores)]))
        ai_correct = sum(pred == label for pred, label in zip(predictions[len(self.human_scores):],
                                                              all_labels[len(self.human_scores):]))

        tnr = human_correct / len(self.human_scores) if self.human_scores else 0
        tpr = ai_correct / len(self.ai_scores) if self.ai_scores else 0

        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 7))

        ax = plt.subplot(111)
        sns.histplot(self.human_scores, bins=15, alpha=0.5, label="Human Emails", color="#009E73", kde=True, ax=ax)
        sns.histplot(self.ai_scores, bins=15, alpha=0.5, label="AI Emails", color="#D55E00", kde=True, ax=ax)

        plt.axvline(x=best_threshold, color='black', linestyle='--', alpha=0.7,
                    label=f'Best Threshold: {best_threshold:.2f}')

        method = "Log-Ratio" if self.use_log_ratio else "Difference"
        plt.text(0.03, 0.95,
                 f"Parameters:\n"
                 f"Span Length: {self.span_length}\n"
                 f"Mask %: {self.pct_mask * 100:.0f}%\n"
                 f"Method: {method}\n"
                 f"Accuracy: {best_accuracy:.2f}\n"
                 f"Human Detection: {tnr:.2f}\n"
                 f"AI Detection: {tpr:.2f}",
                 transform=ax.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.xlabel(
            f"Score ({'log(orig_ll) - log(avg_perturbed_ll)' if self.use_log_ratio else 'orig_ll - avg_perturbed_ll'})",
            fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f"DetectGPT Scores: {self.ai_model_key.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        plt.legend(fontsize=12)
        plt.tight_layout()

        score_method = "logratio" if self.use_log_ratio else "diff"
        filename = f"{self.ai_model_key}_s{self.span_length}_p{int(self.pct_mask * 100)}_{score_method}.png"
        output_path = os.path.join(self.output_dir, filename)

        plt.savefig(output_path, dpi=300)
        self.logger.info(f"Plot saved to {output_path}")
        plt.close()

# ------------------------------------------------------------------
# Experiment Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    model_keys = [

        # Attr. Prompting
        "attr_prompting_claude",
        "attr_prompting_deepseek",
        "attr_prompting_gemini",
        "attr_prompting_gpt4",
        "attr_prompting_mistral",

        # BARE Llama3B
        "bare_llama3b_claude",
        "bare_llama3b_deepseek",
        "bare_llama3b_gemini",
        "bare_llama3b_gpt4",
        "bare_llama3b_mistral",
        "bare_llama8b_claude",

        # BARE Llama8B
        "bare_llama8b_claude",
        "bare_llama8b_deepseek",
        "bare_llama8b_gemini",
        "bare_llama8b_gpt4",
        "bare_llama8b_mistral",

        # Few Shot
        "fewshot_claude",
        "fewshot_deepseek",
        "fewshot_gemini",
        "fewshot_gpt4",
        "fewshot_mistral",

        # Zero Shot
        "zeroshot_claude",
        "zeroshot_deepseek",
        "zeroshot_gemini",
        "zeroshot_gpt4",
        "zeroshot_mistral"
    ]

    span_length = 3
    pct_mask = 0.15
    scoring_method = True
    n_samples = 100
    n_perturbations = 20

    main_logger = CustomLogger(name="Main")
    output_dir = os.path.join(os.getcwd(), "output", "detectgpt")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    main_logger.info("Loading email data...")
    data = load_local_email_data(number_of_samples=n_samples, buffer_size=10, logger=main_logger)
    human_emails = data["original"]

    main_logger.info("Processing human emails once...")
    human_results = process_human_emails(
        human_emails,
        n_perturbations=n_perturbations,
        span_length=span_length,
        pct_mask=pct_mask,
        use_log_ratio=scoring_method,
        logger=main_logger
    )

    human_results_file = os.path.join(data_dir, f"human_results_s{span_length}_p{int(pct_mask * 100)}.json")

    human_results_to_save = {
        "experiment_settings": {
            "n_samples": n_samples,
            "n_perturbations": n_perturbations,
            "span_length": span_length,
            "pct_mask": pct_mask,
            "scoring_method": "log_ratio" if scoring_method else "difference",
            "gpt2_model": gpt2_model_name,
            "t5_model": t5_model_name
        },
        "human_scores": human_results["scores"],
        "human_details": [{k: v for k, v in detail.items() if k != "perturbed_samples"}
                          for detail in human_results["details"]]
    }

    with open(human_results_file, 'w', encoding='utf-8') as f:
        json.dump(human_results_to_save, f, ensure_ascii=False, indent=2)
    main_logger.info(f"Human results saved to {human_results_file}")

    for model_key in model_keys:
        main_logger.info(f"Processing AI model: {model_key}")
        experiment = DetectGPT(
            n_samples=n_samples,
            n_perturbations=n_perturbations,
            ai_model_key=model_key,
            span_length=span_length,
            pct_mask=pct_mask,
            use_log_ratio=scoring_method,
            human_results=human_results
        )

        experiment.run_experiment()
        experiment.plot_results()
        experiment.plot_paper_style_results()

        torch.cuda.empty_cache()

    main_logger.info("All experiments completed successfully!")