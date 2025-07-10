import os
import json
import random
import argparse
import pandas as pd

from config.logger import CustomLogger
from analytics.lexical_diversity import LexicalDiversity
from analytics.semantic_diversity import SemanticDiversity
from analytics.syntactic_diversity import SyntacticDiversity
from analytics.readability_score import ReadabilityScore
from analytics.verbosity_analysis import VerbosityAnalysis
from analytics.sentiment_analysis import EnsembleSentimentAnalysis


random.seed(42)

model_email_paths = {
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

    # # ------------------------------------------------------------------
    # # Iterative Seeded BARE
    # # ------------------------------------------------------------------
    # "iterative_bare_claude": "../data/email_datasets/synthetic/iterativebaserefine/claude/aggregated.json",
    # "iterative_bare_deepseek": "",
    # "iterative_bare_gemini": "",
    # "iterative_bare_gpt4": "",
    # "iterative_bare_mistral": "../data/email_datasets/synthetic/iterativebaserefine/mistral/aggregated.json",
}

def main():
    parser = argparse.ArgumentParser(description="Compute textual diversity metrics for email data.")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="",
        help="Path to the JSON file containing email data. If left blank, all predefined model paths will be processed."
    )
    parser.add_argument(
        "--field", "-F",
        type=str,
        choices=["body", "subject"],
        default="body",
        help="Email field to analyze: 'body' or 'subject'"
    )
    parser.add_argument(
        "--sample_size", "-s",
        type=int,
        default=None,
        help="Number of emails to sample for analysis. If not provided, all emails will be used."
    )

    args = parser.parse_args()

    logger = CustomLogger(name="DiversityAnalytics")
    field = args.field
    sample_size = args.sample_size

    results = []

    if args.file:
        file_paths = {"custom": args.file}

    else:
        file_paths = model_email_paths

    for model_name, file_path in file_paths.items():
        logger.info(f"Processing file: {file_path} (Model: {model_name}) on field: {field}")
        result_entry = {"model": model_name, "file_path": file_path, "field": field}

        # ------------------------------------------------------------------
        # Data Sampling
        # ------------------------------------------------------------------
        file_to_process = file_path
        if sample_size is not None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if len(data) > sample_size:
                    data = random.sample(data, sample_size)

                temp_file_path = file_path.replace(".json", f"_sampled_{sample_size}.json")

                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                file_to_process = temp_file_path
                logger.info(f"Sampled {sample_size} emails from {file_path} into {temp_file_path}")

            except Exception as e:
                logger.error(f"Sampling failed for {model_name} at {file_path}: {e}")
                file_to_process = file_path

        logger.info(f"Processing file: {file_to_process} (Model: {model_name}) on field: {field}")
        result_entry = {"model": model_name, "file_path": file_to_process, "field": field}

        # -------------------------------------------------------------------
        # Semantic Diversity
        # -------------------------------------------------------------------
        try:
            semantic_div_obj = SemanticDiversity(file_to_process, field=field)
            semantic_div = semantic_div_obj.compute_overall_semantic_diversity()
            logger.info(f"Semantic Diversity: {semantic_div:.3f}")
            result_entry["semantic_diversity"] = semantic_div
        except Exception as e:
            logger.error(f"Semantic Diversity computation failed for {model_name}: {e}")
            result_entry["semantic_diversity"] = None

        # -------------------------------------------------------------------
        # Lexical Diversity
        # -------------------------------------------------------------------
        try:
            lexical_div_obj = LexicalDiversity(file_to_process, field=field)
            lexical_div = lexical_div_obj.compute_diversity_metrics()
            logger.info(f"Lexical Diversity: {lexical_div}")
            result_entry["lexical_diversity_ttr"] = lexical_div.get("normalized_ttr")
            result_entry["lexical_diversity_distinct_1"] = lexical_div.get("distinct_1")
            result_entry["lexical_diversity_distinct_2"] = lexical_div.get("distinct_2")
            result_entry["lexical_diversity_distinct_3"] = lexical_div.get("distinct_3")
        except Exception as e:
            logger.error(f"Lexical Diversity computation failed for {model_name}: {e}")
            result_entry["lexical_diversity"] = None

        # -------------------------------------------------------------------
        # Syntactic Diversity
        # -------------------------------------------------------------------
        try:
            syntactic_div_obj = SyntacticDiversity(file_to_process, field=field)
            syntactic_div = syntactic_div_obj.compute_overall_syntactic_diversity()
            logger.info(f"Syntactic Diversity: {syntactic_div:.3f}")
            result_entry["syntactic_diversity"] = syntactic_div
        except Exception as e:
            logger.error(f"Syntactic Diversity computation failed for {model_name}: {e}")
            result_entry["syntactic_diversity"] = None

        # -------------------------------------------------------------------
        # Readability Score
        # -------------------------------------------------------------------
        try:
            readability_score_obj = ReadabilityScore(file_to_process, field)
            readability_score = readability_score_obj.compute_readability()
            logger.info(f"Readability Score: {readability_score:.3f}")
            result_entry["readability_score"] = readability_score
        except Exception as e:
            logger.error(f"Readability Score computation failed for {model_name}: {e}")
            result_entry["readability_score"] = None

        # -------------------------------------------------------------------
        # Sentiment Analysis
        # -------------------------------------------------------------------
        try:
            sentiment_analysis_obj = EnsembleSentimentAnalysis(file_to_process, field=field)
            sentiment_scores = sentiment_analysis_obj.compute_sentiment()
            distribution_output_dir = os.path.join(os.getcwd(), "output", "sentiment_distribution")
            distribution_file = sentiment_analysis_obj.save_sentiment_distribution(distribution_output_dir)
            logger.info(f"Sentiment distribution saved to {distribution_file}")

            logger.info(f"Sentiment Score: {sentiment_scores}")
            result_entry["sentiment_neg"] = sentiment_scores.get("sentiment_neg")
            result_entry["sentiment_neu"] = sentiment_scores.get("sentiment_neu")
            result_entry["sentiment_pos"] = sentiment_scores.get("sentiment_pos")
            result_entry["sentiment_distribution_file"] = distribution_file

        except Exception as e:
            logger.error(f"Sentiment Score computation failed for {model_name}: {e}")
            result_entry["sentiment_neg"] = None
            result_entry["sentiment_neu"] = None
            result_entry["sentiment_pos"] = None
            result_entry["sentiment_distribution_file"] = None

        # -------------------------------------------------------------------
        # Verbosity Analysis
        # -------------------------------------------------------------------
        try:
            verbosity_analysis_obj = VerbosityAnalysis(file_to_process, field)
            verbosity_score = verbosity_analysis_obj.compute_verbosity()
            logger.info(f"Verbosity Score: {verbosity_score}")
            result_entry["avg_words_per_email"] = verbosity_score.get("avg_words_per_email")
            result_entry["avg_sentences_per_email"] = verbosity_score.get("avg_sentences_per_email")
            result_entry["avg_words_per_sentence"] = verbosity_score.get("avg_words_per_sentence")
        except Exception as e:
            logger.error(f"Verbosity Score computation failed for {model_name}: {e}")
            result_entry["avg_words_per_email"] = None
            result_entry["avg_sentences_per_email"] = None
            result_entry["avg_words_per_sentence"] = None

        results.append(result_entry)

    # -------------------------------------------------------------------
    # Output Directory and File Persistence
    # -------------------------------------------------------------------
    output_dir = os.path.join(os.getcwd(), "output", "diversity")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"diversity_results_{field}.csv")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Diversity results saved to {output_file}")

if __name__ == "__main__":
    main()