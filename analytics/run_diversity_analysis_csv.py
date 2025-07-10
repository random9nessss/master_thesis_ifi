import os
import random
import argparse
import pandas as pd
from typing import List, Dict

from analytics.lexical_diversity import LexicalDiversityCSV
from analytics.semantic_diversity import SemanticDiversityCSV
from analytics.verbosity_analysis import VerbosityAnalysisCSV
from analytics.sentiment_analysis import EnsembleSentimentAnalysisCSV

random.seed(42)

from config.logger import CustomLogger
logger = CustomLogger("CSVAnalytics")


def sample_csv(file_path: str, sample_size: int, content_column: str = "parsed_content") -> str:
    """
    Sample rows from a CSV file and save to a new file.

    Args:
        file_path (str): Path to the original CSV file
        sample_size (int): Number of rows to sample
        content_column (str): The column containing text to analyze

    Returns:
        str: Path to the sampled CSV file
    """
    try:
        df = pd.read_csv(file_path)

        if content_column not in df.columns:
            logger.error(f"Column '{content_column}' not found in {file_path}")
            return file_path

        df = df[df[content_column].notna() & df[content_column].str.strip().astype(bool)]

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

        temp_file_path = file_path.replace(".csv", f"_sampled_{sample_size}.csv")

        df.to_csv(temp_file_path, index=False)
        logger.info(f"Sampled {len(df)} rows from {file_path} into {temp_file_path}")

        return temp_file_path

    except Exception as e:
        logger.error(f"Sampling failed for {file_path}: {e}")
        return file_path


def analyze_csv(
        file_path: str,
        content_column: str = "parsed_content",
        sample_size: int = None,
        group_by: str = None
) -> Dict:
    """
    Analyze a CSV file with multiple metrics.

    Args:
        file_path (str): Path to the CSV file to analyze
        content_column (str): Name of the column containing the text to analyze
        sample_size (int, optional): Number of rows to sample for analysis
        group_by (str, optional): Column to group results by

    Returns:
        Dict: Dictionary containing analysis results
    """
    results = {"file_path": file_path, "content_column": content_column}

    file_to_process = file_path
    if sample_size is not None:
        file_to_process = sample_csv(file_path, sample_size, content_column)
        results["sample_size"] = sample_size

    output_base_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_base_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Lexical Diversity
    # -------------------------------------------------------------------
    try:
        logger.info(f"Computing lexical diversity for {file_to_process}")
        lexical_div_obj = LexicalDiversityCSV(file_to_process, content_column=content_column)
        lexical_div = lexical_div_obj.compute_diversity_metrics()

        results["lexical_diversity_ttr"] = lexical_div.get("normalized_ttr")
        results["lexical_diversity_distinct_1"] = lexical_div.get("distinct_1")
        results["lexical_diversity_distinct_2"] = lexical_div.get("distinct_2")
        results["lexical_diversity_distinct_3"] = lexical_div.get("distinct_3")
        logger.info(f"Lexical Diversity: TTR={lexical_div.get('normalized_ttr'):.3f}, " +
                    f"Distinct-1={lexical_div.get('distinct_1'):.3f}, " +
                    f"Distinct-2={lexical_div.get('distinct_2'):.3f}, " +
                    f"Distinct-3={lexical_div.get('distinct_3'):.3f}")

    except Exception as e:
        logger.error(f"Lexical Diversity computation failed: {e}")
        results["lexical_diversity_ttr"] = None
        results["lexical_diversity_distinct_1"] = None
        results["lexical_diversity_distinct_2"] = None
        results["lexical_diversity_distinct_3"] = None

    # -------------------------------------------------------------------
    # Semantic Diversity
    # -------------------------------------------------------------------
    try:
        logger.info(f"Computing semantic diversity for {file_to_process}")
        semantic_div_obj = SemanticDiversityCSV(file_to_process, content_column=content_column)
        semantic_div = semantic_div_obj.compute_overall_semantic_diversity()

        results["semantic_diversity"] = semantic_div
        logger.info(f"Semantic Diversity: {semantic_div:.3f}")

    except Exception as e:
        logger.error(f"Semantic Diversity computation failed: {e}")
        results["semantic_diversity"] = None

    # -------------------------------------------------------------------
    # Sentiment Analysis
    # -------------------------------------------------------------------
    try:
        logger.info(f"Computing sentiment analysis for {file_to_process}")
        sentiment_obj = EnsembleSentimentAnalysisCSV(file_to_process, content_column=content_column)
        sentiment_scores = sentiment_obj.compute_sentiment()

        distribution_output_dir = os.path.join(output_base_dir, "sentiment_distribution")
        os.makedirs(distribution_output_dir, exist_ok=True)
        distribution_file = sentiment_obj.save_sentiment_distribution(distribution_output_dir)

        results["sentiment_neg"] = sentiment_scores.get("sentiment_neg")
        results["sentiment_neu"] = sentiment_scores.get("sentiment_neu")
        results["sentiment_pos"] = sentiment_scores.get("sentiment_pos")
        results["sentiment_distribution_file"] = distribution_file

        logger.info(f"Sentiment Scores: Negative={sentiment_scores.get('sentiment_neg'):.3f}, " +
                    f"Neutral={sentiment_scores.get('sentiment_neu'):.3f}, " +
                    f"Positive={sentiment_scores.get('sentiment_pos'):.3f}")
        logger.info(f"Sentiment distribution saved to {distribution_file}")

    except Exception as e:
        logger.error(f"Sentiment Analysis computation failed: {e}")
        results["sentiment_neg"] = None
        results["sentiment_neu"] = None
        results["sentiment_pos"] = None
        results["sentiment_distribution_file"] = None

    # -------------------------------------------------------------------
    # Verbosity Analysis
    # -------------------------------------------------------------------
    try:
        logger.info(f"Computing verbosity metrics for {file_to_process}")
        verbosity_obj = VerbosityAnalysisCSV(file_to_process, content_column=content_column)
        verbosity_metrics = verbosity_obj.compute_verbosity()

        results["avg_words_per_document"] = verbosity_metrics.get("avg_words_per_document")
        results["avg_sentences_per_document"] = verbosity_metrics.get("avg_sentences_per_document")
        results["avg_words_per_sentence"] = verbosity_metrics.get("avg_words_per_sentence")
        results["total_documents"] = verbosity_metrics.get("total_documents")

        logger.info(f"Verbosity Metrics: " +
                    f"Words/Doc={verbosity_metrics.get('avg_words_per_document'):.2f}, " +
                    f"Sentences/Doc={verbosity_metrics.get('avg_sentences_per_document'):.2f}, " +
                    f"Words/Sentence={verbosity_metrics.get('avg_words_per_sentence'):.2f}")

        if group_by:
            try:
                group_metrics = verbosity_obj.compute_verbosity_by_group(group_by)
                group_results_file = os.path.join(
                    output_base_dir,
                    "verbosity",
                    f"verbosity_by_{group_by}_{os.path.basename(file_to_process).replace('.csv', '.json')}"
                )
                os.makedirs(os.path.dirname(group_results_file), exist_ok=True)

                group_df = pd.DataFrame.from_dict(group_metrics, orient='index')
                group_df.to_csv(group_results_file.replace('.json', '.csv'))

                results[f"verbosity_by_{group_by}_file"] = group_results_file.replace('.json', '.csv')
                logger.info(f"Verbosity metrics by {group_by} saved to {group_results_file.replace('.json', '.csv')}")
            except Exception as e:
                logger.error(f"Verbosity grouping by {group_by} failed: {e}")

    except Exception as e:
        logger.error(f"Verbosity computation failed: {e}")
        results["avg_words_per_document"] = None
        results["avg_sentences_per_document"] = None
        results["avg_words_per_sentence"] = None
        results["total_documents"] = None

    return results


def main():
    """Main function to parse arguments and run the analysis"""
    parser = argparse.ArgumentParser(description="Analyze text content in a CSV file.")
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to the CSV file containing text data."
    )
    parser.add_argument(
        "--column", "-c",
        type=str,
        default="parsed_content",
        help="Name of the column containing the text content to analyze."
    )
    parser.add_argument(
        "--sample_size", "-s",
        type=int,
        default=None,
        help="Number of rows to sample for analysis. If not provided, all rows will be used."
    )
    parser.add_argument(
        "--group_by", "-g",
        type=str,
        default=None,
        help="Column to group results by (for applicable metrics)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="text_analytics_results.csv",
        help="Output file name for the results."
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return

    try:
        logger.info(f"Starting analysis of {args.file}")
        results = analyze_csv(
            file_path=args.file,
            content_column=args.column,
            sample_size=args.sample_size,
            group_by=args.group_by
        )

        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, args.output)

        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)

        logger.info(f"Analysis complete. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()