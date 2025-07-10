import os
import torch
import numpy as np
import logging
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from mteb import MTEB
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deberta_evaluation")


# ----------------------------------------
# DeBERTa Embedding Model for MTEB
# ----------------------------------------
class DeBERTaEmbedder:
    def __init__(self, model_path):
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32, normalize_embeddings=True, **kwargs):
        """
        Returns embeddings for the given sentences
        """
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch = [sent if sent.strip() else " " for sent in batch]

            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input, output_hidden_states=True)
                embeddings = model_output.last_hidden_state[:, 0, :]
                embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)

        if normalize_embeddings:
            all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

        return all_embeddings


# ----------------------------------------
# Result Extraction
# ----------------------------------------
def safe_extract_score(results, task, split="test", metric="spearman"):
    """
    Safely extract scores from MTEB results with fallback handling
    """
    try:
        # list case
        if isinstance(results, list):
            for result in results:
                if result.get("task_name") == task:
                    if "scores" in result:
                        return result["scores"][split][metric]
                    elif split in result:
                        return result[split][metric]

        # dict case
        elif isinstance(results, dict):
            if task in results:
                task_result = results[task]
                if split in task_result:
                    split_result = task_result[split]
                    if isinstance(split_result, dict):
                        if "default" in split_result:
                            return split_result["default"].get(metric)
                        elif metric in split_result:
                            return split_result[metric]
                        else:
                            for key, value in split_result.items():
                                if isinstance(value, dict) and metric in value:
                                    return value[metric]
        return None
    except Exception as e:
        logger.warning(f"Could not extract {metric} for {task}: {e}")
        return None


# ----------------------------------------
# Evaluation Function
# ----------------------------------------
def evaluate_models_comprehensive(models_dict, output_dir="evaluation_results"):
    """
    Evaluate and compare multiple models comprehensively

    Args:
        models_dict: Dict with model names as keys and model paths as values
                    e.g., {"base": "microsoft/deberta-v3-small",
                           "contrastive_bge": "path/to/bge",
                           "contrastive_sbert": "path/to/sbert"}
    """
    os.makedirs(output_dir, exist_ok=True)

    loaded_models = {}
    for name, path in models_dict.items():
        loaded_models[name] = DeBERTaEmbedder(path)

    # ----------------------------------------
    # Task Selection
    # ----------------------------------------

    # Semantic Similarity Tasks (Core)
    sts_tasks = [
        "STSBenchmark",
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS16",
        "STS22"
    ]

    # Classification
    classification_tasks = [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification"
    ]

    # Clustering
    clustering_tasks = [
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "RedditClustering",
        "RedditClusteringP2P",
        "StackExchangeClustering",
        "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering"
    ]

    # Retrieval
    retrieval_tasks = [
        "FiQA2018",
        "ArguAna",
        "ClimateFEVER",
        "CQADupstackRetrieval",
        "DBPedia",
        "FEVER",
        "HotpotQA",
        "MSMARCO",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "SCIDOCS",
        "SciFact",
        "Touche2020",
        "TRECCOVID"
    ]

    # Reranking
    reranking_tasks = [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions"
    ]

    # Selected Task Subset
    all_task_groups = {
        "STS": sts_tasks,
        "Classification": ["AmazonCounterfactualClassification", "Banking77Classification"],
        "Clustering": ["TwentyNewsgroupsClustering"],
        "Retrieval": ["FiQA2018"],
        "Reranking": ["AskUbuntuDupQuestions"]
    }

    results_summary = {}

    # ----------------------------------------
    # Run Evaluations
    # ----------------------------------------
    for task_group, tasks in all_task_groups.items():
        logger.info("=" * 80)
        logger.info(f"Evaluating {task_group} tasks: {tasks}")
        logger.info("=" * 80)

        group_results = {}

        try:
            evaluation = MTEB(tasks=tasks, task_langs=["en"])

            for model_name, model in loaded_models.items():
                logger.info(f"Evaluating {model_name} on {task_group} tasks...")

                try:
                    model_results = evaluation.run(
                        model,
                        output_folder=os.path.join(output_dir, model_name, task_group.lower()),
                        eval_splits=["test"],
                        overwrite_results=True
                    )
                    group_results[model_name] = model_results

                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {task_group}: {e}")
                    group_results[model_name] = None

        except Exception as e:
            logger.error(f"Error setting up {task_group} evaluation: {e}")
            continue

        results_summary[task_group] = group_results

    # ----------------------------------------
    # Generate Results Table
    # ----------------------------------------
    return generate_results_table(results_summary, all_task_groups)


def generate_results_table(results_summary, task_groups):
    """
    Generate a comprehensive results table
    """
    logger.info("=" * 80)
    logger.info("GENERATING RESULTS TABLE")
    logger.info("=" * 80)

    table_data = []

    for task_group, tasks in task_groups.items():
        if task_group not in results_summary:
            continue

        group_results = results_summary[task_group]
        model_names = [name for name in group_results.keys() if group_results[name] is not None]

        if not model_names:
            continue

        for task in tasks:
            row = {"Task Group": task_group, "Task": task}

            for model_name in model_names:
                model_results = group_results[model_name]

                if task_group == "STS":
                    metric = "spearman"

                elif task_group == "Classification":
                    metric = "accuracy"

                elif task_group == "Clustering":
                    metric = "v_measure"

                elif task_group in ["Retrieval", "Reranking"]:
                    metric = "ndcg_at_10"

                else:
                    metric = "main_score"

                score = safe_extract_score(model_results, task, metric=metric)
                row[model_name] = f"{score:.4f}" if score is not None else "N/A"

            table_data.append(row)

    if table_data:
        df = pd.DataFrame(table_data)

        if "base" in df.columns:
            for col in df.columns:
                if col not in ["Task Group", "Task", "base"] and col in df.columns:
                    improvement_col = f"{col} vs Base"
                    df[improvement_col] = df.apply(
                        lambda row: calculate_improvement(row["base"], row[col]), axis=1
                    )

        csv_path = "mteb_comprehensive_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        print("\n" + "=" * 100)
        print("COMPREHENSIVE MTEB EVALUATION RESULTS")
        print("=" * 100)
        print(df.to_string(index=False))

        return df
    else:
        logger.warning("No results to display")
        return None


def calculate_improvement(base_score, model_score):
    """Calculate improvement with error handling"""
    try:
        if base_score == "N/A" or model_score == "N/A":
            return "N/A"
        base_val = float(base_score)
        model_val = float(model_score)
        improvement = model_val - base_val
        return f"{improvement:+.4f}"
    except:
        return "N/A"

# ----------------------------------------
# Main Execution
# ----------------------------------------
if __name__ == "__main__":

    MODELS = {
        "base": "microsoft/deberta-v3-small",
        "contrastive_sbert": "../../models/base/sbert",
        "contrastive_bge": "../../models/base/bge",
        "simcse": "../../models/base/simcse",
        "mlm": "../../models/base/mlm"
    }

    logger.info("Starting comprehensive MTEB evaluation...")
    logger.info(f"Models to evaluate: {list(MODELS.keys())}")

    logger.info("Running comprehensive MTEB evaluation...")
    mteb_results = evaluate_models_comprehensive(MODELS)

    logger.info("All evaluations complete!")