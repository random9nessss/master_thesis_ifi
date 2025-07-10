import argparse
from detectgpt.detect_gpt import DetectGPT


def main():
    parser = argparse.ArgumentParser(description="Run DetectGPT Experiment")
    parser.add_argument("--n_perturbations", type=int, default=50, help="Number of perturbations per email")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of emails to sample (for human and AI)")
    parser.add_argument("--ai_model_key", type=str, default="attr_prompting_claude",
                        help="AI model key to select emails from default paths (e.g. 'attr_prompting_claude')")
    args = parser.parse_args()

    experiment = DetectGPT(n_samples=args.n_samples, n_perturbations=args.n_perturbations,
                           ai_model_key=args.ai_model_key)
    human_scores, ai_scores = experiment.run_experiment()
    experiment.plot_results()


if __name__ == "__main__":
    main()