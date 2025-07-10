import argparse
import sys

from bare.refine_mistral import MistralRefine
from bare.refine_google import GoogleRefine
from bare.refine_openai import OpenAIRefine
from bare.refine_anthropic import AnthropicRefine
from bare.refine_deepseek import DeepseekRefine

def main():
    parser = argparse.ArgumentParser(
        description="Run the refine process for a selected model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use (choose from 'mistral', 'anthropic', 'google', 'openai', 'deepseek')"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="Path to the input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--batch_file",
        type=str,
        required=True,
        help="(Temporary) Batch file name to be used in the refine process"
    )
    parser.add_argument(
        "--number_samples",
        type=int,
        required=False,
        help="Used to restart interrupted tasks. Number of samples will be collected from the tail of the dataframe"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=False,
        help="Indication of model that was used to generate the base chains"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Model Mapping
    # -------------------------------------------------------------------
    refine_mapping = {
        "mistral": MistralRefine,
        "google": GoogleRefine,
        "openai": OpenAIRefine,
        "deepseek": DeepseekRefine,
        "anthropic": AnthropicRefine
    }

    model_key = args.model.lower()
    if model_key not in refine_mapping or refine_mapping[model_key] is None:
        print(f"Error: Model '{args.model}' is not supported.")
        supported = [k for k, v in refine_mapping.items() if v is not None]
        print(f"Supported models: {', '.join(supported)}")
        sys.exit(1)

    if (model_key == "openai" or
        model_key == "anthropic" or
        model_key == "google" or
        model_key == "deepseek") and args.base_model:
        refine_client = refine_mapping[model_key](args.base_model)
    else:
        refine_client = refine_mapping[model_key]()

    if (model_key == "openai"
        or model_key == "anthropic"
        or model_key == "google"
        or model_key == "deepseek") and args.number_samples:
        refine_client.run_refinement(args.input_dataset, args.batch_file, args.number_samples)
    else:
        refine_client.run_refinement(args.input_dataset, args.batch_file)

if __name__ == "__main__":
    main()