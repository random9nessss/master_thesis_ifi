# -------------------------------------------------------------------
#
# QWEN Finetuning (2.5-0.5B-Instruct) with LoRA
#
# -------------------------------------------------------------------

import platform
import sys

_original_python_version = platform.python_version
def clean_python_version():
    version = _original_python_version()
    return version.split('+')[0]

platform.python_version = clean_python_version

import argparse
import json
import os
import sys
import torch
import re
import warnings
from typing import Optional, Dict

import requests
from huggingface_hub import configure_http_backend


# -------------------------------------------------------------------
# Backend Factory
# -------------------------------------------------------------------
def backend_factory() -> requests.Session:
    """Create a Requests session that disables SSL verification."""
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MODEL_NAME = "llm_finetuning/qwen"
OUTPUT_DIR = "qwen2-5-0.5b-maritime-standard-peft"
MAX_SEQ_LENGTH = 1024

# -------------------------------------------------------------------
# LoRA Adapter Configuration
# -------------------------------------------------------------------
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.1
TARGET_MODULES = ["q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj",
                  "gate_proj",
                  "up_proj",
                  "down_proj"]

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO = 2e-5, 3, 0.1
LOGGING_STEPS = 10


# -------------------------------------------------------------------
# Prompting, Data Formatting, and JSON Extraction
# -------------------------------------------------------------------
def get_role_based_prompt(sentence: str) -> str:
    """Generates the direct role-based prompt for extraction."""
    return f"""
      You are a shipping data extractor. Extract shipping information from the text.

      Text: {sentence}

      Return a JSON object with:
      - vessel (ship name)
      - commodity (cargo type)
      - incoterm (trade term like CIF, FOB)
      - locations (list of places)

      JSON:
    """


# -------------------------------------------------------------------
# Formatting Training Examples
# -------------------------------------------------------------------
def format_training_example(example: dict, tokenizer) -> dict:
    """Format a single training example for supervised fine-tuning."""
    instruction = get_role_based_prompt(example["input"])
    entities = {"vessel": None, "commodity": None, "incoterm": None, "locations": []}

    for annotation in example.get("output", []):
        if " <> " in annotation:
            text, type = annotation.split(" <> ", 1)
            type = type.lower().strip()
            if type == "vessel":
                entities["vessel"] = text
            elif type == "commodity":
                entities["commodity"] = text
            elif type == "incoterm":
                entities["incoterm"] = text
            elif type == "location":
                entities["locations"].append(text)

    response = json.dumps(entities, ensure_ascii=False)
    full_text = instruction + response + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_tensors=None
    )

    tokenized["labels"] = list(tokenized["input_ids"])

    return tokenized


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen model with LoRA on shipping data')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to JSON file containing training data')
    parser.add_argument('--model_path', type=str, default=MODEL_NAME,
                        help='Path to the base model directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for the fine-tuned model')
    parser.add_argument('--max_seq_length', type=int, default=MAX_SEQ_LENGTH,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lora_r', type=int, default=LORA_R,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=LORA_ALPHA,
                        help='LoRA alpha')

    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} does not exist")
        sys.exit(1)

    print("=" * 80)
    print("   QWEN FINETUNING WITH UPSCALED LORA ADAPTER")
    print("=" * 80)
    print(f"Data file: {args.data_file}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, dropout={LORA_DROPOUT}")
    print("=" * 80)

    print(f"\nLoading model: {args.model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")
    print(f"Device: {device}")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True
        )

        print("Model and tokenizer loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading approach...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            model = model.to(device)
            print("Model loaded with alternative approach")

        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            print("Trying with force_download=True to clear cache...")

            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, force_download=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    force_download=True
                )
                model = model.to(device)
                print("Model loaded after clearing cache")

            except Exception as e3:
                print(f"All loading attempts failed: {e3}")
                return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer `pad_token` set to `eos_token`.")

    tokenizer.padding_side = "right"
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")

    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    print("\n" + "=" * 50)
    print("CONFIGURING UPSCALED PEFT (LoRA) FOR TRAINING")
    print("=" * 50)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("\n" + "=" * 50)
    print("LOADING AND PREPARING DATASET")
    print("=" * 50)

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training examples")

    dataset = Dataset.from_list(data)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: format_training_example(x, tokenizer),
        num_proc=4,
        remove_columns=dataset.column_names
    )

    print("Checking tokenized data format...")
    first_example = tokenized_dataset[0]
    print(f"Keys: {first_example.keys()}")
    print(f"input_ids type: {type(first_example['input_ids'])}, length: {len(first_example['input_ids'])}")
    print(f"labels type: {type(first_example['labels'])}, length: {len(first_example['labels'])}")
    print(f"First 10 input_ids: {first_example['input_ids'][:10]}")
    print(f"First 10 labels: {first_example['labels'][:10]}")

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_data, eval_data = split_dataset["train"], split_dataset["test"]

    print(f"Training examples: {len(train_data)}")
    print(f"Evaluation examples: {len(eval_data)}")

    print("\n" + "=" * 30)
    print("SETTING UP DATA COLLATOR")
    print("=" * 30)

    print("Using DefaultDataCollator for better compatibility...")
    data_collator = DefaultDataCollator(return_tensors="pt")

    print("\nTesting data collator...")
    try:
        test_batch = [train_data[0], train_data[1]]
        collated = data_collator(test_batch)
        print(f"Data collator test successful")
        print(f"Batch keys: {collated.keys()}")
        print(f"input_ids shape: {collated['input_ids'].shape}")
        print(f"labels shape: {collated['labels'].shape}")
    except Exception as e:
        print(f"DefaultDataCollator failed: {e}")
        print("Creating custom data collator...")

        def custom_data_collator(features):
            """Custom data collator for our specific format."""
            import torch

            max_length = max(len(f['input_ids']) for f in features)

            batch = {}
            batch_size = len(features)

            batch['input_ids'] = torch.full((batch_size, max_length), tokenizer.pad_token_id, dtype=torch.long)
            batch['attention_mask'] = torch.zeros((batch_size, max_length), dtype=torch.long)
            batch['labels'] = torch.full((batch_size, max_length), -100, dtype=torch.long)

            for i, feature in enumerate(features):
                seq_len = len(feature['input_ids'])
                batch['input_ids'][i, :seq_len] = torch.tensor(feature['input_ids'])
                batch['attention_mask'][i, :seq_len] = torch.tensor(feature['attention_mask'])
                batch['labels'][i, :seq_len] = torch.tensor(feature['labels'])

            return batch

        data_collator = custom_data_collator

        try:
            test_batch = [train_data[0], train_data[1]]
            collated = data_collator(test_batch)
            print(f"Custom data collator test successful")
            print(f"Batch keys: {collated.keys()}")
            print(f"input_ids shape: {collated['input_ids'].shape}")
            print(f"labels shape: {collated['labels'].shape}")
        except Exception as e:
            print(f"Custom data collator also failed: {e}")
            return

    print("\n" + "=" * 50)
    print("CONFIGURING TRAINER")
    print("=" * 50)

    total_steps = (len(train_data) // (args.batch_size * GRADIENT_ACCUMULATION_STEPS)) * args.num_epochs
    eval_steps = max(50, total_steps // 20)
    save_steps = eval_steps

    print(f"Total training steps: {total_steps}")
    print(f"Evaluation every {eval_steps} steps")

    # -------------------------------------------------------------------
    # Trainer Configuration
    # -------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=args.learning_rate,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            logging_steps=LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            weight_decay=0.01,
            max_grad_norm=1.0,
        ),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING WITH UPSCALED LORA")
    print("=" * 80)
    print(f"Training on {len(train_data)} examples")
    print(f"Evaluating on {len(eval_data)} examples")
    print(f"LoRA parameters: r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED - SAVING MODEL")
    print("=" * 80)

    final_model_path = f"{args.output_dir}-final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to {final_model_path}")

    training_info = {
        "model_path": args.model_path,
        "data_file": args.data_file,
        "training_samples": len(train_data),
        "eval_samples": len(eval_data),
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES
        },
        "training_args": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_seq_length": args.max_seq_length
        }
    }

    with open(f"{final_model_path}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"Training info saved to {final_model_path}/training_info.json")

    print("\n" + "=" * 80)
    print("FINETUNING COMPLETE!")
    print("=" * 80)
    print(f"Final model location: {final_model_path}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    main()