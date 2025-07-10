import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from huggingface_hub import configure_http_backend
import argparse
from typing import List

from gliner import GLiNERConfig, GLiNER
from gliner.training import TrainingArguments, Trainer
from gliner.data_processing.collator import DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter

from utils import (
    backend_factory,
    setup_environment
)
from data_loading import (
    load_training_data,
    extract_entity_type_names,
    extract_entity_types,
    filter_common_types,
    CustomGLiNERDataset
)
from evaluation import NERTrainer

# -------------------------------------------------------------------
# Model Initialization
# -------------------------------------------------------------------
def initialize_model(config: object) -> tuple:
    """
    Initialize GLiNER model, tokenizer, and configuration.

    Args:
        config: Configuration object containing model parameters

    Returns:
        tuple: (model, tokenizer, model_config)
    """
    print("Initializing model...")

    if hasattr(config, 'prev_path') and config.prev_path and config.prev_path != "null":
        print(f"Loading pretrained model from: {config.prev_path}")
        model = GLiNER.from_pretrained(config.prev_path)
        tokenizer = AutoTokenizer.from_pretrained(config.prev_path)
        model_config = model.config
    else:
        print("Initializing new GLiNER model")
        model_config = GLiNERConfig(**vars(config))

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name,
            model_max_length=model_config.max_len,
            truncation_side="right",
            add_prefix_space=True
        )

        words_splitter_type = getattr(config, 'words_splitter_type', 'whitespace')
        words_splitter = WordsSplitter(words_splitter_type)
        model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)

        if not getattr(config, 'labels_encoder', False):
            model_config.class_token_index = len(tokenizer)
            tokenizer.add_tokens(
                [model_config.ent_token, model_config.sep_token],
                special_tokens=True
            )

            model_config.vocab_size = len(tokenizer)
            model.resize_token_embeddings(
                [model_config.ent_token, model_config.sep_token],
                set_class_token_index=False,
                add_tokens_to_tokenizer=False
            )

    print(f"Model initialized: {model_config.model_name}")
    print(f"-Vocabulary size: {len(tokenizer)}")
    print(f"-Max length: {model_config.max_len}")
    print(f"-Hidden size: {model_config.hidden_size}")
    print(f"-Span mode: {getattr(model_config, 'span_mode', 'default')}")

    return model, tokenizer, model_config

# -------------------------------------------------------------------
# Dataset Preparation
# -------------------------------------------------------------------
def prepare_datasets(train_data: list, model: GLiNER) -> tuple:
    """
    Prepare training dataset and data collator.

    Args:
        train_data: List of training examples
        model: GLiNER model instance

    Returns:
        tuple: (train_dataset, data_collator)
    """
    print("Preparing datasets...")

    train_dataset = CustomGLiNERDataset(train_data)
    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )

    print(f"Dataset prepared: {len(train_dataset)} examples")

    return train_dataset, data_collator

# -------------------------------------------------------------------
# Training Arguments Loader
# -------------------------------------------------------------------
def create_training_arguments(config: object) -> TrainingArguments:
    """
    Create training arguments from configuration.

    Args:
        config: Configuration object

    Returns:
        TrainingArguments: Configured training arguments
    """
    print("Creating training arguments...")

    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

    effective_batch_size = config.train_batch_size * gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir=config.log_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        focal_loss_gamma=float(config.loss_gamma),
        focal_loss_alpha=float(config.loss_alpha),
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.num_steps,
        save_steps=config.eval_every,
        save_total_limit=config.save_total_limit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,
        use_cpu=False,
        bf16=True,
        logging_steps=100,
        metric_for_best_model="f1",
        greater_is_better=True,
        eval_steps=config.eval_every,
        report_to="wandb",
    )

    print(f"Training arguments created:")
    print(f"-Batch size: {config.train_batch_size}")
    print(f"-Gradient accumulation: {gradient_accumulation_steps}")
    print(f"-Effective batch size: {effective_batch_size}")
    print(f"-Learning rate (encoder): {config.lr_encoder}")
    print(f"-Learning rate (others): {config.lr_others}")
    print(f"-Weight decay (encoder): {config.weight_decay_encoder}")
    print(f"-Weight decay (others): {config.weight_decay_other}")
    print(f"-Focal loss gamma: {config.loss_gamma}")
    print(f"-Focal loss alpha: {config.loss_alpha}")
    print(f"-Label smoothing: {getattr(config, 'label_smoothing', 0.0)}")
    print(f"-Max steps: {config.num_steps}")
    print(f"-Eval every: {config.eval_every} steps")
    print(f"-Scheduler: {config.scheduler_type}")
    print(f"-Warmup ratio: {config.warmup_ratio}")

    return training_args


def main():
    """
    Main training function for simple GLiNER training (focal loss only).

    This function orchestrates the entire training process:
    1. Load configuration and setup environment
    2. Load and prepare training data
    3. Initialize model and training components
    4. Create trainer and run training
    5. Save results
    """
    parser = argparse.ArgumentParser(description="Train GLiNER with focal loss only")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    setup_environment()
    configure_http_backend(backend_factory=backend_factory)

    config = load_config_as_namespace(args.config)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("\n" + "=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Model: {config.model_name}")

    print("\nLoading training data...")
    train_data = load_training_data(config)
    print(f"Loaded {len(train_data)} training examples")

    min_examples = getattr(config, 'min_entity_examples', 500)
    if min_examples > 0:
        print(f"\nFiltering dataset to focus on common entities (≥{min_examples} examples)...")
        train_data = filter_common_types(train_data, min_examples=min_examples)

        if not train_data:
            print("No training data remaining after filtering!")
            return None

    entity_type_names = extract_entity_type_names(train_data)
    entity_types = extract_entity_types(train_data)

    model, tokenizer, model_config = initialize_model(config)
    model = model.to(device)

    train_dataset, data_collator = prepare_datasets(train_data, model)
    training_args = create_training_arguments(config)

    print("\nCreating basic GLiNER trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    eval_dir = getattr(config, 'val_data_dir', None)
    if eval_dir and os.path.exists(eval_dir):
        print(f"\nSetting up evaluation on datasets in {eval_dir}")

        ner_trainer = NERTrainer(
            base_trainer=trainer,
            eval_dir=eval_dir,
            entity_types=entity_types,
            eval_steps=config.eval_every,
            eval_sample_size=getattr(config, 'eval_sample_size', 5000),
            early_stopping_patience=config.early_stopping_patience,
            use_threshold_scheduler=config.use_threshold_scheduler,
            threshold_scheduler_config={
                'initial_threshold': config.threshold_initial,
                'final_threshold': config.threshold_final,
                'warmup_steps': config.threshold_warmup_steps,
                'schedule_type': config.threshold_schedule_type
            },
            fixed_evaluation_threshold=config.fixed_evaluation_threshold,
            evaluation_strategy=getattr(config, 'evaluation_strategy', 'all')
        )

        trainer_to_use = ner_trainer
        print("Evaluation setup complete")
        print(f"-Early stopping patience: {config.early_stopping_patience}")
        print(f"-Threshold scheduler: {'ENABLED' if config.use_threshold_scheduler else 'DISABLED'}")
        if config.use_threshold_scheduler:
            print(
                f"-Threshold: {config.threshold_initial:.3f} → {config.threshold_final:.3f} over {config.threshold_warmup_steps} steps")
        else:
            print(f"-Fixed threshold: {config.fixed_evaluation_threshold}")
    else:
        trainer_to_use = trainer
        print("No evaluation directory provided, training without external evaluation")

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Training steps: {config.num_steps}")
    print(f"Batch size: {config.train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Evaluation every: {config.eval_every} steps")
    if eval_dir:
        print(f"Evaluation directory: {eval_dir}")
        print(f"Early stopping patience: {config.early_stopping_patience}")
    print("=" * 60 + "\n")

    try:
        if hasattr(trainer_to_use, 'train'):
            training_result = trainer_to_use.train()
        else:
            training_result = trainer.train()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total training steps: {trainer.state.global_step}")

        final_model_path = os.path.join(config.log_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        if hasattr(trainer_to_use, 'global_best_f1'):
            print(f"\nTraining Summary:")
            print(f"   Best F1 achieved: {trainer_to_use.global_best_f1:.4f}")

            if hasattr(trainer_to_use, 'dataset_best_f1'):
                print(f"-Best F1 per dataset:")
                for dataset, f1 in sorted(trainer_to_use.dataset_best_f1.items(), key=lambda x: x[1], reverse=True):
                    print(f"{dataset}: {f1:.4f}")

        return training_result

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 60)
        print(f"Current step: {trainer.state.global_step}")

        try:
            interrupted_save_path = os.path.join(config.log_dir, "interrupted_checkpoint")
            trainer.save_model(interrupted_save_path)
            tokenizer.save_pretrained(interrupted_save_path)
            print(f"Model saved to: {interrupted_save_path}")
        except Exception as e:
            print(f"Could not save interrupted checkpoint: {e}")

        return None

    except Exception as e:
        print(f"\n" + "=" * 60)
        print("TRAINING FAILED WITH ERROR")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        try:
            emergency_save_path = os.path.join(config.log_dir, "emergency_checkpoint")
            trainer.save_model(emergency_save_path)
            tokenizer.save_pretrained(emergency_save_path)
            print(f"\nEmergency checkpoint saved to: {emergency_save_path}")
        except Exception as save_error:
            print(f"\nCould not save emergency checkpoint: {save_error}")

        return None

if __name__ == '__main__':
    main()