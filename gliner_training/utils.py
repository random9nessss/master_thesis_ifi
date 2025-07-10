# utils

import os
import math
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any


# =============================================================================
# Environment Setup
# =============================================================================

def setup_environment() -> None:
    """
    Set up the training environment with optimal settings.

    Configures CUDA devices, tokenizer parallelism, and memory allocation
    for optimal training performance.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("Environment configured:")
    print("- Tokenizer parallelism: enabled")
    print("- Memory allocation: expandable segments")

    backend_factory()


def backend_factory() -> requests.Session:
    """
    Create a requests session with SSL verification disabled.

    Returns:
        requests.Session: Configured session for HuggingFace Hub

    This is needed to avoid SSL certificate issues when downloading
    models from HuggingFace Hub in some environments.
    """
    session = requests.Session()
    session.verify = False
    return session


# =============================================================================
# Prediction Threshold Scheduling
# =============================================================================

class PredictionThresholdScheduler:
    """
    Scheduler for dynamically adjusting prediction threshold during training.

    This scheduler gradually increases the prediction threshold during training,
    starting with a low threshold to capture more entities early in training,
    then increasing to be more selective as the model improves.

    Args:
        initial_threshold (float): Starting threshold value
        final_threshold (float): Final threshold value
        warmup_steps (int): Number of steps to reach final threshold
        schedule_type (str): Type of schedule ('linear', 'cosine', 'exponential')
    """

    def __init__(
            self,
            initial_threshold: float = 0.1,
            final_threshold: float = 0.5,
            warmup_steps: int = 1000,
            schedule_type: str = 'linear'
    ):
        """Initialize the threshold scheduler."""
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type

        print(f"Threshold scheduler: {initial_threshold:.2f} → {final_threshold:.2f} "
              f"over {warmup_steps} steps ({schedule_type})")

    def get_threshold(self, step: int) -> float:
        """
        Get the prediction threshold for the current training step.

        Args:
            step (int): Current training step

        Returns:
            float: Prediction threshold for this step
        """
        if self.warmup_steps == 0 or step >= self.warmup_steps:
            return self.final_threshold

        progress = step / self.warmup_steps

        if self.schedule_type == 'linear':
            threshold = (self.initial_threshold +
                         (self.final_threshold - self.initial_threshold) * progress)

        elif self.schedule_type == 'cosine':
            threshold = (self.initial_threshold +
                         (self.final_threshold - self.initial_threshold) *
                         (1 - math.cos(math.pi * progress)) / 2)

        elif self.schedule_type == 'exponential':
            threshold = (self.initial_threshold *
                         (self.final_threshold / self.initial_threshold) ** progress)

        else:
            threshold = (self.initial_threshold +
                         (self.final_threshold - self.initial_threshold) * progress)

        return max(min(threshold, self.final_threshold), self.initial_threshold)


# =============================================================================
# Configuration Helpers
# =============================================================================
def print_config_summary(config: object) -> None:
    """
    Print a summary of the current configuration.

    Args:
        config: Configuration object to summarize
    """
    print("\n" + "=" * 50)
    print("CONFIGURATION SUMMARY")
    print("=" * 50)

    print(f"Model: {config.model_name}")
    print(f"Span mode: {getattr(config, 'span_mode', 'default')}")
    print(f"Hidden size: {getattr(config, 'hidden_size', 768)}")

    print(f"Batch size: {config.train_batch_size}")
    print(f"Learning rate (encoder): {config.lr_encoder}")
    print(f"Learning rate (others): {config.lr_others}")
    print(f"Max steps: {config.num_steps}")
    print(f"Evaluation every: {config.eval_every} steps")

    print(f"Focal loss gamma: {getattr(config, 'loss_gamma', 'not set')}")
    print(f"Focal loss alpha: {getattr(config, 'loss_alpha', 'not set')}")
    print(f"Label smoothing: {getattr(config, 'label_smoothing', 'not set')}")

    if hasattr(config, 'disable_contrastive') and not config.disable_contrastive:
        print(f"\nContrastive Learning:")
        print(f"  Temperature: {getattr(config, 'entity_type_temperature', 'not set')}")
        print(f"  Weight: {getattr(config, 'contrastive_initial_weight', 'not set')} → "
              f"{getattr(config, 'contrastive_final_weight', 'not set')}")

    if hasattr(config, 'early_stopping_patience'):
        print(f"\nEvaluation:")
        print(f"  Early stopping patience: {config.early_stopping_patience}")
        print(
            f"  Threshold scheduler: {'enabled' if getattr(config, 'use_threshold_scheduler', False) else 'disabled'}")

    print("=" * 50)

# =============================================================================
# File System Helpers
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Get the latest checkpoint file from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    def get_step_number(checkpoint_path):
        try:
            return int(checkpoint_path.name.split("-")[1])
        except (IndexError, ValueError):
            return 0

    latest = max(checkpoints, key=get_step_number)
    return latest


def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], keep_last: int = 3) -> None:
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if len(checkpoints) <= keep_last:
        return

    def get_step_number(checkpoint_path):
        try:
            return int(checkpoint_path.name.split("-")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_step_number, reverse=True)

    for checkpoint in checkpoints[keep_last:]:
        try:
            if checkpoint.is_dir():
                import shutil
                shutil.rmtree(checkpoint)
            else:
                checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint.name}")
        except Exception as e:
            print(f"Failed to remove checkpoint {checkpoint.name}: {e}")