import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datapreparator import DataPreparator
from entitytypesimilaritymanager import EntityTypeSimilarityManager
from entityextractor import EntityDataset
from batchsampler import StratifiedContrastiveBatchSampler

import torch
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import TrainingArguments, AutoModel, AutoTokenizer
from trainer import ContrastiveTrainer, ContrastiveConfig, EntityEncoder

# -------------------------------------------------------------------
# Backend Factory to Avoid Certificate Issues
# -------------------------------------------------------------------
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    """Create a Requests session that disables SSL verification."""
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# -------------------------------------------------------------------
# Helper Function to Create Stratified DataLoader
# -------------------------------------------------------------------
def create_stratified_contrastive_dataloader(
        data: List[Dict],
        tokenizer,
        entity_type_to_id: Dict[str, int],
        similarity_manager: 'EntityTypeSimilarityManager',
        batch_size: int = 32,
        num_workers: int = 4,
        is_train: bool = True,
        **dataset_kwargs
) -> 'DataLoader':

    dataset = EntityDataset(
        gliner_data=data,
        tokenizer=tokenizer,
        entity_type_to_id=entity_type_to_id,
        max_length=64,
        **dataset_kwargs
    )

    if is_train:
        batch_sampler = StratifiedContrastiveBatchSampler(
            dataset=dataset,
            similarity_manager=similarity_manager,
            batch_size=batch_size,
            similarity_strata=[
                (0.65, 1.0),
                (0.40, 0.65),
                (0.2, 0.4),
                (0.0, 0.2)
            ],
            similarity_temperature=1.0
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloader

# -------------------------------------------------------------------
# Training Data Loading and Normalization
# -------------------------------------------------------------------
data_preparator = DataPreparator(train_data_dir="GLiNER/data/train")
train_data = data_preparator.load_training_data()
filtered_train_data = data_preparator.filter_common_types(train_data=train_data, min_examples=50)

# -------------------------------------------------------------------
# Similarity Matrix Computation
# -------------------------------------------------------------------
similarity_manager = EntityTypeSimilarityManager(
    train_data=filtered_train_data,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.7
)

# -------------------------------------------------------------------
# Creation of DataLoaders
# -------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

train_size = int(0.95 * len(filtered_train_data))
train_split = filtered_train_data[:train_size]
val_split = filtered_train_data[train_size:]
entity_type_to_id = similarity_manager.get_entity_type_to_id()

train_dataloader = create_stratified_contrastive_dataloader(
    data=train_split,
    tokenizer=tokenizer,
    entity_type_to_id=entity_type_to_id,
    similarity_manager=similarity_manager,
    batch_size=64,
    num_workers=4,
    is_train=True
)

val_dataloader = create_stratified_contrastive_dataloader(
    data=val_split,
    tokenizer=tokenizer,
    entity_type_to_id=entity_type_to_id,
    similarity_manager=similarity_manager,
    batch_size=64,
    num_workers=4,
    is_train=False
)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def data_collator(batch):
    """Collate function to properly format batches from EntityDataset"""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'entity_type_id': torch.tensor([x['entity_type_id'] for x in batch]),
        'entity_type': [x['entity_type'] for x in batch],
        'entity_text': [x['entity_text'] for x in batch],
    }

# -------------------------------------------------------------------
# Create Datasets (not dataloaders!) for the Trainer
# -------------------------------------------------------------------
train_dataset = EntityDataset(
    gliner_data=train_split,
    tokenizer=tokenizer,
    entity_type_to_id=entity_type_to_id,
    max_length=64
)

val_dataset = EntityDataset(
    gliner_data=val_split,
    tokenizer=tokenizer,
    entity_type_to_id=entity_type_to_id,
    max_length=64
)

# -------------------------------------------------------------------
# Create Batch Sampler with IMPROVED DIVERSITY PARAMETERS
# -------------------------------------------------------------------
train_batch_sampler = StratifiedContrastiveBatchSampler(
    dataset=train_dataset,
    similarity_manager=similarity_manager,
    batch_size=64,
    anchor_examples=16,
    similar_type_examples=8,
    medium_type_examples=6,
    dissimilar_type_examples=8,
    n_similar_types=1,
    n_medium_types=2,
    n_dissimilar_types=4,
    similarity_strata=[
        (0.8, 1.0),   # Very high similarity
        (0.5, 0.8),   # Medium similarity
        (0.2, 0.5),   # Low similarity
        (0.0, 0.2)    # Very low similarity
    ],
    similarity_temperature=1.0,
    min_dissimilar_threshold=0.3,
    force_diversity=True,
    drop_last=True,
    balance_strategy='adaptive'
)

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
base_model = AutoModel.from_pretrained("microsoft/deberta-v3-small")
model = EntityEncoder(base_model, projection_dim=256)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_full_data",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=64,
    lr_scheduler_type="cosine",
    learning_rate=5e-6,
    warmup_ratio=0.1,
    logging_steps=500,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=3,
    greater_is_better=False,
    fp16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    max_grad_norm=0.5,
    report_to="wandb",
)

# Contrastive Config
contrastive_config = ContrastiveConfig(
    temperature=0.25,
    alignment_weight=1.0,
    uniformity_weight=0.05,
    use_hard_negatives=False,
    similarity_scale=2.0,
    eps=1e-8
)

# Initialize Trainer
trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    train_batch_sampler=train_batch_sampler,
    similarity_manager=similarity_manager,
    contrastive_config=contrastive_config,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Model Saving
print("Saving final model...")
final_model_path = os.path.join(training_args.output_dir, "final_model")
print(f"Saving final model to {final_model_path}")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)