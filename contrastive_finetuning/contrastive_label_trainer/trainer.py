import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional, Tuple, Any, Union, List
import numpy as np
from dataclasses import dataclass
from logger import CustomLogger
from torch.utils.data import DataLoader

@dataclass
class ContrastiveConfig:
    """Configuration for contrastive training"""
    temperature: float = 0.1
    alignment_weight: float = 1.0
    uniformity_weight: float = 0.5
    use_hard_negatives: bool = True
    similarity_scale: float = 1.0
    eps: float = 1e-8


class EntityContrastiveLoss(nn.Module):
    """Simplified contrastive loss for entity embeddings"""

    def __init__(
            self,
            similarity_manager,
            config: ContrastiveConfig
    ):
        super().__init__()
        self.similarity_manager = similarity_manager
        self.config = config
        self._logger = CustomLogger(name="EntityContrastiveLoss")

        self._cache_similarity_matrix()

    def _cache_similarity_matrix(self):
        """Cache the similarity matrix as a tensor"""
        n_types = len(self.similarity_manager.entity_types)
        sim_matrix = torch.zeros(n_types, n_types)

        for i in range(n_types):
            for j in range(n_types):
                sim_matrix[i, j] = self.similarity_manager.get_similarity(i, j)

        self.register_buffer('similarity_matrix', sim_matrix)
        self._logger.info(f"Cached similarity matrix of shape {sim_matrix.shape}")

    def forward(
            self,
            embeddings: torch.Tensor,
            entity_type_ids: torch.Tensor,
            return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute contrastive loss"""

        embeddings = F.normalize(embeddings, p=2, dim=1, eps=self.config.eps)
        similarities = torch.matmul(embeddings, embeddings.T)

        batch_size = entity_type_ids.shape[0]
        target_similarities = self.similarity_matrix[
            entity_type_ids.unsqueeze(1),
            entity_type_ids.unsqueeze(0)
        ]

        target_similarities = target_similarities * self.config.similarity_scale

        losses = {}

        if self.config.alignment_weight > 0:
            alignment_loss = self._compute_infonce_loss(
                similarities, target_similarities, entity_type_ids
            )
            losses['alignment'] = alignment_loss * self.config.alignment_weight

        if self.config.uniformity_weight > 0:
            uniformity_loss = self._compute_uniformity_loss(embeddings)
            losses['uniformity'] = uniformity_loss * self.config.uniformity_weight

        total_loss = sum(losses.values())

        if return_components:
            with torch.no_grad():
                metrics = self._compute_metrics(
                    similarities, target_similarities, entity_type_ids
                )
                metrics.update({k: v.item() for k, v in losses.items()})
            return total_loss, metrics

        return total_loss

    def _compute_infonce_loss(
            self,
            similarities: torch.Tensor,
            target_similarities: torch.Tensor,
            entity_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE loss with soft targets based on entity type similarity"""

        batch_size = similarities.shape[0]
        pos_mask = entity_type_ids.unsqueeze(0) == entity_type_ids.unsqueeze(1)
        pos_mask.fill_diagonal_(False)

        labels = F.softmax(target_similarities / self.config.temperature, dim=1)
        log_probs = F.log_softmax(similarities / self.config.temperature, dim=1)
        loss = -torch.sum(labels * log_probs, dim=1).mean()

        return loss

    def _compute_uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Uniformity loss to prevent collapse"""
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        distances = torch.pdist(embeddings, p=2).pow(2)
        loss = torch.exp(-distances).mean()

        return loss

    def _compute_metrics(
            self,
            similarities: torch.Tensor,
            target_similarities: torch.Tensor,
            entity_type_ids: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for monitoring"""
        metrics = {}

        pos_mask = entity_type_ids.unsqueeze(0) == entity_type_ids.unsqueeze(1)
        pos_mask.fill_diagonal_(False)

        if pos_mask.sum() > 0:
            metrics['pos_sim_mean'] = similarities[pos_mask].mean().item()
            metrics['pos_sim_std'] = similarities[pos_mask].std().item()

        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)

        if neg_mask.sum() > 0:
            metrics['neg_sim_mean'] = similarities[neg_mask].mean().item()
            metrics['neg_sim_std'] = similarities[neg_mask].std().item()

            actual_sims = similarities[neg_mask]
            target_sims = target_similarities[neg_mask]

            if len(actual_sims) > 1:
                correlation = torch.corrcoef(
                    torch.stack([actual_sims, target_sims])
                )[0, 1]
                metrics['alignment_corr'] = correlation.item()

        return metrics


class ContrastiveTrainer(Trainer):
    """Trainer for entity contrastive learning with stratified batch sampling"""

    def __init__(
            self,
            model: nn.Module,
            args: TrainingArguments,
            similarity_manager,
            contrastive_config: Optional[ContrastiveConfig] = None,
            train_dataset=None,
            eval_dataset=None,
            train_batch_sampler=None,
            **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )

        self.contrastive_config = contrastive_config or ContrastiveConfig()
        self.similarity_manager = similarity_manager
        self.train_batch_sampler = train_batch_sampler
        self.base_model_name = "microsoft/deberta-v3-small"

        self.loss_fn = EntityContrastiveLoss(
            similarity_manager=similarity_manager,
            config=self.contrastive_config
        ).to(self.args.device)

        self._logger = CustomLogger(name="ContrastiveTrainer")

    def get_train_dataloader(self) -> DataLoader:
        """Override to use stratified batch sampler"""
        if self.train_batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()

    def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: int = None,
    ) -> torch.Tensor:
        """Compute contrastive loss"""

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        entity_type_ids = inputs['entity_type_id']

        embeddings = model(input_ids, attention_mask)

        if self.state.global_step % self.args.logging_steps == 0:
            loss, metrics = self.loss_fn(
                embeddings, entity_type_ids, return_components=True
            )
            self.log(metrics)
        else:
            loss = self.loss_fn(embeddings, entity_type_ids)

        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to properly save EntityEncoder with all components.

        This method saves:
        - The base model (DeBERTa)
        - The projection head weights
        - Model configuration
        - Tokenizer
        - A convenience loading script
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        self._logger.info(f"Saving model to {output_dir}")

        base_model_path = os.path.join(output_dir, "base_model")
        self.model.base_model.save_pretrained(base_model_path)
        self._logger.info(f"Saved base model to {base_model_path}")

        projection_config = {
            'projection_dim': None,
            'hidden_dim': None,
            'num_projection_layers': 0,
            'use_batch_norm': False,
            'dropout': 0.1
        }

        modules = list(self.model.projection_head.modules())[1:]
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        batchnorm_layers = [m for m in modules if isinstance(m, nn.BatchNorm1d)]
        dropout_layers = [m for m in modules if isinstance(m, nn.Dropout)]

        if linear_layers:
            projection_config['hidden_dim'] = linear_layers[0].in_features
            projection_config['projection_dim'] = linear_layers[-1].out_features
            projection_config['num_projection_layers'] = len(linear_layers)

        projection_config['use_batch_norm'] = len(batchnorm_layers) > 0
        if dropout_layers:
            projection_config['dropout'] = dropout_layers[0].p

        projection_path = os.path.join(output_dir, "projection_head.pt")
        torch.save({
            'projection_head_state_dict': self.model.projection_head.state_dict(),
            'layer_norm_state_dict': self.model.layer_norm.state_dict(),
            'projection_config': projection_config
        }, projection_path)
        self._logger.info(f"Saved projection head to {projection_path}")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            self._logger.info("Saved tokenizer")

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        model_config = {
            'model_class': 'EntityEncoder',
            'base_model_name': self.base_model_name,
            'base_model_path': 'base_model',
            'projection_path': 'projection_head.pt',
            'projection_config': projection_config
        }

        config_path = os.path.join(output_dir, 'entity_encoder_config.json')

        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        self._logger.info(f"Saved model configuration to {config_path}")

        self._logger.info(f"Model successfully saved to {output_dir}")
        self._logger.info("To load the model, use: python load_model.py")


class FixedContrastiveBatchSampler(torch.utils.data.Sampler):
    """Fixed version of the batch sampler"""

    def __init__(
            self,
            dataset,
            similarity_manager,
            batch_size: int = 64,
            n_positives: int = 8,
            n_similar: int = 4,
            n_random: int = 4
    ):
        self.dataset = dataset
        self.similarity_manager = similarity_manager
        self.batch_size = batch_size
        self.n_positives = n_positives
        self.n_similar = n_similar
        self.n_random = n_random

        self._precompute_similar_types()

    def _precompute_similar_types(self):
        """Precompute similar types for each entity type"""
        self.similar_types = {}

        for type1 in self.similarity_manager.entity_types:
            type1_id = self.similarity_manager.entity_type_to_id[type1]
            similarities = []

            for type2 in self.similarity_manager.entity_types:
                if type1 != type2:
                    type2_id = self.similarity_manager.entity_type_to_id[type2]
                    sim = self.similarity_manager.get_similarity(type1_id, type2_id)
                    similarities.append((type2, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            self.similar_types[type1] = similarities

    def __iter__(self):
        """Generate batches"""
        n_batches = len(self)

        for _ in range(n_batches):
            batch_indices = []

            valid_types = [
                t for t, indices in self.dataset.entities_by_type.items()
                if len(indices) >= self.n_positives
            ]

            if not valid_types:
                continue

            anchor_type = np.random.choice(valid_types)

            pos_indices = np.random.choice(
                self.dataset.entities_by_type[anchor_type],
                size=min(self.n_positives, len(self.dataset.entities_by_type[anchor_type])),
                replace=False
            )
            batch_indices.extend(pos_indices)

            similar_types = [t for t, _ in self.similar_types[anchor_type][:10]]
            for similar_type in similar_types[:self.n_similar]:
                if similar_type in self.dataset.entities_by_type:
                    indices = self.dataset.entities_by_type[similar_type]
                    if indices:
                        batch_indices.extend(
                            np.random.choice(indices, size=min(4, len(indices)))
                        )

            while len(batch_indices) < self.batch_size:
                random_idx = np.random.randint(0, len(self.dataset))
                if random_idx not in batch_indices:
                    batch_indices.append(random_idx)

            yield batch_indices[:self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size


class EntityEncoder(nn.Module):
    """
    Entity encoder with projection head for contrastive learning.
    Takes a pre-trained transformer and adds projection layers.
    """

    def __init__(
            self,
            base_model: nn.Module,
            hidden_dim: int = 768,
            projection_dim: int = 256,
            num_projection_layers: int = 2,
            dropout: float = 0.1,
            use_batch_norm: bool = True
    ):
        super().__init__()
        self.base_model = base_model

        layers = []
        input_dim = hidden_dim

        for i in range(num_projection_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        layers.extend([
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim) if use_batch_norm else nn.Identity()
        ])

        self.projection_head = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through base model and projection head.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Normalized embeddings [batch_size, projection_dim]
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.layer_norm(cls_output)
        projected = self.projection_head(cls_output)

        return projected