from __future__ import annotations

from typing import TYPE_CHECKING

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from chemprop import models
from chemprop import nn as chem_nn
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
    MetricCollection,
    Precision,
    Recall,
)
from transformers import AutoModel, get_linear_schedule_with_warmup

if TYPE_CHECKING:
    from .config import Config


def build_mpnn_model(batch_norm=False, aggregation="mean", metrics=None, use_atom_mp=False, **kwargs):
    if use_atom_mp:
        mp = chem_nn.AtomMessagePassing(dropout=0.1)
    else:
        mp = chem_nn.BondMessagePassing(dropout=0.1)

    if aggregation == "mean":
        agg = chem_nn.MeanAggregation()
    elif aggregation == "sum":
        agg = chem_nn.SumAggregation()
    elif aggregation == "attn":
        agg = chem_nn.AttentiveAggregation(output_size=mp.output_dim)
    elif aggregation == "norm":
        agg = chem_nn.NormAggregation()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    ffn = chem_nn.BinaryClassificationFFN(n_tasks=1, dropout=0.1)

    return models.MPNN(mp, agg, ffn, batch_norm=batch_norm, metrics=metrics, **kwargs)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.classifier(x.float())


class FpBinaryClassifier(L.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        input_dim = self._resolve_fp_dim(config.fp_type, config.fp_kwargs)
        self.classifier = Classifier(
            input_dim=input_dim,
            hidden_dim=128,
            dropout=0.1,
        )

        # Metrics
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary"),
                "f1_score": F1Score(task="binary"),
                "auroc": AUROC(task="binary"),
                "mcc": MatthewsCorrCoef(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        logits = self.classifier(x)
        return logits

    def _compute_loss(self, logits, labels):
        weight = None
        if self.config.pos_weight is not None:
            weight = torch.tensor([self.config.pos_weight], device=self.device)

        return F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float(), pos_weight=weight
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_features"])
        loss = self._compute_loss(logits, batch["labels"])

        probs = torch.sigmoid(logits.view(-1))
        self.train_metrics(probs, batch["labels"])

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["input_features"])
        loss = self._compute_loss(logits, batch["labels"])

        probs = torch.sigmoid(logits.view(-1))
        self.val_metrics(probs, batch["labels"])

        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_features"])
        probs = torch.sigmoid(logits.view(-1))
        self.test_metrics(probs, batch["labels"])
        self.log_dict(self.test_metrics, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["input_features"])
        return torch.sigmoid(logits.view(-1))

    @staticmethod
    def _resolve_fp_dim(fp_type: str, fp_kwargs: dict) -> int:
        """Return the actual feature dimension for the given fingerprint type."""
        _FIXED_DIMS = {"MACCS": 166, "RDKit2D": 200, "PubChem": 881}
        if fp_type in _FIXED_DIMS:
            return _FIXED_DIMS[fp_type]
        return fp_kwargs.get("fp_size", 1024)

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class ChemBERTaBinaryClassifier(L.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # 1. Load Base Model
        self.backbone = AutoModel.from_pretrained(self.config.model_name)
        H_SIZE = self.backbone.config.hidden_size

        self.attn_pool = nn.Sequential(
            nn.Linear(H_SIZE, H_SIZE // 2),
            nn.Tanh(),
            nn.Linear(H_SIZE // 2, 1),
        )

        self.classifier = Classifier(
            input_dim=H_SIZE,
            hidden_dim=128,
            dropout=0.1,
        )

        # 3. Metrics
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary"),
                "f1_score": F1Score(task="binary"),
                "auroc": AUROC(task="binary"),
                "mcc": MatthewsCorrCoef(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 4. LoRA Configuration
        if self.config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["query", "value"],
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                bias="lora_only",
                lora_dropout=0.1,
                use_dora=True,
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        # 1. Get Transformer Embeddings
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        pooled = self._pool(
            self.config.pooling, outputs.last_hidden_state, attention_mask
        )
        # 2. Pass to Classifier
        logits = self.classifier(pooled)
        return logits

    def _pool(self, type, last_hidden_state, attention_mask):
        if type == "cls":
            return last_hidden_state[:, 0, :]
        elif type == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts
        elif type == "attn":
            scores = self.attn_pool(last_hidden_state)
            scores = scores.masked_fill(  # masking out padding tokens
                attention_mask.unsqueeze(-1) == 0, float("-inf")
            )
            alpha = F.softmax(scores, dim=1)
            return (alpha * last_hidden_state).sum(dim=1)
        elif type == "norm":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            # Dividing by 100.0 to match the default Chemprop NormAggregation constant
            return summed / 100.0
        else:
            raise ValueError(f"Unknown pooling type: {type}")

    def _compute_loss(self, logits, labels):
        weight = None
        if self.config.pos_weight is not None:
            weight = torch.tensor([self.config.pos_weight], device=self.device)

        return F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float(), pos_weight=weight
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self._compute_loss(logits, batch["labels"])

        probs = torch.sigmoid(logits.view(-1))
        self.train_metrics(probs, batch["labels"])

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self._compute_loss(logits, batch["labels"])

        probs = torch.sigmoid(logits.view(-1))
        self.val_metrics(probs, batch["labels"])

        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        probs = torch.sigmoid(logits.view(-1))
        self.test_metrics(probs, batch["labels"])
        self.log_dict(self.test_metrics, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["input_ids"], batch["attention_mask"])
        return torch.sigmoid(logits.view(-1))

    def param_groups(self, module, lr, weight_decay):
        no_decay = ["bias", "LayerNorm.weight", "raw_norm.weight", "raw_norm.bias"]

        decay_params = []
        no_decay_params = []

        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        return [
            {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        optimizer_grouped_parameters = []

        # Backbone (lower LR)
        optimizer_grouped_parameters += self.param_groups(
            self.backbone,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Classifier (higher LR)
        optimizer_grouped_parameters += self.param_groups(
            self.classifier,
            lr=self.config.lr * 10,
            weight_decay=self.config.weight_decay,
        )

        # Attention Pool (higher LR, only if pooling is attn)
        if self.config.pooling == "attn":
            optimizer_grouped_parameters += self.param_groups(
                self.attn_pool,
                lr=self.config.lr * 10,
                weight_decay=self.config.weight_decay,
            )

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ── Factory ──────────────────────────────────────────────────────────────────


def build_model(config: Config, metrics=None):
    """Instantiate the correct model from a unified config.

    Parameters
    ----------
    config : Config
        One of ``MPNNConfig``, ``ChemBERTaConfig``, or ``FingerprintConfig``.
    metrics : list, optional
        Chemprop metric objects (only used for MPNN).

    Returns
    -------
    model
        A Lightning module (or chemprop ``MPNN``).
    """
    if config.model_type in ("mpnn", "mpnn_atom"):
        return build_mpnn_model(
            batch_norm=config.batch_norm,
            aggregation=config.aggregation,
            warmup_epochs=config.warmup_epochs,
            init_lr=config.init_lr,
            max_lr=config.max_lr,
            final_lr=config.final_lr,
            metrics=metrics,
            use_atom_mp=(config.model_type == "mpnn_atom"),
        )
    elif config.model_type == "chemberta":
        return ChemBERTaBinaryClassifier(config=config)
    elif config.model_type == "fingerprint":
        return FpBinaryClassifier(config=config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
