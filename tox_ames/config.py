"""Unified configuration for all model pipelines.

Usage (CLI)::

    python -m tox_ames.train mpnn --batch-norm
    python -m tox_ames.train chemberta --model-name DeepChem/ChemBERTa-77M-MTR
    python -m tox_ames.train fingerprint --fp-type Avalon

Usage (Python)::

    from tox_ames.config import FingerprintConfig
    cfg = FingerprintConfig(fp_type="Avalon", fp_kwargs={"fp_size": 512})
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Union

from pydantic import BaseModel, Field

# ── Shared base ──────────────────────────────────────────────────────────────


class BaseConfig(BaseModel):
    """Fields common to every pipeline."""

    # Data ----
    seed: int | None = 42
    smiles_col: str = "smiles"
    label_col: str = "target"

    # Split ----
    n_splits: int = 25
    split_type: Literal["random", "SCAFFOLD_BALANCED"] = "SCAFFOLD_BALANCED"

    # DataLoader ----
    batch_size: int = Field(64, gt=0)
    num_workers: int = 12
    shuffle: bool = False
    drop_last: bool = False
    train_shuffle: bool = True
    train_drop_last: bool = True
    train_cls_balance: bool = True

    # Optimizer (used by ChemBERTa & Fingerprint) ----
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    pos_weight: float | None = None

    # Trainer ----
    log_dir: str = "experiment"
    max_epochs: int = 20
    threshold: float = 0.5
    early_stopping_patience: int | None = 10
    gradient_clip_val: float | None = None


# ── Model-specific configs ───────────────────────────────────────────────────


class MPNNConfig(BaseConfig):
    """Chemprop MPNN configuration."""

    model_type: Literal["mpnn", "mpnn_atom"] = "mpnn"
    batch_norm: bool = True
    aggregation: Literal["mean", "sum", "attn", "norm"] = "mean"
    warmup_epochs: int = 2
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4


class ChemBERTaConfig(BaseConfig):
    """ChemBERTa transformer configuration."""

    model_type: Literal["chemberta"] = "chemberta"
    model_name: str = "DeepChem/ChemBERTa-77M-MTR"
    max_length: int = Field(128, gt=0)
    pooling: Literal["cls", "mean", "attn", "norm"] = "mean"
    freeze_backbone: bool = False
    use_lora: bool = False
    r: int = 16
    lora_alpha: int = 32


class FingerprintConfig(BaseConfig):
    """Molecular fingerprint configuration."""

    model_type: Literal["fingerprint"] = "fingerprint"
    fp_type: Literal["ECFP", "Avalon", "MACCS", "RDKit2D", "PubChem", "custom"] = "ECFP"
    fp_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"fp_size": 1024},
    )


class ClassicalMLConfig(BaseConfig):
    """Conventional (non-neural) fingerprint baseline configuration.

    Used by :mod:`tox_ames.train_ml` for random forest and histogram
    gradient boosting baselines with per-fold, validation-based
    hyperparameter tuning.
    """

    model_type: Literal["ml"] = "ml"
    estimator: Literal["rf", "gb"] = "rf"
    fp_type: Literal["ECFP", "Avalon", "MACCS", "RDKit2D", "PubChem", "custom"] = "ECFP"
    fp_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"fp_size": 1024},
    )

    # Tuning ----
    tune: bool = True
    fixed_params: Dict[str, Any] = Field(default_factory=dict)
    n_trials: int = Field(50, gt=0)
    tune_metric: Literal["auroc", "mcc", "f1"] = "auroc"
    n_jobs: int = -1

    # Data ----
    data_path: str = "data/ames-combined/merged/ames_final.csv"


# ── Discriminated union for tyro CLI ─────────────────────────────────────────

Config = Annotated[
    Union[MPNNConfig, ChemBERTaConfig, FingerprintConfig, ClassicalMLConfig],
    Field(discriminator="model_type"),
]
