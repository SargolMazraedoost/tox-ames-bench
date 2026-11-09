"""Unified cross-validation training for all model pipelines.

Usage (CLI)::

    python -m tox_ames.train mpnn --batch-norm --log-dir exp-mpnn
    python -m tox_ames.train chemberta --model-name DeepChem/ChemBERTa-77M-MTR
    python -m tox_ames.train fingerprint --fp-type Avalon --fp-kwargs '{"fp_size": 512}'

Usage (Python)::

    from tox_ames.config import FingerprintConfig
    from tox_ames.train import run_cv

    config = FingerprintConfig(fp_type="Avalon", log_dir="exp-fp-avalon")
    results, summary, predictions = run_cv(df, config)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from pydantic_yaml import to_yaml_file

from tox_ames.data import build_datamodule
from tox_ames.model import build_model
from tox_ames.utils.helper import build_fold_predictions, make_splits
from tox_ames.utils.logger import get_logger

if TYPE_CHECKING:
    from tox_ames.config import Config

log = get_logger(__name__)


# ── Default MPNN metrics (lazy import to avoid chemprop overhead) ────────────


def _default_mpnn_metrics():
    """Return a default list of chemprop binary metrics."""
    from chemprop.nn import metrics

    return [
        metrics.BinaryAccuracy(),
        metrics.BinaryF1Score(),
        metrics.BinaryAUROC(),
        metrics.BinaryMCCMetric(),
    ]


# ── Unified CV ───────────────────────────────────────────────────────────────


def run_cv(
    df: pd.DataFrame,
    config: Config,
    save_results: bool = True,
    mpnn_metrics: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified k-fold cross-validation for any supported model type.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with SMILES and label columns.
    config : Config
        One of ``MPNNConfig``, ``ChemBERTaConfig``, or ``FingerprintConfig``.
    save_results : bool
        Whether to persist per-fold predictions and aggregated CSVs.
    mpnn_metrics : list, optional
        Chemprop metric objects for MPNN. If ``None``, sensible defaults are
        used (Accuracy, F1, AUROC, MCC).

    Returns
    -------
    results_df : pd.DataFrame
        Per-fold test metrics.
    summary_df : pd.DataFrame
        Mean ± std across folds.
    predictions_df : pd.DataFrame
        Per-sample predictions with fold labels.
    """
    fold_metrics: list[dict] = []
    fold_predictions: list[pd.DataFrame] = []

    smiles = df[config.smiles_col].values
    y = df[config.label_col].values

    splits, data_mols = make_splits(
        smiles,
        y,
        config.split_type,
        config.seed,
        config.n_splits,
    )

    save_dir = Path("logs") / config.log_dir
    if save_results:
        save_dir.mkdir(parents=True, exist_ok=True)
        to_yaml_file(save_dir / "config.yaml", config)

    # Resolve MPNN metrics once
    if config.model_type in ("mpnn", "mpnn_atom") and mpnn_metrics is None:
        mpnn_metrics = _default_mpnn_metrics()

    for fold, split in enumerate(splits, start=1):
        log.info("=" * 50)
        log.info(f"Fold {fold} / {config.n_splits}")
        log.info(f"  train={len(split[0])}  val={len(split[1])}  test={len(split[2])}")
        log.info("=" * 50)

        # ── DataModule (factory dispatch) ────────────────────────────────
        datamodule = build_datamodule(
            config,
            df=df,
            split=split,
            data_mols=data_mols,
        )

        # ── Model (fresh weights every fold) ─────────────────────────────
        model = build_model(config, metrics=mpnn_metrics)

        # ── Callbacks ────────────────────────────────────────────────────
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"fold{fold}-{{epoch:02d}}-{{val_loss:.4f}}",
        )

        callbacks: list = [TQDMProgressBar(), checkpoint]

        if config.early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=config.early_stopping_patience,
                    mode="min",
                    verbose=True,
                )
            )

        # ── Trainer ──────────────────────────────────────────────────────
        fold_logger = CSVLogger("logs", name=config.log_dir, version=f"fold{fold}")
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            logger=fold_logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            callbacks=callbacks,
            gradient_clip_val=config.gradient_clip_val,
        )

        trainer.fit(model, datamodule=datamodule)

        # ── Evaluation ───────────────────────────────────────────────────
        test_metrics = trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path="best",
            weights_only=False,
        )[0]

        test_metrics["fold"] = fold
        test_metrics["best_val_loss"] = float(checkpoint.best_model_score)
        test_metrics["n_train"] = len(split[0])
        test_metrics["n_val"] = len(split[1])
        test_metrics["n_test"] = len(split[2])
        fold_metrics.append(test_metrics)

        # ── Per-sample predictions ───────────────────────────────────────
        pred_batches = trainer.predict(
            model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path="best",
            weights_only=False,
        )
        fold_pred_df = build_fold_predictions(
            pred_batches, y, split[2], smiles, fold, config.threshold
        )
        fold_predictions.append(fold_pred_df)

        if save_results:
            fold_pred_df.to_csv(save_dir / f"predictions_fold{fold}.csv", index=False)

    # ── Aggregate ────────────────────────────────────────────────────────
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    results_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)

    skip_cols = {"fold", "n_train", "n_val", "n_test", "best_val_loss"}
    metric_cols = [
        c
        for c in results_df.columns
        if c not in skip_cols and pd.api.types.is_numeric_dtype(results_df[c])
    ]
    summary_df = (
        results_df[metric_cols]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )

    if save_results:
        results_df.to_csv(save_dir / "cv_results.csv", index=False)
        summary_df.to_csv(save_dir / "cv_summary.csv", index=False)
        predictions_df.to_csv(save_dir / "cv_predictions.csv", index=False)
        log.info(f"Results saved to {save_dir}")

    return results_df, summary_df, predictions_df


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tyro

    from tox_ames.config import Config

    config = tyro.cli(Config)
    print(config)
