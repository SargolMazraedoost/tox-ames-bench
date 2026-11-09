"""Conventional machine-learning baselines on molecular fingerprints.

Usage (CLI)::

    python -m tox_ames.train_ml --estimator rf --fp-type ECFP
    python -m tox_ames.train_ml --estimator gb --fp-type MACCS --n-trials 50
    python -m tox_ames.train_ml --estimator rf --fp-type ECFP --no-tune

Usage (Python)::

    from tox_ames.config import ClassicalMLConfig
    from tox_ames.train_ml import run_ml_cv

    config = ClassicalMLConfig(estimator="gb", fp_type="ECFP")
    results, summary, predictions = run_ml_cv(df, config)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic_yaml import to_yaml_file
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tox_ames.data import get_fingerprint
from tox_ames.utils.helper import build_fold_predictions, make_splits
from tox_ames.utils.logger import get_logger

if TYPE_CHECKING:
    from tox_ames.config import ClassicalMLConfig

log = get_logger(__name__)

ESTIMATOR_NAMES = {
    "rf": "RandomForest",
    "gb": "GradientBoosting",
}


# ── Featurization ────────────────────────────────────────────────────────────


def featurize(df: pd.DataFrame, config: ClassicalMLConfig) -> np.ndarray:
    """Compute the fingerprint matrix for the whole dataset once.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with the SMILES column named by ``config.smiles_col``.
    config : ClassicalMLConfig
        Pipeline configuration.

    Returns
    -------
    np.ndarray
        Float feature matrix of shape ``(n_molecules, n_features)``.
    """
    if config.fp_type == "custom":
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        try:
            feat_cols = sorted(feat_cols, key=lambda c: int(c.split("_")[1]))
        except (IndexError, ValueError):
            feat_cols = sorted(feat_cols)
        return df[feat_cols].to_numpy(dtype=float)

    smiles = df[config.smiles_col].tolist()

    # Cache the fingerprint matrix so repeated runs over the same dataset and
    # fingerprint (e.g. random forest then gradient boosting) featurize once.
    key = hashlib.sha1(
        json.dumps(
            {
                "fp_type": config.fp_type,
                "fp_kwargs": config.fp_kwargs,
                "n": len(smiles),
                "smiles": smiles,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]
    cache_path = Path("data/fp_cache") / f"{config.fp_type}_{key}.npy"

    if cache_path.exists():
        log.info(f"Loading cached {config.fp_type} fingerprints from {cache_path}")
        return np.load(cache_path)

    log.info(f"Computing {config.fp_type} fingerprints for {len(smiles)} molecules")
    fps = np.asarray(
        get_fingerprint(smiles, fp_type=config.fp_type, **config.fp_kwargs),
        dtype=float,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, fps)
    log.info(f"Cached fingerprints to {cache_path}")
    return fps


# ── Search spaces ────────────────────────────────────────────────────────────


def suggest_params(trial: Any, estimator: str) -> dict[str, Any]:
    """Sample one hyperparameter configuration for the given estimator."""
    if estimator == "rf":
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [50, 100, 200]
            ),
            "max_depth": trial.suggest_categorical("max_depth", [3, 8, 10]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    if estimator == "gb":
        # ``GradientBoostingClassifier`` fits exact splits on a single core, so
        # the ranges are kept moderate and ``subsample``/``max_features`` are
        # tuned to bound per-trial cost.
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [50, 100, 200]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
        }
    raise ValueError(f"Unknown estimator: {estimator}")


def build_estimator(
    estimator: str,
    params: dict[str, Any],
    seed: int | None,
    n_jobs: int,
):
    """Instantiate an unfitted scikit-learn classifier."""
    if estimator == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if estimator == "gb":
        # Single-threaded by design: ``n_jobs`` does not apply.
        return GradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown estimator: {estimator}")


def _score(y_true: np.ndarray, y_prob: np.ndarray, metric: str, threshold: float) -> float:
    """Score validation predictions with the configured tuning metric."""
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unknown tuning metric: {metric}")


def _test_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute test metrics using the same column names as ``train.py``."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "test/acc": float(accuracy_score(y_true, y_pred)),
        "test/auroc": float(roc_auc_score(y_true, y_prob)),
        "test/f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "test/mcc": float(matthews_corrcoef(y_true, y_pred)),
        "test/precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test/recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ── Per-fold tuning ──────────────────────────────────────────────────────────


def tune_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ClassicalMLConfig,
    fold: int,
) -> tuple[dict[str, Any], float]:
    """Select hyperparameters for one fold on its validation split.

    Returns
    -------
    tuple[dict, float]
        Best parameter dictionary and its validation score.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seed = (config.seed or 0) + fold

    def objective(trial: Any) -> float:
        params = suggest_params(trial, config.estimator)
        model = build_estimator(config.estimator, params, seed, config.n_jobs)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_val)[:, 1]
        return _score(y_val, y_prob, config.tune_metric, config.threshold)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


# ── Unified CV ───────────────────────────────────────────────────────────────


def run_ml_cv(
    df: pd.DataFrame,
    config: ClassicalMLConfig,
    save_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeated scaffold-balanced cross-validation for a conventional baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with SMILES and label columns.
    config : ClassicalMLConfig
        Pipeline configuration.
    save_results : bool
        Whether to persist per-fold predictions, metrics, and chosen
        hyperparameters.

    Returns
    -------
    results_df : pd.DataFrame
        Per-fold test metrics.
    summary_df : pd.DataFrame
        Mean and standard deviation across folds.
    predictions_df : pd.DataFrame
        Per-sample test predictions with fold labels.
    """
    smiles = df[config.smiles_col].to_numpy()
    y = df[config.label_col].to_numpy()

    splits, _ = make_splits(
        smiles,
        y,
        config.split_type,
        config.seed,
        config.n_splits,
    )

    features = featurize(df, config)

    save_dir = Path("logs") / config.log_dir
    if save_results:
        save_dir.mkdir(parents=True, exist_ok=True)
        to_yaml_file(save_dir / "config.yaml", config)

    fold_metrics: list[dict[str, Any]] = []
    fold_predictions: list[pd.DataFrame] = []
    fold_val_predictions: list[pd.DataFrame] = []
    best_params_log: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        test_idx = np.asarray(test_idx)

        log.info("=" * 50)
        log.info(
            f"Fold {fold} / {config.n_splits} "
            f"[{ESTIMATOR_NAMES[config.estimator]} / {config.fp_type}]"
        )
        log.info(
            f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
        )
        log.info("=" * 50)

        x_train, y_train = features[train_idx], y[train_idx].astype(int)
        x_val, y_val = features[val_idx], y[val_idx].astype(int)
        x_test = features[test_idx]

        if config.tune:
            params, val_score = tune_fold(
                x_train, y_train, x_val, y_val, config, fold
            )
            log.info(
                f"  best val {config.tune_metric}={val_score:.4f}  params={params}"
            )
        else:
            params, val_score = dict(config.fixed_params), float("nan")

        seed = (config.seed or 0) + fold
        model = build_estimator(config.estimator, params, seed, config.n_jobs)
        model.fit(x_train, y_train)

        y_prob_test = model.predict_proba(x_test)[:, 1]
        y_prob_val = model.predict_proba(x_val)[:, 1]

        metrics = _test_metrics(y[test_idx].astype(int), y_prob_test, config.threshold)
        metrics["fold"] = fold
        metrics[f"best_val_{config.tune_metric}"] = val_score
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        metrics["n_test"] = len(test_idx)
        fold_metrics.append(metrics)
        log.info(
            f"  test auroc={metrics['test/auroc']:.4f}  mcc={metrics['test/mcc']:.4f}"
        )

        fold_pred_df = build_fold_predictions(
            [y_prob_test], y, test_idx, smiles, fold, config.threshold
        )
        fold_predictions.append(fold_pred_df)

        # Validation predictions support validation-based configuration choices
        # (e.g. selecting a fingerprint) without touching the test split.
        fold_val_predictions.append(
            build_fold_predictions(
                [y_prob_val], y, val_idx, smiles, fold, config.threshold
            )
        )

        best_params_log.append({"fold": fold, "val_score": val_score, **params})

        if save_results:
            fold_pred_df.to_csv(save_dir / f"predictions_fold{fold}.csv", index=False)

    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    val_predictions_df = pd.concat(fold_val_predictions, ignore_index=True)
    results_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)

    skip_cols = {
        "fold",
        "n_train",
        "n_val",
        "n_test",
        f"best_val_{config.tune_metric}",
    }
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
        val_predictions_df.to_csv(save_dir / "cv_val_predictions.csv", index=False)
        (save_dir / "best_params.json").write_text(
            json.dumps(best_params_log, indent=2)
        )
        log.info(f"Results saved to {save_dir}")

    return results_df, summary_df, predictions_df


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """Run one conventional baseline from the command line."""
    import tyro

    from tox_ames.config import ClassicalMLConfig

    config = tyro.cli(ClassicalMLConfig)

    if config.log_dir == "experiment":
        tag = "tuned" if config.tune else "default"
        config.log_dir = (
            f"exp-ml_{config.estimator}_{config.fp_type.lower()}_{tag}-scf"
        )

    log.info(f"Config: {config}")
    df = pd.read_csv(Path(config.data_path))
    _, summary_df, _ = run_ml_cv(df, config)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
