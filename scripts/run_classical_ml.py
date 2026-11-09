#!/usr/bin/env python
"""
Run Conventional Machine Learning Baselines (Random Forest & Gradient Boosting)
==============================================================================

Provides a streamlined runner for evaluating Random Forest and Gradient Boosting
across the four molecular fingerprints (ECFP4, MACCS, Avalon, PubChem) over the
25 repeated scaffold-balanced splits with Optuna validation-based tuning.
"""

import argparse
from pathlib import Path
import pandas as pd
from tox_ames.config import ClassicalMLConfig
from tox_ames.train_ml import run_ml_cv


def main():
    parser = argparse.ArgumentParser(description="Run classical ML baselines on Ames benchmark.")
    parser.add_argument("--estimator", choices=["rf", "gb"], default="rf", help="Estimator family (rf or gb).")
    parser.add_argument("--fp-type", choices=["ECFP", "MACCS", "Avalon", "PubChem"], default="MACCS", help="Fingerprint representation.")
    parser.add_argument("--tune", action="store_true", help="Enable validation-based Optuna hyperparameter tuning.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna tuning trials per split (default: 10).")
    parser.add_argument("--data-path", type=str, default="data/ames_final.csv", help="Path to benchmark dataset CSV.")
    parser.add_argument("--n-splits", type=int, default=25, help="Number of scaffold-balanced splits (default: 25).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for CPU fitting.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    tag = "tuned" if args.tune else "default"
    log_dir = f"exp-ml_{args.estimator}_{args.fp_type.lower()}_{tag}-scf"

    config = ClassicalMLConfig(
        estimator=args.estimator,
        fp_type=args.fp_type,
        tune=args.tune,
        n_trials=args.n_trials,
        data_path=str(data_path),
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        log_dir=log_dir,
    )

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} compounds. Running {args.n_splits}-fold scaffold cross-validation for {args.estimator.upper()} ({args.fp_type})...")

    results_df, summary_df, _ = run_ml_cv(df, config, save_results=True)
    print("\n" + "=" * 60)
    print("Cross-Validation Summary:")
    print("=" * 60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
