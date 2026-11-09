#!/usr/bin/env python
"""
Evaluate Statistical Significance with Nadeau-Bengio Resampled Correction
========================================================================

Computes dependent-sample corrected resampled t-tests (Nadeau & Bengio, 2003)
and paired Wilcoxon signed-rank tests across repeated scaffold-balanced splits,
properly accounting for training set overlap across repeated holdouts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def nadeau_bengio_correction(
    diffs: np.ndarray,
    n_train: int = 9445,
    n_test: int = 1180,
    confidence: float = 0.95,
) -> dict:
    """
    Compute Nadeau & Bengio (2003) corrected resampled t-test and CI.

    Parameters
    ----------
    diffs : np.ndarray
        Array of paired differences across K splits (e.g., metric_A - metric_B).
    n_train : int
        Number of training samples per split.
    n_test : int
        Number of test samples per split.
    confidence : float
        Confidence level for interval estimation.

    Returns
    -------
    dict
        Dictionary containing mean diff, naive SE/CI, corrected SE/CI,
        t-statistic, p-value, and variance inflation factor.
    """
    K = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    std_d = float(np.sqrt(var_d))

    # Overlap correction factor: 1/K + n_test / n_train
    vif = 1.0 + K * (n_test / n_train)
    corr_factor = np.sqrt(1.0 / K + (n_test / n_train))
    se_naive = std_d / np.sqrt(K)
    se_corr = std_d * corr_factor

    df = K - 1
    t_crit = stats.t.ppf((1.0 + confidence) / 2.0, df=df)

    ci_naive = (mean_d - t_crit * se_naive, mean_d + t_crit * se_naive)
    ci_corr = (mean_d - t_crit * se_corr, mean_d + t_crit * se_corr)

    t_stat = mean_d / se_corr if se_corr > 0 else 0.0
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))

    return {
        "K": K,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se_naive": se_naive,
        "ci_naive_95": ci_naive,
        "se_corrected": se_corr,
        "ci_corrected_95": ci_corr,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "vif": vif,
    }


def main():
    print("=" * 70)
    print("Nadeau-Bengio Resampled Significance Evaluation")
    print("=" * 70)
    print("Correction parameters: K = 25 splits, n_train = 9,445, n_test = 1,180")
    print("Variance inflation factor (VIF): 4.123x (SE multiplier: ~2.03x)\n")

    # Example: Ensemble vs Uni-Mol v2 paired differences (ROC-AUC %)
    diffs_ens_unimol = np.array([
        -0.91, +3.52, -1.23, +0.15, +3.37, -0.87, +0.14, -0.41, -0.61, +0.46,
        +0.76, -0.54, +1.35, +1.82, -0.19, -0.37, -0.86, +0.57, -0.92, -1.10,
        +0.91, +0.73, +2.28, -0.76, -0.86
    ])

    res = nadeau_bengio_correction(diffs_ens_unimol, n_train=9445, n_test=1180)
    w_stat, p_wilcoxon = stats.wilcoxon(diffs_ens_unimol)

    print(f"Comparison: Cross-Representation Ensemble vs. Uni-Mol v2")
    print(f"Mean Paired Difference:      {res['mean_diff']:+.3f}%")
    print(f"Sample Standard Deviation:   {res['std_diff']:.3f}%")
    print(f"Naive SE (iid assumption):   {res['se_naive']:.3f}% -> 95% CI: [{res['ci_naive_95'][0]:+.2f}%, {res['ci_naive_95'][1]:+.2f}%]")
    print(f"Corrected SE (Nadeau-Bengio):{res['se_corrected']:.3f}% -> 95% CI: [{res['ci_corrected_95'][0]:+.2f}%, {res['ci_corrected_95'][1]:+.2f}%]")
    print(f"Corrected t-statistic:       t({res['df']}) = {res['t_stat']:.3f}, p = {res['p_value']:.4f}")
    print(f"Paired Wilcoxon Signed-Rank: W = {w_stat:.1f}, p = {p_wilcoxon:.4f}")
    print(f"Conclusion: Difference is NOT statistically significant (p > 0.05).")
    print("=" * 70)


if __name__ == "__main__":
    main()
