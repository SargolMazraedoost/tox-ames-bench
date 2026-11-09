from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# ── 1. CONSENSUS ENSEMBLE MODELING ───────────────────────────────────────────


def load_predictions(exp_dir: Path) -> pd.DataFrame:
    """Load and return cv_predictions.csv from an experiment directory."""
    pred_path = exp_dir / "cv_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found at {pred_path}")
    return pd.read_csv(pred_path)


def build_consensus_ensemble(exp_dirs: list[Path]) -> pd.DataFrame:
    """Build a consensus ensemble by averaging predicted probabilities of multiple runs."""
    dfs = []
    for idx, exp_dir in enumerate(exp_dirs):
        df = load_predictions(exp_dir)
        # Rename columns to avoid collision, keeping key identifiers
        df = df.rename(columns={"y_prob": f"y_prob_{idx}", "y_pred": f"y_pred_{idx}"})
        dfs.append(df)

    # Merge all dataframes on smiles, fold, and y_actual
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=["smiles", "fold", "y_actual"])

    # Compute ensemble probability
    prob_cols = [f"y_prob_{i}" for i in range(len(exp_dirs))]
    merged["y_prob"] = merged[prob_cols].mean(axis=1)
    merged["y_pred"] = (merged["y_prob"] >= 0.5).astype(int)

    return merged


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standard classification metrics fold-by-fold."""
    metrics = []
    for fold, group in df.groupby("fold"):
        y_true = group["y_actual"].values
        y_pred = group["y_pred"].values
        y_prob = group["y_prob"].values

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auroc = np.nan

        metrics.append(
            {"fold": fold, "Accuracy": acc, "F1": f1, "AUROC": auroc, "MCC": mcc}
        )

    return pd.DataFrame(metrics)


# ── 2. ERROR PROFILING BY PHYSICOCHEMICAL PROPERTIES ─────────────────────────


def compute_descriptors(smiles: str) -> dict:
    """Compute molecular descriptors for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "QED": Descriptors.qed(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "Fsp3": Descriptors.FractionCSP3(mol),
        }
    except Exception:
        return {}


def profile_errors_by_property(
    df: pd.DataFrame,
    prop_name: str | list[str],
    num_bins: int = 5,
    save_path: Path = None,
):
    """Plot fold-level error metrics for one or more properties."""
    prop_names = [prop_name] if isinstance(prop_name, str) else prop_name
    if not prop_names:
        raise ValueError("At least one property is required")

    results = {}
    for prop in prop_names:
        results[prop] = _calculate_property_error_profile(df, prop, num_bins)

    metric_order = ["Accuracy", "False Positive Rate", "False Negative Rate"]
    palette = {
        "Accuracy": "teal",
        "False Positive Rate": "turquoise",
        "False Negative Rate": "palegreen",
    }
    figsize = (10, 5) if len(prop_names) == 1 else (4 * len(prop_names), 3.5)
    fig, axes = plt.subplots(
        1, len(prop_names), figsize=figsize, sharey=len(prop_names) > 1, squeeze=False
    )

    for ax, prop in zip(axes.flat, prop_names):
        bin_df, df_long = results[prop]
        sns.barplot(
            data=df_long,
            x="Bin Range",
            y="Score",
            hue="Metric",
            order=bin_df["Bin Range"].tolist(),
            hue_order=metric_order,
            errorbar="sd",
            capsize=0.3,
            edgecolor="k",
            palette=palette,
            alpha=0.6,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=15)
        ax.set_xlabel(f"{prop} Range")
        ax.set_ylabel("Metric Score")
        ax.set_title(f"by {prop}")
        ax.grid(alpha=0.7, linestyle="--")

    if len(prop_names) == 1:
        axes.flat[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
    else:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.get_legend().remove()
        fig.legend(handles, labels, loc="upper center", ncol=len(metric_order))
        fig.tight_layout(rect=(0, 0, 1, 0.9))

    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()

    bin_dfs = {prop: result[0] for prop, result in results.items()}
    return bin_dfs[prop_names[0]] if len(prop_names) == 1 else bin_dfs


def _calculate_property_error_profile(
    df: pd.DataFrame, prop_name: str, num_bins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fold-level and aggregated error metrics for one property."""
    # Compute descriptors if not already present
    if prop_name not in df.columns:
        print(f"Computing {prop_name} descriptor for all unique compounds...")
        unique_smiles = df["smiles"].unique()
        desc_dict = {}
        for s in unique_smiles:
            desc_dict[s] = compute_descriptors(s).get(prop_name, np.nan)
        df[prop_name] = df["smiles"].map(desc_dict)

    # Drop rows where descriptor computation failed
    df_clean = df.dropna(subset=[prop_name]).copy()

    # Bin the property
    df_clean["bin"] = pd.qcut(df_clean[prop_name], q=num_bins, duplicates="drop")

    records = []
    for (fold, bin_val), group in df_clean.groupby(["fold", "bin"], observed=True):
        y_true = group["y_actual"].values
        y_pred = group["y_pred"].values

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Round interval limits for clean range display
        if prop_name == "MW":
            left = int(round(bin_val.left))
            right = int(round(bin_val.right))
        else:
            left = round(bin_val.left, 2)
            right = round(bin_val.right, 2)

        bin_label = f"({left}, {right}]"

        records.append(
            {
                "fold": fold,
                "Bin Range": bin_label,
                "Bin Center": bin_val.mid,
                "Accuracy": acc,
                "FPR": fpr,
                "FNR": fnr,
                "Count": len(group),
            }
        )

    fold_bin_df = pd.DataFrame(records)
    bin_df = (
        fold_bin_df.groupby(["Bin Range", "Bin Center"], sort=False)
        .agg(
            Accuracy=("Accuracy", "mean"),
            Accuracy_SD=("Accuracy", "std"),
            FPR=("FPR", "mean"),
            FPR_SD=("FPR", "std"),
            FNR=("FNR", "mean"),
            FNR_SD=("FNR", "std"),
            Count=("Count", "sum"),
        )
        .reset_index()
    )

    # Prepare long-form DataFrame for Seaborn plotting
    fold_bin_df = fold_bin_df.rename(
        columns={"FPR": "False Positive Rate", "FNR": "False Negative Rate"}
    )

    df_long = fold_bin_df.melt(
        id_vars=["fold", "Bin Range"],
        value_vars=["Accuracy", "False Positive Rate", "False Negative Rate"],
        var_name="Metric",
        value_name="Score",
    )

    return bin_df, df_long


# ── 3. UNCERTAINTY QUANTIFICATION (UQ) ───────────────────────────────────────


def plot_confidence_accuracy_curve(df: pd.DataFrame, save_path: Path = None):
    """Plot accuracy against confidence percentile threshold."""
    # Confidence is high when probability is close to 0 or 1
    df = df.copy()
    # Let's define confidence = |y_prob - 0.5|
    df["confidence"] = np.abs(df["y_prob"] - 0.5)

    thresholds = np.percentile(df["confidence"], np.linspace(0, 95, 20))
    accuracies = []
    retained_fractions = []

    for t in thresholds:
        retained = df[df["confidence"] >= t]
        acc = accuracy_score(retained["y_actual"], retained["y_pred"])
        accuracies.append(acc)
        retained_fractions.append(len(retained) / len(df))

    plt.figure(figsize=(8, 5))
    plt.plot(retained_fractions, accuracies, marker="o", color="#10b981", linewidth=2)
    plt.xlabel("Fraction of Data Retained (Highest Confidence)")
    plt.ylabel("Accuracy on Retained Subset")
    plt.title("Confidence-Accuracy Curve")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().invert_xaxis()  # Retained fraction goes from 1.0 (all data) to 0.05 (most confident 5%)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return pd.DataFrame(
        {"Retained Fraction": retained_fractions, "Accuracy": accuracies}
    )


# ── 4. TANIMOTO SIMILARITY VS ACCURACY (APPLICABILITY DOMAIN) ────────────────


def get_train_test_split_smiles(
    config_path: Path, data_path: Path, fold: int
) -> tuple[list[str], list[str]]:
    """Reconstruct train and test SMILES for a given fold using saved configuration."""
    from tox_ames.utils.helper import make_splits

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(data_path)
    smiles = df[config.get("smiles_col", "smiles")].values
    y = df[config.get("label_col", "target")].values

    splits, _ = make_splits(
        smiles,
        y,
        config.get("split_type", "SCAFFOLD_BALANCED"),
        config.get("seed", 42),
        config.get("n_splits", 25),
    )

    # fold is 1-indexed, split index is 0-indexed
    train_idx, val_idx, test_idx = splits[fold - 1]

    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]

    return train_smiles, test_smiles


def compute_similarity_to_training(
    train_smiles: list[str], test_smiles: list[str]
) -> list[float]:
    """Compute maximum Tanimoto similarity of each test compound to the training set."""
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]

    train_mols = [m for m in train_mols if m is not None]

    # Compute fingerprints
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in train_mols
    ]

    max_similarities = []
    for m in test_mols:
        if m is None:
            max_similarities.append(0.0)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_similarities.append(max(sims) if sims else 0.0)

    return max_similarities


# ── 5. REPEATED-HOLDOUT METRIC SUMMARIES ────────────────────────────────────


def collect_metrics_by_model(
    ensemble_df: pd.DataFrame, model_dirs: list[Path], model_names: list[str]
) -> dict[str, pd.DataFrame]:
    """Collect per-holdout metrics without treating holdouts as independent."""
    metrics_by_model: dict[str, pd.DataFrame] = {}
    metrics_by_model["Cross-repr."] = calculate_metrics(ensemble_df)

    for name, exp_dir in zip(model_names, model_dirs):
        model_key = name.split("_")[0].replace("exp-", "").upper()
        pred_df = pd.read_csv(exp_dir / "cv_predictions.csv")
        metrics_by_model[model_key] = calculate_metrics(pred_df)

    return metrics_by_model


def plot_cv_metrics_comparison(
    metrics_by_model: dict[str, pd.DataFrame], save_path: Path = None
):
    """Plot metric distributions across the repeated holdouts."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target_metrics = ["Accuracy", "F1", "AUROC", "MCC"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 6),
        sharex=True,
    )
    axes = axes.ravel()

    # Prepare long-form DataFrame for plotting
    plot_data = []
    for model_name, df in metrics_by_model.items():
        for metric in target_metrics:
            for val in df[metric].values:
                plot_data.append({"Model": model_name, "Metric": metric, "Score": val})
    plot_df = pd.DataFrame(plot_data)

    for idx, metric in enumerate(target_metrics):
        ax = axes[idx]
        metric_df = plot_df[plot_df["Metric"] == metric]

        sns.boxplot(
            data=metric_df,
            x="Model",
            y="Score",
            hue="Model",
            ax=ax,
            palette="GnBu_r",
            legend=False,
            showmeans=True,
            meanprops={
                "marker": "^",
                "markeredgecolor": "black",
                "markerfacecolor": "white",
            },
        )
        ax.set_title(
            f"Repeated-Holdout {metric} Comparison", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        # right-align x-tick labels and rotate for better readability
        ax.tick_params(axis="x", rotation=30, labelsize=10)

        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
