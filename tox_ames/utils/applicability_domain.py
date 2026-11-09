from __future__ import annotations

import datamol as dm
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tox_ames.utils.helper import make_splits


def compute_morgan_fps(
    smiles_list: list[str], radius: int = 2, n_bits: int = 1024
) -> list[DataStructs.ExplicitBitVect | None]:
    """Compute Morgan fingerprints (ECFP4) for a list of SMILES strings."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smiles in smiles_list:
        if not isinstance(smiles, str) or not smiles:
            fps.append(None)
            continue
        # Standardize and parse SMILES safely via datamol
        mol = dm.to_mol(smiles)
        if mol is None:
            fps.append(None)
        else:
            fps.append(generator.GetFingerprint(mol))
    return fps


def compute_max_similarities(
    query_smiles: list[str], reference_smiles: list[str]
) -> list[float]:
    """Compute the maximum Tanimoto similarity for each query compound relative to a reference library."""
    query_fps = compute_morgan_fps(query_smiles)
    reference_fps = [
        fp for fp in compute_morgan_fps(reference_smiles) if fp is not None
    ]

    if not reference_fps:
        return [0.0] * len(query_smiles)

    max_sims = []
    for fp in query_fps:
        if fp is None:
            max_sims.append(np.nan)
        else:
            # BulkTanimotoSimilarity is highly optimized in C++
            sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
            max_sims.append(max(sims))
    return max_sims


def get_fold_similarities(
    df: pd.DataFrame,
    split_type: str = "SCAFFOLD_BALANCED",
    seed: int = 42,
    n_splits: int = 25,
) -> pd.DataFrame:
    """Reconstruct CV splits and compute max similarity to training set for every test compound.

    Returns a DataFrame with columns: ['smiles', 'fold', 'max_similarity_to_train']
    """
    smiles = df["smiles"].values
    y = df["target"].values

    splits, _ = make_splits(
        smiles, y, split_type=split_type, seed=seed, n_splits=n_splits
    )

    records = []
    for fold_idx, split in enumerate(splits, start=1):
        train_idx, _, test_idx = split

        train_smiles = smiles[train_idx]
        test_smiles = smiles[test_idx]

        max_sims = compute_max_similarities(test_smiles.tolist(), train_smiles.tolist())

        for s, sim in zip(test_smiles, max_sims):
            records.append(
                {
                    "smiles": s,
                    "fold": fold_idx,
                    "max_similarity_to_train": sim,
                }
            )

    return pd.DataFrame(records)


def plot_applicability_domain(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    plot_type: str = "line",
    save_path: str | Path = None,
):
    """Plot metric across similarity bins for all models using scienceplots styling."""
    if results_df is None or results_df.empty or "model" not in results_df.columns:
        print("Warning: results_df is empty or missing 'model' column. Cannot plot.")
        return

    plot_type = plot_type.lower()
    if plot_type not in ["line", "bar", "heatmap"]:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Expected one of ['line', 'bar', 'heatmap']"
        )

    import matplotlib.pyplot as plt
    import seaborn as sns

    from tox_ames.utils.style import (
        MODEL_STYLES,
        get_manuscript_label,
        get_model_style,
    )

    plt.style.use("science")

    def slabel(style):
        # Short manuscript label, identical to Table 3 and the other figures.
        return get_manuscript_label(style.name)

    available_models = set(results_df["model"].unique())
    ordered_models = []
    palette = {}

    # First, add predefined models in their clean family order
    for style in MODEL_STYLES:
        if style.name in available_models or style.label in available_models:
            ordered_models.append(style)
            palette[slabel(style)] = style.color

    # Next, add any unrecognized models (e.g. unfinished or custom runs) with fallback styling
    for model_name in sorted(available_models):
        mapped_label = get_manuscript_label(model_name)
        if mapped_label not in palette:
            style = get_model_style(model_name)
            ordered_models.append(style)
            palette[slabel(style)] = style.color

    plot_df = results_df.copy()
    plot_df["Model Label"] = plot_df["model"].map(get_manuscript_label)

    metric_title = metric.upper().replace("_", "-")

    if plot_type == "line":
        plt.figure(figsize=(8, 6), dpi=300)
        # Plot line for each model individually to respect custom marker and color
        for style in ordered_models:
            model_data = plot_df[plot_df["Model Label"] == slabel(style)]
            if model_data.empty:
                continue
            summary = (
                model_data.groupby("bin", observed=True, sort=False)[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            plt.errorbar(
                summary["bin"].astype(str),
                summary["mean"],
                yerr=summary["std"],
                marker=style.marker,
                color=style.color,
                ecolor="black",
                label=slabel(style),
                linewidth=2.0,
                markersize=6,
                elinewidth=1.0,
                capsize=3,
            )
        plt.legend(
            title="Model Architectures", bbox_to_anchor=(1.02, 1), loc="upper left"
        )

    elif plot_type == "bar":
        plt.figure(figsize=(10.5, 3.5))
        hue_order = [slabel(style) for style in ordered_models]
        palette_dict = {slabel(style): style.color for style in ordered_models}
        sns.barplot(
            data=plot_df,
            x="bin",
            y=metric,
            hue="Model Label",
            hue_order=hue_order,
            palette=palette_dict,
            edgecolor="black",
            errorbar="sd",
            linewidth=0.5,
            capsize=0.3,
        )
        plt.legend(
            title="Model Architectures", bbox_to_anchor=(1.02, 1), loc="upper left"
        )
        plt.grid(linestyle="--", alpha=0.7, zorder=0)

        # increase tick label font size for better readability
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    elif plot_type == "heatmap":
        pivot_df = plot_df.pivot_table(
            index="Model Label", columns="bin", values=metric, aggfunc="mean"
        )
        row_order = [
            slabel(style) for style in ordered_models if slabel(style) in pivot_df.index
        ]
        pivot_df = pivot_df.reindex(row_order)

        plt.figure(figsize=(8, 6), dpi=150)
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": f"Test {metric_title}"},
        )

    # plt.title(
    #     f"Domain of Applicability Profile ({metric_title} by Structural Similarity to Train Set)",
    #     fontsize=13,
    #     fontweight="bold",
    #     pad=15,
    # )

    if plot_type != "heatmap":
        plt.xlabel(
            "Tanimoto Similarity to Nearest Training Molecule ($T_{max}$)",
            fontsize=12,
            labelpad=10,
        )
        plt.ylabel(f"Test {metric_title}", fontsize=11, labelpad=10)

        # Dynamic y-limit based on metric value range
        min_val = plot_df[metric].min()
        if pd.notna(min_val) and min_val < 0.5:
            plt.ylim(max(0.0, min_val - 0.05), 1.02)
        else:
            plt.ylim(0.6, 1.02)
    else:
        plt.xlabel(
            "Tanimoto Similarity to Nearest Training Molecule ($T_{max}$)",
            labelpad=10,
        )
        plt.ylabel("Model Architecture", labelpad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
