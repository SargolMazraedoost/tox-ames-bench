"""Plot descriptive within-holdout ROC-AUC or F1 ranks for the manuscript."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from tox_ames.utils.style import get_model_style, set_science_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "notebooks" / "logs" / "combined"
OUTPUT_DIR = (
    PROJECT_ROOT / "notebooks" / "Figs" / "results" / "combined" / "average_rank"
)
OUTPUT_PATH = OUTPUT_DIR / "plot.png"
ENSEMBLE_OUTPUT_PATH = OUTPUT_DIR / "plot_with_ensemble.png"
F1_OUTPUT_PATH = OUTPUT_DIR / "plot_f1.png"
F1_ENSEMBLE_OUTPUT_PATH = OUTPUT_DIR / "plot_f1_with_ensemble.png"
ENSEMBLE_NAME = "Cross-repr."
ENSEMBLE_PATH = LOGS_DIR / "advanced_analysis" / "ensemble_predictions.csv"
ENSEMBLE_COLOR = "gray"

MODELS = {
    "ECFP4": "exp-fp_ecfp-scf",
    "Avalon": "exp-fp_avalon-scf",
    "MACCS": "exp-fp_maccs-scf",
    "PubChem": "exp-fp_pubchem-scf",
    "ChemBERTa MLM": "exp-chemberta_77M-MLM_mean-scf",
    "ChemBERTa MTR": "exp-chemberta_77M-MTR_mean-scf",
    "Atom MPNN": "exp-mpnn_mpnn_atom_norm-scf",
    "Bond MPNN": "exp-mpnn_mpnn_norm-scf",
    "Uni-Mol v1": "exp-unimolv1_ft-scf",
    "Uni-Mol v2": "exp-unimolv2_ft-scf",
}

METRICS = {
    "roc_auc": {
        "column": "y_prob",
        "score": roc_auc_score,
        "label": "Normalized ROC-AUC rank (higher is better)",
        "paths": (OUTPUT_PATH, ENSEMBLE_OUTPUT_PATH),
    },
    "f1": {
        "column": "y_pred",
        "score": lambda y_true, y_score: f1_score(y_true, y_score, zero_division=0),
        "label": "Normalized F1 rank (higher is better)",
        "paths": (F1_OUTPUT_PATH, F1_ENSEMBLE_OUTPUT_PATH),
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-ensemble",
        action="store_true",
        help="Include the consensus ensemble and save the *_with_ensemble.png variant.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default="roc_auc",
        help="Metric ranked within each holdout.",
    )
    return parser.parse_args()


def plot_normalized_auc_ranks(
    include_ensemble: bool = True, metric: str = "roc_auc"
) -> None:
    """Plot mean normalized ranks with their interquartile ranges."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}.")
    metric_spec = METRICS[metric]
    records = []
    prediction_paths = {
        model: LOGS_DIR / directory / "cv_predictions.csv"
        for model, directory in MODELS.items()
    }
    if include_ensemble:
        prediction_paths[ENSEMBLE_NAME] = ENSEMBLE_PATH

    for model, prediction_path in prediction_paths.items():
        predictions = pd.read_csv(prediction_path)
        for fold, fold_df in predictions.groupby("fold"):
            records.append(
                {
                    "Model": model,
                    "Fold": fold,
                    "Score": metric_spec["score"](
                        fold_df["y_actual"], fold_df[metric_spec["column"]]
                    ),
                }
            )

    scores = pd.DataFrame(records)
    scores["Normalized rank"] = scores.groupby("Fold")["Score"].rank(pct=True)
    summary = (
        scores.groupby("Model")["Normalized rank"]
        .agg(
            mean="mean",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .sort_values("mean")
    )

    set_science_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for y_position, (model, row) in enumerate(summary.iterrows()):
        if model == ENSEMBLE_NAME:
            marker, color = "*", ENSEMBLE_COLOR
        else:
            style = get_model_style(MODELS[model])
            marker, color = style.marker, style.color
        ax.hlines(
            y_position,
            xmin=0,
            xmax=row["mean"],
            color="#c7c7c7",
            linewidth=1.1,
            zorder=1,
        )
        ax.errorbar(
            row["mean"],
            y_position,
            xerr=[[row["mean"] - row["q1"]], [row["q3"] - row["mean"]]],
            fmt=marker,
            color=color,
            ecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=11 if model == ENSEMBLE_NAME else 9,
            elinewidth=2.2,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )

    ax.set_yticks(range(len(summary)), labels=summary.index, color="black")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(metric_spec["label"])
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    fig.tight_layout()
    plain_path, ensemble_path = metric_spec["paths"]
    output_path = ensemble_path if include_ensemble else plain_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    plot_normalized_auc_ranks(
        include_ensemble=args.include_ensemble, metric=args.metric
    )
