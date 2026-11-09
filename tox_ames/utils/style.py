from dataclasses import dataclass

import seaborn as sns


def set_science_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            # "figure.figsize": (6, 4),
            "font.family": "Nimbus Roman",
            # # Legend
            "legend.fontsize": 11,
            "legend.frameon": False,
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Spines
            # "axes.spines.top": False,
            # "axes.spines.right": False,
            # Figure quality
            "figure.dpi": 100,
            "savefig.dpi": 600,
            # # Better spacing
            # "figure.autolayout": True,
            # # Lines
            # "lines.linewidth": 1,
        },
        font_scale=1.5,
    )


DATASET_LABEL_MAPPING = {
    "combined": "Combined",
    "hansen": "Hansen",
    "tdc": "TDC",
    "isstox": "ISSTOX",
    "pharma": "Pharma",
    "deepk": "DeepK",
}


def get_dataset_label(dataset_name: str) -> str:
    """Returns human-readable dataset label."""
    return DATASET_LABEL_MAPPING.get(dataset_name.lower(), dataset_name)


@dataclass
class ModelStyle:
    name: str
    label: str
    marker: str
    color: str
    cmap: str


MODEL_STYLES = [
    # Fingerprints (Purple theme)
    ModelStyle(
        "exp-fp_ecfp-scf",
        "ECFP4 (Descriptor)",
        "s",
        "#dac3e8",
        "Purples",
    ),
    ModelStyle(
        "exp-fp_avalon-scf",
        "Avalon (Descriptor)",
        "s",
        "#ffcbf2",
        "Purples",
    ),
    ModelStyle(
        "exp-fp_maccs-scf",
        "MACCS (Descriptor)",
        "s",
        "#b892ff",
        "Purples",
    ),
    ModelStyle(
        "exp-fp_pubchem-scf",
        "PubChem (Descriptor)",
        "s",
        "#6e44ff",
        "Purples",
    ),
    # Chemprop MPNN (Blue theme)
    ModelStyle(
        "exp-mpnn_mpnn_mean-scf",
        "Chemprop MPNN (mean) (GNN)",
        "^",
        "#e3f2fd",
        "Blues",
    ),
    ModelStyle(
        "exp-mpnn_mpnn_attn-scf",
        "Chemprop MPNN (attn) (GNN)",
        "^",
        "#bbdefb",
        "Blues",
    ),
    ModelStyle(
        "exp-mpnn_mpnn_norm-scf",
        "Chemprop MPNN (norm) (GNN)",
        "^",
        "#90caf9",
        "Blues",
    ),
    ModelStyle(
        "exp-mpnn_mpnn_atom_mean-scf",
        "Chemprop Atom MPNN (mean) (GNN)",
        "^",
        "#64b5f6",
        "Blues",
    ),
    ModelStyle(
        "exp-mpnn_mpnn_atom_attn-scf",
        "Chemprop Atom MPNN (attn) (GNN)",
        "^",
        "#2196f3",
        "Blues",
    ),
    ModelStyle(
        "exp-mpnn_mpnn_atom_norm-scf",
        "Chemprop Atom MPNN (norm) (GNN)",
        "^",
        "#1976d2",
        "Blues",
    ),
    # ChemBERTa (Red theme)
    ModelStyle(
        "exp-chemberta_77M-MLM_mean_fz-scf",
        "ChemBERTa MLM (mean Frozen) (LM)",
        "p",
        "#f6cacc",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MTR_mean_fz-scf",
        "ChemBERTa MTR (mean Frozen) (LM)",
        "p",
        "#f1a7a9",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MLM_cls-scf",
        "ChemBERTa MLM (cls) (LM)",
        "p",
        "#ec8385",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MLM_attn-scf",
        "ChemBERTa MLM (attn) (LM)",
        "p",
        "#e66063",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MLM_mean-scf",
        "ChemBERTa MLM (mean) (LM)",
        "p",
        "#e35053",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MTR_cls-scf",
        "ChemBERTa MTR (cls) (LM)",
        "p",
        "#dd2c2f",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MTR_attn-scf",
        "ChemBERTa MTR (attn) (LM)",
        "p",
        "#bd1f21",
        "Reds",
    ),
    ModelStyle(
        "exp-chemberta_77M-MTR_mean-scf",
        "ChemBERTa MTR (mean) (LM)",
        "p",
        "#ac1c1e",
        "Reds",
    ),
    # Uni-Mol (Green theme)
    ModelStyle(
        "exp-unimolv1_fz-scf",
        "Uni-Mol v1 (3D Frozen)",
        "D",
        "#d8f3dc",
        "Greens",
    ),
    ModelStyle(
        "exp-unimolv2_fz-scf",
        "Uni-Mol v2 (3D Frozen)",
        "D",
        "#95d5b2",
        "Greens",
    ),
    ModelStyle(
        "exp-unimolv1_ft-scf",
        "Uni-Mol v1 (3D FT)",
        "D",
        "#74c69d",
        "Greens",
    ),
    ModelStyle(
        "exp-unimolv2_ft-scf",
        "Uni-Mol v2 (3D FT)",
        "D",
        "#40916c",
        "Greens",
    ),
]

MODEL_STYLE_MAPPING = {style.name: style for style in MODEL_STYLES}

# Canonical short labels for the ten selected pipelines used in the manuscript
# main-text figures and Table 3. Keep these identical to the LaTeX table so a
# reader can map every figure entry to a table row without ambiguity.
MANUSCRIPT_LABELS = {
    "exp-fp_ecfp-scf": "ECFP4",
    "exp-fp_avalon-scf": "Avalon",
    "exp-fp_maccs-scf": "MACCS",
    "exp-fp_pubchem-scf": "PubChem",
    "exp-chemberta_77M-MLM_mean-scf": "ChemBERTa MLM",
    "exp-chemberta_77M-MTR_mean-scf": "ChemBERTa MTR",
    "exp-mpnn_mpnn_atom_norm-scf": "Atom MPNN",
    "exp-mpnn_mpnn_norm-scf": "Bond MPNN",
    "exp-unimolv1_ft-scf": "Uni-Mol v1",
    "exp-unimolv2_ft-scf": "Uni-Mol v2",
}


def get_manuscript_label(model_name: str) -> str:
    """Return the short manuscript label, falling back to the full label."""
    if model_name in MANUSCRIPT_LABELS:
        return MANUSCRIPT_LABELS[model_name]
    return get_model_label(model_name)


def get_model_style(model_name: str) -> ModelStyle:
    """Returns ModelStyle for a given model name."""
    if model_name in MODEL_STYLE_MAPPING:
        return MODEL_STYLE_MAPPING[model_name]
    return ModelStyle(model_name, model_name, "o", "gray", "Grays")


def get_model_label(model_name: str) -> str:
    """Returns human-readable model label."""
    if model_name in MODEL_STYLE_MAPPING:
        return MODEL_STYLE_MAPPING[model_name].label
    return model_name
