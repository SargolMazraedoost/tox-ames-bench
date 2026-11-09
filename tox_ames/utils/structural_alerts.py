from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# Define Benigni-Bossa structural alerts from JRC43157 (Toxtree module)
# Keys are the alert numbers/descriptions, values can be a single SMARTS or a list of SMARTS
STRUCTURAL_ALERTS = {
    "SA_1: Acyl Halides": "[CX3](=[OX1])[F,Cl,Br,I]",
    "SA_2: Alkyl/Benzyl Ester of Sulfonic/Phosphonic Acid": [
        "[CX4,c][S](=O)(=O)[OX2][CX4]",
        "[CX4,c][P](=O)(O)[OX2][CX4]",
    ],
    "SA_3: N-Methylol Derivatives": "[NX3][CH2][OX2H]",
    "SA_4: Monohaloalkene": "[CX3](=[CX3])[F,Cl,Br,I]",
    "SA_5: S or N Mustard": ["[S][CH2][CH2][Cl,Br,I]", "[NX3][CH2][CH2][Cl,Br,I]"],
    "SA_6: Propiolactones or Propiosultones": ["C1CC(=O)O1", "C1COS(=O)(=O)1"],
    "SA_7: Epoxides and Aziridines": ["C1OC1", "C1NC1"],
    "SA_8: Aliphatic Halogens": "[Cl,Br,I][CX4]",
    "SA_9: Alkyl Nitrite": "CON=O",
    "SA_10: Alpha, Beta-Unsaturated Carbonyls": "[CX3]=[CX3]-[CX3]=[OX1]",
    "SA_11: Simple Aldehyde": "[CX3H1](=[OX1])[#6,#1]",
    "SA_12: Quinones": [
        "O=C1[C,c]=[C,c]C(=O)[C,c]=[C,c]1",
        "O=C1C(=O)[C,c]=[C,c][C,c]=[C,c]1",
    ],
    "SA_13: Hydrazine": "[NX3;H1,H2,H3][NX3;H1,H2,H3]",
    "SA_14: Aliphatic Azo and Azoxy": ["[CX4][N]=[N][CX4]", "[CX4][N+](=O)=[N-][CX4]"],
    "SA_15: Isocyanate and Isothiocyanate Groups": "[NX2]=[CX2]=[O,S]",
    "SA_16: Alkyl Carbamate and Thiocarbamate": [
        "[NX3][CX3](=[OX1])[OX2][CX4]",
        "[NX3][CX3](=[SX1])[OX2][CX4]",
    ],
    "SA_17: Thiocarbonyl (Nongenotoxic)": "[CX3]=[SX1]",
    "SA_18: Polycyclic Aromatic Hydrocarbons": [
        "[c]12c(ccc3ccccc13)cccc2",
        "c1ccc2c(c1)cc3ccccc3c2",
        "c1ccc2c(c1)ccc3ccccc23",
    ],
    "SA_19: Heterocyclic Polycyclic Aromatic Hydrocarbons": [
        "c1ccc2c(c1)cc3cnccc3c2",
        "c1ccc2c(c1)cc3c(c2)cncc3",
    ],
    "SA_21: Alkyl and Aryl N-Nitroso Groups": "[NX3][N]=[O]",
    "SA_22: Azide and Triazene Groups": ["[N-]=[N+]=[N]", "[NX3][N]=[NX2]"],
    "SA_23: Aliphatic N-Nitro Group": "[NX3][N+](=O)[O-]",
    "SA_24: Alpha, Beta-Unsaturated Aliphatic Alkoxy Group": "[CX3]=[CX3][OX2][CX4]",
    # NX2 excludes the three-coordinate nitrogen of an aromatic nitro group.
    "SA_25: Aromatic Nitroso Group": "a[NX2]=[OX1]",
    "SA_26: Aromatic Ring N-Oxide": "[n+][o-]",
    "SA_27: Nitro-Aromatic": "[N+](=O)([O-])a",
    "SA_28: Primary Aromatic Amine / Hydroxylamine": ["[N;H2]a", "a[NH][OH]"],
    "SA_28bis: Aromatic Mono- and Dialkylamine": "a[NX3]([CX4])[CX4,H]",
    "SA_28ter: Aromatic N-Acyl Amine": "a[NX3]C(=O)[CX4,H]",
    # Exclude the terminal N=N of aryl azides, which SA_22 covers.
    "SA_29: Aromatic Diazo": "a[N]=[N;!$([N+]=[N-])]",
    "SA_30: Coumarins and Furocoumarins": [
        "O=C1OC=Cc2ccccc12",
        "O=C1oc=cc2c1ccc3occc23",
    ],
}


def analyze_structural_alerts(exp_name: str):
    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "notebooks" / "logs" / "combined"
    exp_dir = logs_dir / exp_name
    pred_path = exp_dir / "cv_predictions.csv"

    if not pred_path.exists():
        print(f"Prediction file not found for {exp_name} at: {pred_path}")
        return

    print(f"Analyzing structural alerts for model: {exp_name}...")
    df = pd.read_csv(pred_path)

    # Cache RDKit mol objects to speed up SMARTS matching
    mols = []
    for smi in df["smiles"].values:
        try:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)
        except:
            mols.append(None)

    alert_results = []

    for alert_name, smarts_input in STRUCTURAL_ALERTS.items():
        if isinstance(smarts_input, str):
            smarts_list = [smarts_input]
        else:
            smarts_list = smarts_input

        patterns = []
        for s in smarts_list:
            pat = Chem.MolFromSmarts(s)
            if pat is not None:
                patterns.append(pat)
            else:
                print(f"Warning: Invalid SMARTS pattern '{s}' for alert '{alert_name}'")

        if not patterns:
            continue

        # Get indices of molecules matching any of the SMARTS patterns
        matched_mask = []
        for mol in mols:
            if mol is None:
                matched_mask.append(False)
            else:
                matched_mask.append(any(mol.HasSubstructMatch(p) for p in patterns))

        matched_mask = np.array(matched_mask)
        matched_df = df[matched_mask]

        total_matched = len(matched_df)
        smarts_rep = ", ".join(smarts_list)

        if total_matched == 0:
            alert_results.append(
                {
                    "Structural Alert": alert_name,
                    "SMARTS": smarts_rep,
                    "Total Count": 0,
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                    "Sensitivity": np.nan,
                    "Specificity": np.nan,
                    "Accuracy": np.nan,
                }
            )
            continue

        y_true = matched_df["y_actual"].values
        y_pred = matched_df["y_pred"].values

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        acc = (tp + tn) / total_matched

        alert_results.append(
            {
                "Structural Alert": alert_name,
                "SMARTS": smarts_rep,
                "Total Count": total_matched,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Sensitivity": sens,
                "Specificity": spec,
                "Accuracy": acc,
            }
        )

    results_df = pd.DataFrame(alert_results)

    # Save CSV
    out_csv = exp_dir / "structural_alerts_performance.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Structural alerts analysis saved to: {out_csv}")
    return results_df


def plot_structural_alerts(
    df_alerts: pd.DataFrame,
    save_path: str | Path | None = None,
    model_name: str | None = None,
):
    """
    Plots structural alerts prevalence and model performance.

    Args:
        df_alerts: DataFrame containing columns: Structural Alert, Total Count, Sensitivity, Specificity, Accuracy
        save_path: Optional path to save the generated figure.
        model_name: Optional model name to include in the plot title.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from tox_ames.utils.style import set_science_style

    # Set styling
    set_science_style()

    # Filter out alerts that have 0 occurrences and sort by Total Count descending
    df_plot = df_alerts[df_alerts["Total Count"] > 0].copy()

    # Ensure numeric types
    for col in ["Sensitivity", "Specificity", "Accuracy"]:
        if col in df_plot.columns and df_plot[col].dtype == object:
            df_plot[col] = df_plot[col].astype(str).str.rstrip("%")
            df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce") / 100.0

    if "Total Count" in df_plot.columns and df_plot["Total Count"].dtype == object:
        df_plot["Total Count"] = pd.to_numeric(
            df_plot["Total Count"].astype(str).str.replace(",", ""), errors="coerce"
        )

    df_plot = df_plot.sort_values(by="Total Count", ascending=False).reset_index(
        drop=True
    )

    if len(df_plot) == 0:
        print("No structural alerts with positive counts to plot.")
        return

    # Create figure with 2 panels side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    # Plot 1: Prevalence
    max_count = df_plot["Total Count"].max()
    axes[0].set_xlim(0, max_count * 1.15)  # leave space for labels

    # Muted crest gradient color palette
    colors = sns.color_palette("crest_r", n_colors=len(df_plot))

    sns.barplot(
        data=df_plot,
        x="Total Count",
        y="Structural Alert",
        ax=axes[0],
        palette=colors,
        hue="Structural Alert",
        legend=False,
    )

    # Add count labels
    for p in axes[0].patches:
        width = p.get_width()
        if not np.isnan(width) and width > 0:
            axes[0].text(
                width + (max_count * 0.015),
                p.get_y() + p.get_height() / 2.0,
                f"{int(width)}",
                ha="left",
                va="center",
                fontsize=11,
                weight="bold",
                color="#374151",
            )

    axes[0].set_title(
        "Alert Prevalence (Total Count)", fontsize=14, weight="bold", pad=15
    )
    axes[0].set_xlabel("Count", fontsize=12, labelpad=10)
    axes[0].set_ylabel("Structural Alert", fontsize=12)
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Plot 2: Performance
    # Melt performance metrics to long format
    df_metrics = df_plot.melt(
        id_vars=["Structural Alert"],
        value_vars=["Sensitivity", "Specificity", "Accuracy"],
        var_name="Metric",
        value_name="Score",
    )

    metric_colors = {
        "Accuracy": "#3b82f6",  # Modern Blue
        "Sensitivity": "#ec4899",  # Modern Rose
        "Specificity": "#10b981",  # Modern Emerald
    }

    sns.barplot(
        data=df_metrics,
        x="Score",
        y="Structural Alert",
        hue="Metric",
        ax=axes[1],
        palette=metric_colors,
    )

    axes[1].set_title("Model Performance Metrics", fontsize=14, weight="bold", pad=15)
    axes[1].set_xlabel("Score", fontsize=12, labelpad=10)
    axes[1].set_ylabel("")  # shared y-axis
    axes[1].set_xlim(0, 1.05)
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=11,
    )

    # Main title
    title_suffix = f" (Model: {model_name})" if model_name else ""
    plt.suptitle(
        f"Benigni-Bossa Structural Alerts Analysis{title_suffix}",
        fontsize=16,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def clean_alert_name(name: str) -> str:
    """Removes the SA_XX: prefix from the structural alert name."""
    if ":" in name:
        return name.split(":", 1)[1].strip()
    return name


def plot_alert_class_performance(
    exp_name: str,
    save_path: str | Path,
    top_n: int = 10,
    min_class_support: int = 50,
    pred_file: str = "cv_predictions.csv",
) -> pd.DataFrame:
    """Plot per-alert sensitivity and specificity across repeated holdouts.

    ``pred_file`` selects the prediction table within the experiment directory,
    so the cross-representation ensemble can be scored through this same path.
    """
    import matplotlib.pyplot as plt

    from tox_ames.utils.style import set_science_style

    project_root = Path(__file__).resolve().parent.parent.parent
    pred_path = project_root / "notebooks" / "logs" / "combined" / exp_name / pred_file
    predictions = pd.read_csv(pred_path)
    unique_compounds = predictions.drop_duplicates("smiles")
    molecules = {
        smiles: Chem.MolFromSmiles(smiles) for smiles in unique_compounds["smiles"]
    }

    rows = []
    for alert_name, smarts_input in STRUCTURAL_ALERTS.items():
        smarts = [smarts_input] if isinstance(smarts_input, str) else smarts_input
        patterns = [pattern for item in smarts if (pattern := Chem.MolFromSmarts(item))]
        matching_smiles = {
            smiles
            for smiles, molecule in molecules.items()
            if molecule is not None
            and any(molecule.HasSubstructMatch(pattern) for pattern in patterns)
        }
        matched_unique = unique_compounds[
            unique_compounds["smiles"].isin(matching_smiles)
        ]
        n_positive = int((matched_unique["y_actual"] == 1).sum())
        n_negative = int((matched_unique["y_actual"] == 0).sum())
        if min(n_positive, n_negative) < min_class_support:
            continue

        fold_rates = []
        matched = predictions[predictions["smiles"].isin(matching_smiles)]
        for fold, fold_data in matched.groupby("fold"):
            positive = fold_data[fold_data["y_actual"] == 1]
            negative = fold_data[fold_data["y_actual"] == 0]
            if positive.empty or negative.empty:
                continue
            fold_rates.append(
                {
                    "fold": fold,
                    "sensitivity": float((positive["y_pred"] == 1).mean()),
                    "specificity": float((negative["y_pred"] == 0).mean()),
                }
            )

        rates = pd.DataFrame(fold_rates)
        rows.append(
            {
                "alert": clean_alert_name(alert_name),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "sensitivity_mean": rates["sensitivity"].mean(),
                "sensitivity_sd": rates["sensitivity"].std(ddof=1),
                "specificity_mean": rates["specificity"].mean(),
                "specificity_sd": rates["specificity"].std(ddof=1),
                "folds": len(rates),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .assign(total_support=lambda frame: frame["n_positive"] + frame["n_negative"])
        .nlargest(top_n, "total_support")
        .sort_values("total_support")
        .reset_index(drop=True)
    )
    if summary.empty:
        raise ValueError("No alert classes satisfy the minimum class support.")

    set_science_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 5.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 0.32], "wspace": 0.18},
    )
    y = np.arange(len(summary))
    panels = (
        ("sensitivity", "Ames-positive compounds\nSensitivity", "dodgerblue", "o"),
        ("specificity", "Ames-negative compounds\nSpecificity", "salmon", "s"),
    )
    for panel_index, (axis, (metric, title, color, marker)) in enumerate(
        zip(axes[:2], panels)
    ):
        axis.errorbar(
            summary[f"{metric}_mean"],
            y,
            xerr=summary[f"{metric}_sd"],
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=6,
            capsize=3,
            linewidth=1.4,
        )
        axis.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=0.9)
        axis.set_xlim(0, 1)
        ticks = np.linspace(0, 1, 5)
        labels = [f"{value:.0%}" for value in ticks]
        labels[4 if panel_index == 0 else 0] = ""
        axis.set_xticks(ticks, labels)
        axis.set_xlabel("Rate")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", visible=False)

    axes[0].set_yticks(y, summary["alert"])
    axes[2].set_xlim(0, 1)
    axes[2].axis("off")
    for position, row in summary.iterrows():
        axes[2].text(
            0.5,
            position,
            f"{row.n_positive} / {row.n_negative}",
            ha="center",
            va="center",
        )
    axes[2].text(
        0.5,
        1.02,
        "Unique $n$\n(+ / $-$)",
        ha="center",
        va="bottom",
        fontweight="bold",
        transform=axes[2].transAxes,
    )

    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.12, top=0.86)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary_path = pred_path.parent / "structural_alerts_by_class.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def draw_alert_molecule(
    smiles: str, smarts_list: list[str], size: tuple[int, int] = (150, 120)
) -> str:
    """
    Renders a molecule SVG with the matching alert substructures highlighted.
    """
    import datamol as dm
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = dm.to_mol(smiles)
    if mol is None:
        return ""

    # Generate coordinates for drawing
    mol = dm.copy_mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol)

    highlight_atoms = set()
    highlight_bonds = set()

    for s in smarts_list:
        pat = Chem.MolFromSmarts(s)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            highlight_atoms.update(match)
            # Find bonds between matched atoms
            for i in range(len(match)):
                for j in range(i + 1, len(match)):
                    bond = mol.GetBondBetweenAtoms(match[i], match[j])
                    if bond is not None:
                        highlight_bonds.add(bond.GetIdx())

    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True

    # Soft rose color for highlighting: rgb(244, 63, 94) -> (0.95, 0.25, 0.37)
    atom_cols = {idx: (0.95, 0.25, 0.37) for idx in highlight_atoms}
    bond_cols = {idx: (0.95, 0.25, 0.37) for idx in highlight_bonds}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        highlightAtomColors=atom_cols,
        highlightBonds=list(highlight_bonds),
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Clean up SVG XML headers
    svg_start = svg.find("<svg")
    if svg_start != -1:
        svg = svg[svg_start:]
    return svg


def generate_structural_alerts_table_html(exp_name: str, top_n: int = 5) -> str:
    """
    Generates an HTML table showing the top structural alerts and representative molecules.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from rdkit import Chem

    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "notebooks" / "logs" / "combined"
    exp_dir = logs_dir / exp_name
    pred_path = exp_dir / "cv_predictions.csv"

    if not pred_path.exists():
        print(f"Prediction file not found for {exp_name} at: {pred_path}")
        return ""

    df = pd.read_csv(pred_path)

    # Cache RDKit mol objects
    mols = []
    for smiles in df["smiles"].values:
        try:
            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)
        except:
            mols.append(None)

    # For each alert, identify matching indices
    alert_matches = {}
    for alert_name, smarts_input in STRUCTURAL_ALERTS.items():
        if isinstance(smarts_input, str):
            smarts_list = [smarts_input]
        else:
            smarts_list = smarts_input

        patterns = []
        for s in smarts_list:
            pat = Chem.MolFromSmarts(s)
            if pat is not None:
                patterns.append(pat)

        if not patterns:
            continue

        matched_indices = []
        for idx, mol in enumerate(mols):
            if mol is not None:
                if any(mol.HasSubstructMatch(p) for p in patterns):
                    matched_indices.append(idx)

        alert_matches[alert_name] = {"indices": matched_indices, "smarts": smarts_list}

    # Now calculate positive and negative prediction sets for each alert
    pos_records = []
    neg_records = []

    for alert_name, match_info in alert_matches.items():
        indices = match_info["indices"]
        smarts = match_info["smarts"]
        if not indices:
            continue

        alert_df = df.iloc[indices]

        # Positives: y_pred == 1
        pos_df = alert_df[alert_df["y_pred"] == 1]
        tp = int(np.sum(pos_df["y_actual"] == 1))
        fp = int(np.sum(pos_df["y_actual"] == 0))
        total_pos = len(pos_df)

        if total_pos > 0:
            best_smiles_pos = None
            min_atoms_pos = float("inf")
            for smiles in pos_df["smiles"].values:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    atoms = m.GetNumHeavyAtoms()
                    if atoms < min_atoms_pos:
                        min_atoms_pos = atoms
                        best_smiles_pos = smiles

            pos_records.append(
                {
                    "alert_name": alert_name,
                    "clean_name": clean_alert_name(alert_name),
                    "smarts": smarts,
                    "tp": tp,
                    "fp": fp,
                    "total": total_pos,
                    "smiles": best_smiles_pos,
                }
            )

        # Negatives: y_pred == 0
        neg_df = alert_df[alert_df["y_pred"] == 0]
        tn = int(np.sum(neg_df["y_actual"] == 0))
        fn = int(np.sum(neg_df["y_actual"] == 1))
        total_neg = len(neg_df)

        if total_neg > 0:
            best_smiles_neg = None
            min_atoms_neg = float("inf")
            for smiles in neg_df["smiles"].values:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    atoms = m.GetNumHeavyAtoms()
                    if atoms < min_atoms_neg:
                        min_atoms_neg = atoms
                        best_smiles_neg = smiles

            neg_records.append(
                {
                    "alert_name": alert_name,
                    "clean_name": clean_alert_name(alert_name),
                    "smarts": smarts,
                    "tn": tn,
                    "fn": fn,
                    "total": total_neg,
                    "smiles": best_smiles_neg,
                }
            )

    # Sort Positives by total predicted positive descending
    pos_records = sorted(pos_records, key=lambda x: x["total"], reverse=True)[:top_n]
    # Sort Negatives by total predicted negative descending
    neg_records = sorted(neg_records, key=lambda x: x["total"], reverse=True)[:top_n]

    # Generate SVGs
    for record in pos_records:
        record["svg"] = draw_alert_molecule(record["smiles"], record["smarts"])
    for record in neg_records:
        record["svg"] = draw_alert_molecule(record["smiles"], record["smarts"])

    # Create HTML table
    html_lines = []

    # CSS styling
    html_lines.append("""
    <style>
        .alert-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Nimbus Roman', 'Times New Roman', serif;
            margin: 25px 0;
            font-size: 0.9em;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
            background-color: #ffffff;
        }
        .alert-table th {
            background-color: #f3f4f6;
            color: #374151;
            text-align: center;
            font-weight: bold;
            border: 1px solid #d1d5db;
            padding: 8px;
        }
        .alert-table td {
            padding: 10px;
            border: 1px solid #e5e7eb;
            text-align: center;
            vertical-align: middle;
        }
        .alert-table tr:hover {
            background-color: #f9fafb;
        }
        .alert-table-title {
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            color: #111827;
        }
        .alert-name-label {
            font-size: 0.85em;
            color: #6b7280;
            margin-top: 5px;
            font-style: italic;
        }
        .structure-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
    </style>
    """)

    html_lines.append(
        f'<div class="alert-table-title">Table 7. Most Common Structural Alerts with Respect to the Model Classification Outcomes Depicted (Model: {exp_name})</div>'
    )
    html_lines.append('<table class="alert-table">')

    # Main Header
    html_lines.append("  <thead>")
    html_lines.append("    <tr>")
    html_lines.append(
        '      <th colspan="3" style="border-right: 2px solid #9ca3af; width: 50%;">Positives</th>'
    )
    html_lines.append('      <th colspan="3" style="width: 50%;">Negatives</th>')
    html_lines.append("    </tr>")
    html_lines.append("    <tr>")
    html_lines.append('      <th style="width: 26%;">Structure</th>')
    html_lines.append('      <th style="width: 12%;">True Positives</th>')
    html_lines.append(
        '      <th style="width: 12%; border-right: 2px solid #9ca3af;">False Positives</th>'
    )
    html_lines.append('      <th style="width: 26%;">Structure</th>')
    html_lines.append('      <th style="width: 12%;">True Negatives</th>')
    html_lines.append('      <th style="width: 12%;">False Negatives</th>')
    html_lines.append("    </tr>")
    html_lines.append("  </thead>")

    # Table Body
    html_lines.append("  <tbody>")

    max_rows = max(len(pos_records), len(neg_records))
    for i in range(max_rows):
        html_lines.append("    <tr>")

        # Positive prediction column
        if i < len(pos_records):
            rec = pos_records[i]
            html_lines.append("      <td>")
            html_lines.append('        <div class="structure-container">')
            html_lines.append(f"          {rec['svg']}")
            html_lines.append(
                f'          <div class="alert-name-label">{rec["clean_name"]}</div>'
            )
            html_lines.append("        </div>")
            html_lines.append("      </td>")
            html_lines.append(
                f'      <td style="font-size: 1.1em; font-weight: bold; color: #10b981;">{rec["tp"]}</td>'
            )
            html_lines.append(
                f'      <td style="font-size: 1.1em; font-weight: bold; color: #ef4444; border-right: 2px solid #9ca3af;">{rec["fp"]}</td>'
            )
        else:
            html_lines.append(
                '      <td></td><td></td><td style="border-right: 2px solid #9ca3af;"></td>'
            )

        # Negative prediction column
        if i < len(neg_records):
            rec = neg_records[i]
            html_lines.append("      <td>")
            html_lines.append('        <div class="structure-container">')
            html_lines.append(f"          {rec['svg']}")
            html_lines.append(
                f'          <div class="alert-name-label">{rec["clean_name"]}</div>'
            )
            html_lines.append("        </div>")
            html_lines.append("      </td>")
            html_lines.append(
                f'      <td style="font-size: 1.1em; font-weight: bold; color: #10b981;">{rec["tn"]}</td>'
            )
            html_lines.append(
                f'      <td style="font-size: 1.1em; font-weight: bold; color: #ef4444;">{rec["fn"]}</td>'
            )
        else:
            html_lines.append("      <td></td><td></td><td></td>")

        html_lines.append("    </tr>")

    html_lines.append("  </tbody>")
    html_lines.append("</table>")

    html_content = "\n".join(html_lines)

    # Save the HTML to the log directory
    out_html = exp_dir / "structural_alerts_table.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Structural alerts table saved to: {out_html}")

    return html_content


def draw_alert_molecule_png(
    smiles: str, smarts_list: list[str], size: tuple[int, int] = (160, 130)
) -> bytes:
    """
    Renders a molecule PNG bytes with the matching alert substructures highlighted.
    """
    import datamol as dm
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = dm.to_mol(smiles)
    if mol is None:
        return b""

    # Generate coordinates for drawing
    mol = dm.copy_mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol)

    highlight_atoms = set()
    highlight_bonds = set()

    for s in smarts_list:
        pat = Chem.MolFromSmarts(s)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            highlight_atoms.update(match)
            # Find bonds between matched atoms
            for i in range(len(match)):
                for j in range(i + 1, len(match)):
                    bond = mol.GetBondBetweenAtoms(match[i], match[j])
                    if bond is not None:
                        highlight_bonds.add(bond.GetIdx())

    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True

    # Soft rose color for highlighting: rgb(244, 63, 94) -> (0.95, 0.25, 0.37)
    atom_cols = {idx: (0.95, 0.25, 0.37) for idx in highlight_atoms}
    bond_cols = {idx: (0.95, 0.25, 0.37) for idx in highlight_bonds}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        highlightAtomColors=atom_cols,
        highlightBonds=list(highlight_bonds),
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def generate_structural_alerts_table_png(
    exp_name: str, top_n: int = 5, save_path: str | Path | None = None
):
    """
    Generates a PNG visual layout showing the top structural alerts and representative molecules.
    """
    import io
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from rdkit import Chem

    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "notebooks" / "logs" / "combined"
    exp_dir = logs_dir / exp_name
    pred_path = exp_dir / "cv_predictions.csv"

    if not pred_path.exists():
        print(f"Prediction file not found for {exp_name} at: {pred_path}")
        return None

    df = pd.read_csv(pred_path)

    # Cache RDKit mol objects
    mols = []
    for smiles in df["smiles"].values:
        try:
            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)
        except:
            mols.append(None)

    # For each alert, identify matching indices
    alert_matches = {}
    for alert_name, smarts_input in STRUCTURAL_ALERTS.items():
        if isinstance(smarts_input, str):
            smarts_list = [smarts_input]
        else:
            smarts_list = smarts_input

        patterns = []
        for s in smarts_list:
            pat = Chem.MolFromSmarts(s)
            if pat is not None:
                patterns.append(pat)

        if not patterns:
            continue

        matched_indices = []
        for idx, mol in enumerate(mols):
            if mol is not None:
                if any(mol.HasSubstructMatch(p) for p in patterns):
                    matched_indices.append(idx)

        alert_matches[alert_name] = {"indices": matched_indices, "smarts": smarts_list}

    # Now calculate positive and negative prediction sets for each alert
    pos_records = []
    neg_records = []

    for alert_name, match_info in alert_matches.items():
        indices = match_info["indices"]
        smarts = match_info["smarts"]
        if not indices:
            continue

        alert_df = df.iloc[indices]

        # Positives: y_pred == 1
        pos_df = alert_df[alert_df["y_pred"] == 1]
        tp = int(np.sum(pos_df["y_actual"] == 1))
        fp = int(np.sum(pos_df["y_actual"] == 0))
        total_pos = len(pos_df)

        if total_pos > 0:
            best_smiles_pos = None
            min_atoms_pos = float("inf")
            for smiles in pos_df["smiles"].values:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    atoms = m.GetNumHeavyAtoms()
                    if atoms < min_atoms_pos:
                        min_atoms_pos = atoms
                        best_smiles_pos = smiles

            pos_records.append(
                {
                    "alert_name": alert_name,
                    "clean_name": clean_alert_name(alert_name),
                    "smarts": smarts,
                    "tp": tp,
                    "fp": fp,
                    "total": total_pos,
                    "smiles": best_smiles_pos,
                }
            )

        # Negatives: y_pred == 0
        neg_df = alert_df[alert_df["y_pred"] == 0]
        tn = int(np.sum(neg_df["y_actual"] == 0))
        fn = int(np.sum(neg_df["y_actual"] == 1))
        total_neg = len(neg_df)

        if total_neg > 0:
            best_smiles_neg = None
            min_atoms_neg = float("inf")
            for smiles in neg_df["smiles"].values:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    atoms = m.GetNumHeavyAtoms()
                    if atoms < min_atoms_neg:
                        min_atoms_neg = atoms
                        best_smiles_neg = smiles

            neg_records.append(
                {
                    "alert_name": alert_name,
                    "clean_name": clean_alert_name(alert_name),
                    "smarts": smarts,
                    "tn": tn,
                    "fn": fn,
                    "total": total_neg,
                    "smiles": best_smiles_neg,
                }
            )

    # Sort Positives by total predicted positive descending
    pos_records = sorted(pos_records, key=lambda x: x["total"], reverse=True)[:top_n]
    # Sort Negatives by total predicted negative descending
    neg_records = sorted(neg_records, key=lambda x: x["total"], reverse=True)[:top_n]

    # Create visual layout table using Matplotlib
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.axis("off")

    # Set fonts and colors
    header_font = {"fontsize": 17, "weight": "bold", "color": "#1f2937", "ha": "center"}
    subheader_font = {
        "fontsize": 13,
        "weight": "bold",
        "color": "#4b5563",
        "ha": "center",
    }
    alert_name_font = {
        "fontsize": 11,
        "color": "#4b5563",
        "style": "italic",
        "ha": "center",
    }
    count_font_tp_tn = {
        "fontsize": 16,
        "weight": "bold",
        "color": "#10b981",
        "ha": "center",
        "va": "center",
    }
    count_font_fp_fn = {
        "fontsize": 16,
        "weight": "bold",
        "color": "#ef4444",
        "ha": "center",
        "va": "center",
    }

    # Draw headers
    ax.text(0.275, 0.95, "Positives", **header_font)
    ax.text(0.775, 0.95, "Negatives", **header_font)

    ax.text(0.15, 0.88, "Structure", **subheader_font)
    ax.text(0.33, 0.88, "True\nPositives", **subheader_font)
    ax.text(0.44, 0.88, "False\nPositives", **subheader_font)

    ax.text(0.65, 0.88, "Structure", **subheader_font)
    ax.text(0.83, 0.88, "True\nNegatives", **subheader_font)
    ax.text(0.94, 0.88, "False\nNegatives", **subheader_font)

    # Draw horizontal rules
    ax.plot([0.02, 0.98], [0.98, 0.98], color="#9ca3af", linewidth=2.0)  # top rule
    ax.plot(
        [0.02, 0.98], [0.85, 0.85], color="#d1d5db", linewidth=1.5
    )  # header divider
    ax.plot([0.02, 0.98], [0.02, 0.02], color="#9ca3af", linewidth=2.0)  # bottom rule

    # Draw vertical divider
    ax.plot([0.5, 0.5], [0.02, 0.98], color="#9ca3af", linewidth=1.5, linestyle="--")

    max_rows = max(len(pos_records), len(neg_records))
    row_height = 0.80 / max_rows

    for i in range(max_rows):
        y = 0.80 - (i * row_height) - (row_height / 2.0)

        # Positives
        if i < len(pos_records):
            rec = pos_records[i]
            img_data = draw_alert_molecule_png(
                rec["smiles"], rec["smarts"], size=(160, 130)
            )
            if img_data:
                img = plt.imread(io.BytesIO(img_data))
                imagebox = OffsetImage(img, zoom=0.65)
                ab = AnnotationBbox(imagebox, (0.15, y + 0.02), frameon=False)
                ax.add_artist(ab)
            ax.text(0.15, y - 0.05, rec["clean_name"], **alert_name_font)
            ax.text(0.33, y, str(rec["tp"]), **count_font_tp_tn)
            ax.text(0.44, y, str(rec["fp"]), **count_font_fp_fn)

        # Negatives
        if i < len(neg_records):
            rec = neg_records[i]
            img_data = draw_alert_molecule_png(
                rec["smiles"], rec["smarts"], size=(160, 130)
            )
            if img_data:
                img = plt.imread(io.BytesIO(img_data))
                imagebox = OffsetImage(img, zoom=0.65)
                ab = AnnotationBbox(imagebox, (0.65, y + 0.02), frameon=False)
                ax.add_artist(ab)
            ax.text(0.65, y - 0.05, rec["clean_name"], **alert_name_font)
            ax.text(0.83, y, str(rec["tn"]), **count_font_tp_tn)
            ax.text(0.94, y, str(rec["fn"]), **count_font_fp_fn)

    # Title
    fig.suptitle(
        f"Most Common Structural Alerts with Classification Outcomes (Model: {exp_name})",
        fontsize=18,
        weight="bold",
        y=0.995,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Table PNG saved to: {save_path}")

    plt.close()
    return fig


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "exp-chemberta_77M-MTR_mean-scf"
    analyze_structural_alerts(model_name)
