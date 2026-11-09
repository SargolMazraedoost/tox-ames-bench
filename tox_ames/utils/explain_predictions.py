from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import confusion_matrix


def analyze_model_errors(exp_name: str, confidence_threshold: float = 0.80):
    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "notebooks" / "logs" / "combined"
    exp_dir = logs_dir / exp_name
    pred_path = exp_dir / "cv_predictions.csv"

    if not pred_path.exists():
        print(f"Prediction file not found for {exp_name} at: {pred_path}")
        return

    print(f"Analyzing predictions for model: {exp_name}...")
    df = pd.read_csv(pred_path)

    # Define Best Predictions: Correct and Confident
    # Probability close to 1 for active (>= 1 - delta) or close to 0 for inactive (<= delta)
    best_df = df[
        ((df["y_actual"] == 1) & (df["y_prob"] >= confidence_threshold)) |
        ((df["y_actual"] == 0) & (df["y_prob"] <= 1 - confidence_threshold))
    ].copy()

    # Define Worst Predictions: High-Confidence Errors (False Positives / False Negatives)
    worst_df = df[
        ((df["y_actual"] == 0) & (df["y_prob"] >= confidence_threshold)) |
        ((df["y_actual"] == 1) & (df["y_prob"] <= 1 - confidence_threshold))
    ].copy()

    print(f"  Total samples: {len(df)}")
    print(f"  Best predictions (correct, high-confidence): {len(best_df)}")
    print(f"  Worst predictions (errors, high-confidence): {len(worst_df)}")

    # ── 1. Bemis-Murcko Scaffold Analysis ──────────────────────────────────
    def get_scaffolds(smiles_list):
        scaffs = []
        for smi in smiles_list:
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
                if sc:  # If the molecule has a scaffold (acyclic ones don't have rings)
                    scaffs.append(sc)
                else:
                    scaffs.append("Acyclic / No Scaffold")
            except:
                scaffs.append("Invalid SMILES")
        return scaffs

    best_df["scaffold"] = get_scaffolds(best_df["smiles"].tolist())
    worst_df["scaffold"] = get_scaffolds(worst_df["smiles"].tolist())

    best_scaff_counts = best_df["scaffold"].value_counts()
    worst_scaff_counts = worst_df["scaffold"].value_counts()

    # Save CSVs of scaffold counts
    best_scaff_counts.to_csv(exp_dir / "best_prediction_scaffolds.csv", index_label="scaffold", header=["count"])
    worst_scaff_counts.to_csv(exp_dir / "worst_prediction_scaffolds.csv", index_label="scaffold", header=["count"])

    # Draw Top 5 Scaffolds in Best Predictions
    draw_top_scaffolds(best_scaff_counts, exp_dir / "best_prediction_scaffolds.png", "Top Scaffolds in Best Predictions")
    # Draw Top 5 Scaffolds in Worst Predictions
    draw_top_scaffolds(worst_scaff_counts, exp_dir / "worst_prediction_scaffolds.png", "Top Scaffolds in Worst Predictions")

    # ── 2. Morgan Fingerprint Bit Environment Analysis ─────────────────────
    # We find which Morgan bits are highly enriched in "Best" vs "Worst" prediction sets
    # We compute bit occurrences and save representative fragment environments
    compute_and_draw_enriched_bits(
        best_df["smiles"].tolist(),
        worst_df["smiles"].tolist(),
        exp_dir,
        num_bits=6
    )


def draw_top_scaffolds(scaff_counts: pd.Series, output_path: Path, title: str):
    top_scaffs = scaff_counts.index[:6]  # Get top 6 scaffolds
    mols = []
    legends = []
    
    for idx, sc in enumerate(top_scaffs):
        if sc in ("Acyclic / No Scaffold", "Invalid SMILES"):
            continue
        mol = Chem.MolFromSmiles(sc)
        if mol is not None:
            mols.append(mol)
            count = scaff_counts[sc]
            legends.append(f"Rank {idx+1} (Count: {count})")

    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(250, 200),
            legends=legends,
            useSVG=False
        )
        if hasattr(img, "save"):
            img.save(output_path)
        elif hasattr(img, "data"):
            with open(output_path, "wb") as f:
                f.write(img.data)
        else:
            raise AttributeError("RDKit image object has neither save nor data attribute.")
        print(f"  Scaffold grid plot saved to: {output_path}")


def compute_and_draw_enriched_bits(best_smiles, worst_smiles, exp_dir: Path, num_bits: int = 6):
    # Map to track bit environments: bit_id -> list of (mol, atom_id, radius)
    bit_env_map = {}
    
    best_counts = np.zeros(1024, dtype=int)
    for smiles in best_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bi = {}
        # Calculate Morgan FP of radius 2 (ECFP4 equivalent)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bi)
        for bit_id in bi.keys():
            best_counts[bit_id] += 1
            if bit_id not in bit_env_map:
                bit_env_map[bit_id] = []
            # Store first few instances of the environment
            if len(bit_env_map[bit_id]) < 5:
                # bi[bit_id] is a tuple of (atom_id, radius)
                for atom_id, rad in bi[bit_id]:
                    bit_env_map[bit_id].append((mol, atom_id, rad))

    worst_counts = np.zeros(1024, dtype=int)
    for smiles in worst_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bi)
        for bit_id in bi.keys():
            worst_counts[bit_id] += 1
            if bit_id not in bit_env_map:
                bit_env_map[bit_id] = []
            if len(bit_env_map[bit_id]) < 5:
                for atom_id, rad in bi[bit_id]:
                    bit_env_map[bit_id].append((mol, atom_id, rad))

    n_best = max(1, len(best_smiles))
    n_worst = max(1, len(worst_smiles))

    # Calculate odds ratios/enrichments in Worst predictions vs Best predictions
    # to identify toxicophores or structures driving the model to make errors
    enrichments_in_worst = []
    for bit_id in range(1024):
        p_best = best_counts[bit_id] / n_best
        p_worst = worst_counts[bit_id] / n_worst
        
        # Avoid division by zero
        enrichment = (p_worst + 1e-5) / (p_best + 1e-5)
        enrichments_in_worst.append((bit_id, enrichment, worst_counts[bit_id], best_counts[bit_id]))

    # Sort to find bits highly enriched in errors (worst predictions)
    enrichments_in_worst.sort(key=lambda x: x[1], reverse=True)

    # Draw the top bits (fragments)
    draw_bits_grid(enrichments_in_worst, bit_env_map, exp_dir / "worst_prediction_enriched_fragments.png", "Enriched in Confident Errors")

    # Sort to find bits highly enriched in correct (best predictions)
    enrichments_in_best = sorted(enrichments_in_worst, key=lambda x: x[1], reverse=False)
    draw_bits_grid(enrichments_in_best, bit_env_map, exp_dir / "best_prediction_enriched_fragments.png", "Enriched in Confident Correct")


def draw_bits_grid(enrichment_list, bit_env_map, output_path: Path, label_prefix: str):
    mols = []
    legends = []
    subattrs = []

    count = 0
    for bit_id, score, count_worst, count_best in enrichment_list:
        if count >= 6:
            break
        if bit_id not in bit_env_map or not bit_env_map[bit_id]:
            continue
            
        # Draw from first available molecule instance
        mol, atom_id, radius = bit_env_map[bit_id][0]
        
        mols.append(mol)
        legends.append(f"Bit {bit_id} (Enr: {score:.1f}x)\nErr: {count_worst} | Ok: {count_best}")
        subattrs.append((atom_id, radius))
        count += 1

    if mols:
        # DrawMorganBit environment highlighting
        images = []
        for mol, (atom_id, radius) in zip(mols, subattrs):
            try:
                # env holds the atoms and bonds in the environment
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                images.append(submol)
            except:
                images.append(mol)

        img = Draw.MolsToGridImage(
            images,
            molsPerRow=3,
            subImgSize=(250, 200),
            legends=legends,
            useSVG=False
        )
        if hasattr(img, "save"):
            img.save(output_path)
        elif hasattr(img, "data"):
            with open(output_path, "wb") as f:
                f.write(img.data)
        else:
            raise AttributeError("RDKit image object has neither save nor data attribute.")
        print(f"  Fragment environment grid plot saved to: {output_path}")


if __name__ == "__main__":
    # Test execution for representative models
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "exp-chemberta_77M-MTR_mean-scf"
    analyze_model_errors(model_name)
