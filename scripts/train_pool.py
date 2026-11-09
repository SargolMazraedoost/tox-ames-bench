"""
Batch Training Pool for Deep Learning Pipelines
===============================================

Runs cross-validation across all datasets (combined and individual sources)
for ChemBERTa, Chemprop D-MPNN, and Fingerprint baseline configurations.
"""

import logging
from pathlib import Path

import datamol as dm
import pandas as pd

from tox_ames.config import ChemBERTaConfig, FingerprintConfig, MPNNConfig
from tox_ames.train import run_cv
from tox_ames.utils import get_logger

log = get_logger(__name__)
dm.disable_rdkit_log()
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"

datasets = {
    "combined": DATA_DIR / "ames_final.csv",
    "hansen": SOURCES_DIR / "ds_hansen.csv",
    "tdc": SOURCES_DIR / "ds_tdc.csv",
    "isstox": SOURCES_DIR / "ds_isstox.csv",
    "pharma": SOURCES_DIR / "ds_pharma.csv",
    "deepk": SOURCES_DIR / "ds_deepk.csv",
}

dataset_names = ["hansen", "tdc", "isstox", "pharma", "deepk", "combined"]
SPLIT_TYPE = "SCAFFOLD_BALANCED"
SUFFIX = "scf" if SPLIT_TYPE == "SCAFFOLD_BALANCED" else "rnd"
SMILES_COL = "smiles"
TARGET_COL = "target"
N_FOLDS = 25
MAX_EPOCHS = 30


def run_batch_training():
    for ds_name in dataset_names:
        PREFIX = ds_name
        data_file = datasets[ds_name]
        if not data_file.exists():
            log.warning(f"Dataset {ds_name} not found at {data_file}, skipping.")
            continue

        log.info(f"Loading data for {ds_name} from {data_file}...")
        df = pd.read_csv(data_file)

        # ── 1. Chemprop D-MPNN ──────────────────────────────────────────────
        # model_types = ["mpnn", "mpnn_atom"]
        # aggregates = ["norm", "mean", "attn"]
        # for mt in model_types:
        #     for agg in aggregates:
        #         config = MPNNConfig(
        #             model_type=mt,
        #             log_dir=f"{PREFIX}/exp-mpnn_{mt}_{agg}-{SUFFIX}",
        #             init_lr=1e-4,
        #             max_lr=1e-3,
        #             final_lr=1e-5,
        #             aggregation=agg,
        #             n_splits=N_FOLDS,
        #             split_type=SPLIT_TYPE,
        #             max_epochs=MAX_EPOCHS,
        #         )
        #         results, summary, predictions = run_cv(df, config)

        # ── 2. Fingerprints ──────────────────────────────────────────────────
        # fp_types = {
        #     "Avalon": 512,
        #     "MACCS": 167,
        #     "ECFP": 1024,
        #     "PubChem": None,
        # }
        # for fp_type, fp_size in fp_types.items():
        #     fp_kwargs = {"fp_size": fp_size} if fp_size is not None else {}
        #     config = FingerprintConfig(
        #         fp_type=fp_type,
        #         fp_kwargs=fp_kwargs,
        #         log_dir=f"{PREFIX}/exp-fp_{fp_type.lower()}-{SUFFIX}",
        #         lr=5e-3,
        #         gradient_clip_val=5.0,
        #         n_splits=N_FOLDS,
        #         split_type=SPLIT_TYPE,
        #         max_epochs=MAX_EPOCHS,
        #     )
        #     results, summary, predictions = run_cv(df, config)

        # ── 3. ChemBERTa with LoRA / PEFT ────────────────────────────────────
        model_types = ["77M-MTR", "77M-MLM"]
        pool_methods = ["mean"]

        for mt in model_types:
            for agg in pool_methods:
                config = ChemBERTaConfig(
                    model_name=f"DeepChem/ChemBERTa-{mt}",
                    log_dir=f"{PREFIX}/exp-chemberta_{mt}_{agg}_lora-{SUFFIX}",
                    pooling=agg,
                    lr=3e-4,
                    use_lora=True,
                    r=8,
                    lora_alpha=16,
                    freeze_backbone=False,
                    weight_decay=0.1,
                    warmup_steps=500,
                    gradient_clip_val=1.0,
                    n_splits=N_FOLDS,
                    split_type=SPLIT_TYPE,
                    max_epochs=MAX_EPOCHS,
                )
                results, summary, predictions = run_cv(df, config)


if __name__ == "__main__":
    run_batch_training()
