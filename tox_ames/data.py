from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

import lightning.pytorch as L
import pandas as pd
from chemprop import data, featurizers
from datasets import Dataset, DatasetDict
from rdkit import Chem
from skfp.fingerprints import (
    AvalonFingerprint,
    ECFPFingerprint,
    MACCSFingerprint,
    RDKit2DDescriptorsFingerprint,
    RDKitFingerprint,
    PubChemFingerprint,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


def get_fingerprint(smiles, fp_type="ECFP", **kwargs):
    if fp_type == "ECFP":
        fp = ECFPFingerprint(**kwargs)
    elif fp_type == "Avalon":
        fp = AvalonFingerprint(**kwargs)
    elif fp_type == "RDKit":
        fp = RDKitFingerprint(**kwargs)
    elif fp_type == "MACCS":
        if "fp_size" in kwargs:
            kwargs.pop("fp_size")
        fp = MACCSFingerprint(**kwargs)
    elif fp_type == "RDKit2D":
        if "fp_size" in kwargs:
            kwargs.pop("fp_size")
        fp = RDKit2DDescriptorsFingerprint(**kwargs)
    elif fp_type == "PubChem":
        if "fp_size" in kwargs:
            kwargs.pop("fp_size")
        fp = PubChemFingerprint(**kwargs)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
    if isinstance(smiles, list):
        return fp.transform(smiles)
    return fp.transform([smiles])[0]


@dataclass
class MPNNDataModule(L.LightningDataModule):
    data_mols: list
    split: tuple
    config: dict

    def __post_init__(self):
        super().__init__()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def setup(self, stage=None):
        train_ids, val_ids, test_ids = self.split
        self.train_dset = data.MoleculeDataset(
            [self.data_mols[i] for i in train_ids], self.featurizer
        )
        self.val_dset = data.MoleculeDataset(
            [self.data_mols[i] for i in val_ids], self.featurizer
        )
        self.test_dset = data.MoleculeDataset(
            [self.data_mols[i] for i in test_ids], self.featurizer
        )

    def train_dataloader(self):
        return data.build_dataloader(
            self.train_dset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            shuffle=self.config.train_shuffle,
            drop_last=self.config.train_drop_last,
            class_balance=self.config.train_cls_balance,
        )

    def val_dataloader(self):
        return data.build_dataloader(
            self.val_dset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            drop_last=self.config.drop_last,
        )

    def test_dataloader(self):
        return data.build_dataloader(
            self.test_dset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            drop_last=self.config.drop_last,
        )


@dataclass
class ChemBERTaDataModule(L.LightningDataModule):
    df: pd.DataFrame
    split: tuple
    config: dict

    def __post_init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _train_transform(self, batch):
        row_indices = batch["row_idx"]

        # Handle both single item and batched queries
        if not isinstance(row_indices, list):
            variant = random.choice(self.train_tokenized_pool[row_indices])
            return {
                "input_ids": variant["input_ids"],
                "attention_mask": variant["attention_mask"],
                "labels": batch[self.config.label_col],
            }

        batch_input_ids = []
        batch_attention_mask = []
        for idx in row_indices:
            variant = random.choice(self.train_tokenized_pool[idx])
            batch_input_ids.append(variant["input_ids"])
            batch_attention_mask.append(variant["attention_mask"])

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch[self.config.label_col],
        }

    def setup(self, stage: str = None):
        # 1. Attach a unique index to all samples
        df_copy = self.df.copy()
        df_copy["row_idx"] = list(range(len(self.df)))
        raw_ds = Dataset.from_pandas(df_copy)

        train_idx, val_idx, test_idx = self.split

        # 2. Pre-tokenize validation and test sets (no transforms needed during training)
        def _static_tokenize(batch):
            return self.tokenizer(
                batch[self.config.smiles_col],
                truncation=True,
                max_length=self.config.max_length,
            )

        val_ds = raw_ds.select(val_idx).map(
            _static_tokenize, batched=True, desc="Tokenizing val set"
        )
        val_ds = val_ds.select_columns(
            ["input_ids", "attention_mask", self.config.label_col]
        )
        val_ds = val_ds.rename_column(self.config.label_col, "labels")
        val_ds.set_format(type="torch")

        test_ds = raw_ds.select(test_idx).map(
            _static_tokenize, batched=True, desc="Tokenizing test set"
        )
        test_ds = test_ds.select_columns(
            ["input_ids", "attention_mask", self.config.label_col]
        )
        test_ds = test_ds.rename_column(self.config.label_col, "labels")
        test_ds.set_format(type="torch")

        # 3. Precompute a pool of 5 randomized + tokenized variants for each training sample
        train_df = df_copy.iloc[train_idx]
        self.train_tokenized_pool = {}

        smiles_list = train_df[self.config.smiles_col].tolist()
        row_indices = train_df["row_idx"].tolist()

        num_variants = 5
        for row_idx, s in zip(row_indices, smiles_list):
            variants = [s]  # Always include the original SMILES
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    for _ in range(num_variants - 1):
                        aug_s = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                        variants.append(aug_s)
            except Exception:
                pass

            tokenized_vars = []
            for var in variants:
                t = self.tokenizer(
                    var, truncation=True, max_length=self.config.max_length
                )
                tokenized_vars.append(
                    {
                        "input_ids": t["input_ids"],
                        "attention_mask": t["attention_mask"],
                    }
                )
            self.train_tokenized_pool[row_idx] = tokenized_vars

        # 4. Create train dataset with transform lookup
        train_ds = raw_ds.select(train_idx)
        train_ds.set_transform(self._train_transform)

        self.ds = DatasetDict(
            {
                "train": train_ds,
                "val": val_ds,
                "test": test_ds,
            }
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds["train"],
            collate_fn=self.data_collator,
            batch_size=self.config.batch_size,
            shuffle=self.config.train_shuffle,
            drop_last=self.config.train_drop_last,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds["val"],
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds["test"],
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
        )


@dataclass
class FpDataModule(L.LightningDataModule):
    df: pd.DataFrame
    split: tuple
    config: dict

    def __post_init__(self):
        super().__init__()
        if self.config.fp_type == "custom":
            feat_cols = [c for c in self.df.columns if c.startswith("feat_")]
            self.config.fp_kwargs["fp_size"] = len(feat_cols)

    def _compute_fps(self, batch):
        fps = get_fingerprint(
            batch[self.config.smiles_col],
            fp_type=self.config.fp_type,
            **self.config.fp_kwargs,
        )
        return {
            "input_features": fps,
            "labels": batch[self.config.label_col],
        }

    def setup(self, stage: str = None):
        if self.config.fp_type == "custom":
            feat_cols = [c for c in self.df.columns if c.startswith("feat_")]
            try:
                feat_cols = sorted(feat_cols, key=lambda x: int(x.split("_")[1]))
            except Exception:
                feat_cols = sorted(feat_cols)

            features = self.df[feat_cols].values.tolist()
            labels = self.df[self.config.label_col].tolist()

            temp_df = pd.DataFrame({"input_features": features, "labels": labels})
            processed_ds = Dataset.from_pandas(temp_df)
        else:
            raw_ds = Dataset.from_pandas(self.df)
            processed_ds = raw_ds.map(
                self._compute_fps,
                batched=True,
                desc="Precomputing fingerprints",
            )

        processed_ds = processed_ds.select_columns(["input_features", "labels"])
        processed_ds.set_format(type="torch", columns=["input_features", "labels"])

        train_idx, val_idx, test_idx = self.split
        self.ds = DatasetDict(
            {
                "train": processed_ds.select(train_idx),
                "val": processed_ds.select(val_idx),
                "test": processed_ds.select(test_idx),
            }
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds["train"],
            batch_size=self.config.batch_size,
            shuffle=self.config.train_shuffle,
            drop_last=self.config.train_drop_last,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds["val"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds["test"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def build_datamodule(
    config: Config,
    df: pd.DataFrame | None = None,
    split: tuple = (),
    data_mols: list | None = None,
):
    """Instantiate the correct DataModule from a unified config.

    Parameters
    ----------
    config : Config
        One of ``MPNNConfig``, ``ChemBERTaConfig``, or ``FingerprintConfig``.
    df : pd.DataFrame, optional
        Source dataframe (required for ChemBERTa and Fingerprint pipelines).
    split : tuple
        ``(train_idx, val_idx, test_idx)`` index arrays.
    data_mols : list, optional
        Pre-built ``MoleculeDatapoint`` list (required for MPNN pipeline).

    Returns
    -------
    L.LightningDataModule
    """
    if config.model_type in ("mpnn", "mpnn_atom"):
        if data_mols is None:
            raise ValueError("data_mols is required for MPNN pipeline")
        return MPNNDataModule(data_mols=data_mols, split=split, config=config)
    elif config.model_type == "chemberta":
        if df is None:
            raise ValueError("df is required for ChemBERTa pipeline")
        return ChemBERTaDataModule(df=df, split=split, config=config)
    elif config.model_type == "fingerprint":
        if df is None:
            raise ValueError("df is required for Fingerprint pipeline")
        return FpDataModule(df=df, split=split, config=config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
