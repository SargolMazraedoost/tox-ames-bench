import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chemprop import data
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import auc, confusion_matrix, roc_curve


def make_splits(smiles, y, split_type="random", seed=42, n_splits=5):
    # build datapoints + mols
    all_data = [
        data.MoleculeDatapoint.from_smi(smi, [int(y)]) for smi, y in zip(smiles, y)
    ]

    mols = [d.mol for d in all_data]

    # get split indices
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols,
        split=split_type,
        sizes=(0.8, 0.1, 0.1),
        seed=seed,
        num_replicates=n_splits,
    )

    # bundle per replicate
    splits = list(zip(train_indices, val_indices, test_indices))

    return splits, all_data


def _flatten_prediction_batches(pred_batches):
    flat = []
    for batch in pred_batches:
        if isinstance(batch, torch.Tensor):
            arr = batch.detach().cpu().numpy()
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            first = batch[0]
            if isinstance(first, torch.Tensor):
                arr = first.detach().cpu().numpy()
            else:
                arr = np.asarray(first)
        else:
            arr = np.asarray(batch)
        flat.append(arr.reshape(-1))

    if not flat:
        return np.array([], dtype=float)

    y_score = np.concatenate(flat).astype(float)

    # If outputs are logits, convert to probabilities.
    if np.any(y_score < 0.0) or np.any(y_score > 1.0):
        y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_score, -20, 20)))
    else:
        y_prob = y_score

    return y_prob


def build_fold_predictions(pred_batches, y, test_idx, smiles, fold, threshold=0.5):

    y_prob = _flatten_prediction_batches(pred_batches)

    test_idx = np.asarray(test_idx)
    y = np.asarray(y)
    smiles = np.asarray(smiles)

    y_actual = y[test_idx].astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    fold_pred_df = pd.DataFrame(
        {
            "smiles": smiles[test_idx],
            "fold": fold,
            "y_actual": y_actual,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    return fold_pred_df


# Plots
def plot_confusion_matrix_from_cv(
    predictions_df, fold=None, title="Confusion Matrix", cmap="Blues", save_path=None
):
    if fold:
        fold_df = predictions_df[predictions_df["fold"] == fold].copy()
    else:
        fold_df = predictions_df.copy()
    if fold_df.empty:
        raise ValueError(f"No predictions found for fold={fold}")

    cm = confusion_matrix(
        y_true=fold_df["y_actual"].astype(int),
        y_pred=fold_df["y_pred"].astype(int),
        labels=[0, 1],
    )

    fig, ax = plot_confusion_matrix(
        conf_mat=cm,
        class_names=["0", "1"],
        show_absolute=True,
        show_normed=True,
        cmap=cmap,
        colorbar=True,
        figsize=(5, 4),
    )
    ax.set_title(f"{title} - Combined All Fold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()


def plot_mean_roc(
    prediction_df,
    fold_col="fold",
    y_true_col="y_actual",
    y_score_col="y_prob",
    n_points=100,
    color="green",
    title="ROC with Std Dev (Across Folds)",
    save_path=None,
):
    # common FPR grid
    mean_fpr = np.linspace(0, 1, n_points)

    tprs = []
    aucs = []

    for fold, df_fold in prediction_df.groupby(fold_col):
        y_true = df_fold[y_true_col]
        y_scores = df_fold[y_score_col]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # interpolate TPR to common FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    tprs = np.array(tprs)

    # mean and std
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # bounds (±1 std)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # plot
    plt.figure(figsize=(6, 6))

    plt.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})",
    )

    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        color=color,
        alpha=0.2,
        label="±1 std",
    )

    plt.plot([0, 1], [0, 1], linestyle="--", color="red")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


# @dataclass
# class MPNNDataModule(L.LightningDataModule):
#     data_mols: list
#     split: tuple
#     cfg: dict

#     def __post_init__(self):
#         super().__init__()
#         self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

#     def setup(self, stage=None):
#         train_ids, val_ids, test_ids = self.split
#         self.train_dset = data.MoleculeDataset([self.data_mols[i] for i in train_ids], self.featurizer)
#         self.val_dset = data.MoleculeDataset([self.data_mols[i] for i in val_ids], self.featurizer)
#         self.test_dset = data.MoleculeDataset([self.data_mols[i] for i in test_ids], self.featurizer)

#     def train_dataloader(self):
#         return data.build_dataloader(self.train_dset, **self.cfg.data.loader.model_dump())

#     def val_dataloader(self):
#         return data.build_dataloader(self.val_dset, **self.cfg.val_data.loader.model_dump())

#     def test_dataloader(self):
#         return data.build_dataloader(self.test_dset, **self.cfg.val_data.loader.model_dump())


# def build_mpnn_model(batch_norm=False, metrics=None, **kwargs):
#     mp = nn.BondMessagePassing()
#     agg = nn.MeanAggregation()
#     ffn = nn.BinaryClassificationFFN(n_tasks=1)

#     return models.MPNN(mp, agg, ffn, batch_norm=batch_norm, metrics=metrics, **kwargs)
