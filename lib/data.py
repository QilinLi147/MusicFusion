import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, SequentialSampler

from .config import resolve_dataset_paths, dataset_input_shapes


class MusicEmotionDataset(Dataset):
    """Music emotion dataset for dual-view Mel and cochleagram features.

    Each item is returned as a dictionary:

    {
        "mel": Tensor[num_mel_nodes, feat_dim],
        "coch": Tensor[num_coch_nodes, feat_dim],
        "label": Tensor[],
        "index": int
    }
    """

    def __init__(
        self,
        dataset: str,
        label_mode: str = "a",
        data_root: str = "/data/qilin.li/dataset",
        cache_in_memory: bool = True,
    ) -> None:
        super().__init__()
        if label_mode not in {"a", "v"}:
            raise ValueError("label_mode must be either 'a' (Arousal) or 'v' (Valence).")

        paths = resolve_dataset_paths(data_root, dataset)
        mel_file = paths["mel"]
        coch_file = paths["coch"]
        label_file = paths[f"label_{label_mode.lower()}"]

        if not (os.path.exists(mel_file) and os.path.exists(coch_file) and os.path.exists(label_file)):
            missing = [p for p in [mel_file, coch_file, label_file] if not os.path.exists(p)]
            raise FileNotFoundError(f"The following files are missing: {missing}")

        mel_np = np.load(mel_file, mmap_mode=None if cache_in_memory else "r")
        coch_np = np.load(coch_file, mmap_mode=None if cache_in_memory else "r")
        labels_np = np.load(label_file)

        if mel_np.shape[0] != coch_np.shape[0] or mel_np.shape[0] != labels_np.shape[0]:
            raise ValueError(
                f"Mismatch between feature and label sample counts: "
                f"mel={mel_np.shape}, coch={coch_np.shape}, labels={labels_np.shape}"
            )

        labels_np = labels_np.astype(np.float32)
        threshold = 0.5 if dataset.lower() == "pmemo" else 0.0
        labels_np = np.where(labels_np > threshold, 1, 0).astype(np.int64)

        self.dataset = dataset
        self.label_mode = label_mode
        self.mel = torch.from_numpy(mel_np).float()
        self.coch = torch.from_numpy(coch_np).float()
        self.labels = torch.from_numpy(labels_np).long()

        self.mel_nodes, self.coch_nodes, feat_dim, _ = dataset_input_shapes(dataset)
        self.feat_dim = feat_dim

        self.mel = self.mel.view(-1, self.mel_nodes, feat_dim)
        self.coch = self.coch.view(-1, self.coch_nodes, feat_dim)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mel = self.mel[idx]
        coch = self.coch[idx]
        label = self.labels[idx]
        return {
            "mel": mel,
            "coch": coch,
            "label": label,
            "index": torch.tensor(idx, dtype=torch.long),
        }


def stratified_split(
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    labels_np = labels.cpu().numpy()
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels_np):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n_train = int(len(cls_idx) * train_ratio)
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_dataloaders(
    dataset: MusicEmotionDataset,
    batch_size: int = 256,
    train_ratio: float = 0.7,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_idx, val_idx = stratified_split(dataset.labels, train_ratio=train_ratio, seed=seed)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=RandomSampler(train_subset, replacement=False),
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        sampler=SequentialSampler(val_subset),
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def get_dataset_stats(dataset: MusicEmotionDataset) -> Dict[str, float]:
    labels = dataset.labels.cpu().numpy()
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    total = float(labels.shape[0])
    return {
        "total": total,
        "positives": pos,
        "negatives": neg,
        "pos_ratio": pos / total if total > 0 else 0.0,
        "neg_ratio": neg / total if total > 0 else 0.0,
    }
