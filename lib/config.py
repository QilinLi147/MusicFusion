"""Dataset paths and input shapes used by the public training entry point."""

from typing import Dict, Tuple


DATASET_FILES: Dict[str, Dict[str, str]] = {
    "memo": {
        "mel": "memo/mel_spec.npy",
        "coch": "memo/cochlegram.npy",
        "label_a": "memo/labels/label_a.npy",
        "label_v": "memo/labels/label_v.npy",
    },
    "pmemo": {
        "mel": "data_PMEmo/mel_spec.npy",
        "coch": "data_PMEmo/cochlegram.npy",
        "label_a": "data_PMEmo/label_a.npy",
        "label_v": "data_PMEmo/label_v.npy",
    },
    "songs1000": {
        "mel": "data_1000songs/mel_spec.npy",
        "coch": "data_1000songs/cochlegram.npy",
        "label_a": "data_1000songs/label_a.npy",
        "label_v": "data_1000songs/label_v.npy",
    },
    "1000songs": {
        "mel": "data_1000songs/mel_spec.npy",
        "coch": "data_1000songs/cochlegram.npy",
        "label_a": "data_1000songs/label_a.npy",
        "label_v": "data_1000songs/label_v.npy",
    },
    "deam": {
        "mel": "DEAM/processed/mel_spec.npy",
        "coch": "DEAM/processed/cochlegram.npy",
        "label_a": "DEAM/processed/labels/label_a.npy",
        "label_v": "DEAM/processed/labels/label_v.npy",
    },
}


def resolve_dataset_paths(data_root: str, dataset: str) -> Dict[str, str]:
    """Resolve the feature and label files for a supported dataset."""

    name = dataset.lower()
    if name not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset '{dataset}'; choose from {list(DATASET_FILES)}")
    mapping = DATASET_FILES[name]
    return {key: f"{data_root.rstrip('/')}/{value}" for key, value in mapping.items()}


def dataset_input_shapes(dataset: str) -> Tuple[int, int, int, int]:
    """Return ``(mel_nodes, coch_nodes, feature_dim, num_classes)``."""

    name = dataset.lower()
    if name in {"memo", "deam"}:
        return 128, 84, 87, 2
    if name in {"pmemo", "songs1000", "1000songs"}:
        return 128, 84, 44, 2
    raise ValueError(f"Unsupported dataset '{dataset}'; choose from {list(DATASET_FILES)}")
