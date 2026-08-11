"""Public MusicFusion library entry point."""

from .config import dataset_input_shapes, resolve_dataset_paths
from .data import MusicEmotionDataset, build_dataloaders, get_dataset_stats, stratified_split
from .models import MusicFusionModel
from .redesign_modules import ProtoAlign, ReliaPseudo, TriDistill
from .train_utils import set_seed

__all__ = [
    "dataset_input_shapes",
    "resolve_dataset_paths",
    "MusicEmotionDataset",
    "build_dataloaders",
    "get_dataset_stats",
    "stratified_split",
    "MusicFusionModel",
    "ProtoAlign",
    "ReliaPseudo",
    "TriDistill",
    "set_seed",
]
