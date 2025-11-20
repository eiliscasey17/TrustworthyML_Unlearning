from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    data_dir: Path = Path("./data")
    dataset: str = "cifar10"  # or "mnist"
    batch_size: int = 128
    num_workers: int = 4
    train_split: float = 0.5
    importance_split: float = 0.2
    retain_split: float = 0.2
    forget_split: float = 0.1
    forget_classes: List[int] = field(default_factory=lambda: [0, 1])
    retain_classes: List[int] = field(default_factory=lambda: list(range(10)))


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    checkpoint_path: Path = Path("cri_resnet18_base.pt")


@dataclass
class ImportanceConfig:
    retain_weight: float = 1.0
    selection_ratio: float = 0.1
    protection_ratio: float = 0.05


@dataclass
class UnlearningConfig:
    epochs: int = 20
    lr: float = 1e-3
    kl_weight: float = 0.5
    protection_weight: float = 0.1
    uniform_forget_weight: float = 1.0
    log_dir: Path = Path("artifacts")


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
