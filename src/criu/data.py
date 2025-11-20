from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DataConfig


@dataclass
class DataBundle:
    train: DataLoader
    forget: DataLoader
    retain: DataLoader
    test: DataLoader
    forget_importance: DataLoader
    retain_importance: DataLoader


class CIFAR10DataModule:
    """
    Provides the train/forget/retain/test splits used throughout the project.
    """

    def __init__(self, cfg: DataConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA device not available; please enable a GPU runtime.")
        torch.backends.cudnn.benchmark = True

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Returns (train_tf, eval_tf) tailored to the dataset. For MNIST we
        upsample to 32x32 and replicate channel to 3 so the ResNet front-end
        stays unchanged.
        """
        if self.cfg.dataset.lower() == "mnist":
            mean = (0.1307, 0.1307, 0.1307)
            std = (0.3081, 0.3081, 0.3081)
            base = [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
            ]
            train_tf = transforms.Compose(base + [transforms.ToTensor(), transforms.Normalize(mean, std)])
            eval_tf = transforms.Compose(base + [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:  # default CIFAR-10
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
            train_tf = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            eval_tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return train_tf, eval_tf

    def _subset_indices(self, labels: List[int], target_classes: Iterable[int], fraction: float) -> List[int]:
        pool = [idx for idx, label in enumerate(labels) if label in target_classes]
        if not pool:
            raise ValueError("Requested classes not present in CIFAR10 training set.")
        rng = random.Random(self.seed)
        count = max(1, int(len(pool) * fraction))
        rng.shuffle(pool)
        return pool[:count]

    def setup(self) -> DataBundle:
        train_tf, eval_tf = self._build_transforms()
        ds_name = self.cfg.dataset.lower()
        if ds_name == "mnist":
            train_dataset = datasets.MNIST(
                root=self.cfg.data_dir,
                train=True,
                download=True,
                transform=train_tf,
            )
            base_eval_dataset = datasets.MNIST(
                root=self.cfg.data_dir,
                train=True,
                download=False,
                transform=eval_tf,
            )
            test_dataset = datasets.MNIST(
                root=self.cfg.data_dir,
                train=False,
                download=True,
                transform=eval_tf,
            )
        else:
            train_dataset = datasets.CIFAR10(
                root=self.cfg.data_dir,
                train=True,
                download=True,
                transform=train_tf,
            )
            base_eval_dataset = datasets.CIFAR10(
                root=self.cfg.data_dir,
                train=True,
                download=False,
                transform=eval_tf,
            )
            test_dataset = datasets.CIFAR10(
                root=self.cfg.data_dir,
                train=False,
                download=True,
                transform=eval_tf,
            )

        forget_indices = self._subset_indices(
            base_eval_dataset.targets,
            self.cfg.forget_classes,
            self.cfg.forget_split,
        )
        retain_indices = self._subset_indices(
            base_eval_dataset.targets,
            [c for c in self.cfg.retain_classes if c not in self.cfg.forget_classes],
            self.cfg.retain_split,
        )

        pin_memory = self.device.type == "cuda"
        common_kwargs: Dict[str, int | bool] = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": pin_memory,
        }

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs,
        )

        forget_subset = Subset(base_eval_dataset, forget_indices)
        retain_subset = Subset(base_eval_dataset, retain_indices)

        forget_loader = DataLoader(
            forget_subset,
            shuffle=True,
            **common_kwargs,
        )
        retain_loader = DataLoader(
            retain_subset,
            shuffle=True,
            **common_kwargs,
        )
        forget_importance_loader = DataLoader(
            forget_subset,
            shuffle=False,
            **common_kwargs,
        )
        retain_importance_loader = DataLoader(
            retain_subset,
            shuffle=False,
            **common_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **common_kwargs,
        )

        return DataBundle(
            train=train_loader,
            forget=forget_loader,
            retain=retain_loader,
            test=test_loader,
            forget_importance=forget_importance_loader,
            retain_importance=retain_importance_loader,
        )
