from __future__ import annotations

import copy
import json
from itertools import cycle
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, UnlearningConfig
from .model import CRIResNet


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(1, total)


def train_base(model: CRIResNet, train_loader: DataLoader, test_loader: DataLoader, cfg: TrainConfig, device: torch.device) -> CRIResNet:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        print(f"[train] epoch={epoch+1} test_acc={acc:.2f}%")

    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"Saved base checkpoint to {cfg.checkpoint_path}")
    return model


def _freeze_base_weights(model: CRIResNet) -> None:
    for name, param in model.named_parameters():
        if "gate_logits" in name or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def _uniform_targets(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.full((num_classes,), 1.0 / num_classes, device=device)


def run_unlearning(
    model: CRIResNet,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    cfg: UnlearningConfig,
    device: torch.device,
    train_all: bool = False,
) -> dict[str, List[float]]:
    base_model = copy.deepcopy(model).to(device)
    base_model.eval()

    model = model.to(device)
    if train_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        _freeze_base_weights(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    metrics = {"forget_acc": [], "retain_acc": [], "test_acc": []}
    uniform = _uniform_targets(model.fc.out_features, device)

    forget_iter = cycle(forget_loader)
    retain_iter = cycle(retain_loader)
    steps_per_epoch = max(len(forget_loader), len(retain_loader))

    for epoch in range(cfg.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            forget_images, forget_labels = next(forget_iter)
            retain_images, retain_labels = next(retain_iter)

            forget_images = forget_images.to(device, non_blocking=True)
            retain_images = retain_images.to(device, non_blocking=True)
            forget_labels = forget_labels.to(device, non_blocking=True)
            retain_labels = retain_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            forget_logits = model(forget_images)
            retain_logits = model(retain_images)

            uniform_targets = uniform.unsqueeze(0).expand(forget_logits.size(0), -1)
            uniform_loss = F.kl_div(F.log_softmax(forget_logits, dim=1), uniform_targets, reduction="batchmean")
            retain_ce = F.cross_entropy(retain_logits, retain_labels)

            with torch.no_grad():
                base_logits = base_model(retain_images)
            kl_anchor = F.kl_div(
                F.log_softmax(retain_logits, dim=1),
                F.softmax(base_logits, dim=1),
                reduction="batchmean",
            )

            protect_loss = model.protection_regularizer()

            loss = (
                cfg.uniform_forget_weight * uniform_loss
                + retain_ce
                + cfg.kl_weight * kl_anchor
                + cfg.protection_weight * protect_loss
            )
            loss.backward()
            optimizer.step()

        forget_acc = evaluate(model, forget_loader, device)
        retain_acc = evaluate(model, retain_loader, device)
        test_acc = evaluate(model, test_loader, device)
        metrics["forget_acc"].append(forget_acc)
        metrics["retain_acc"].append(retain_acc)
        metrics["test_acc"].append(test_acc)
        print(
            f"[unlearn] epoch={epoch+1} forget_acc={forget_acc:.2f}% "
            f"retain_acc={retain_acc:.2f}% test_acc={test_acc:.2f}%"
        )

    return metrics


def save_selection_artifacts(
    log_dir: Path,
    selection_masks: dict[str, torch.Tensor],
    protection_masks: dict[str, torch.Tensor],
    audit_log: Iterable[dict],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"selection": selection_masks, "protection": protection_masks}, log_dir / "masks.pt")
    with (log_dir / "audit_log.json").open("w", encoding="utf-8") as fp:
        json.dump(list(audit_log), fp, indent=2)
