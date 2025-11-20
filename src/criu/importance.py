from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import CRIResNet


def _blank_importance(model: CRIResNet) -> Dict[str, torch.Tensor]:
    imp: Dict[str, torch.Tensor] = {}
    for name, module in model.cri_blocks():
        imp[name] = torch.zeros(module.bn2.num_features, device=module.bn2.weight.device)
    return imp


def compute_channel_importance(model: CRIResNet, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.train(False)
    importance = _blank_importance(model)
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        total_samples += images.size(0)

        for block_name, block in model.cri_blocks():
            grad = block.bn2.weight.grad
            if grad is None:
                continue
            importance[block_name] += grad.detach().abs()

        model.zero_grad(set_to_none=True)

    for key in importance:
        importance[key] = (importance[key] / max(1, total_samples)).detach().cpu()
    model.train(True)
    return importance

