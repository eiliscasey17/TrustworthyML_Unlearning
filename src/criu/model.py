from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CRIBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, adapter_rank: int = 16) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.adapter = nn.Sequential(
            conv1x1(planes, adapter_rank),
            nn.ReLU(inplace=True),
            conv1x1(adapter_rank, planes),
        )
        nn.init.zeros_(self.adapter[-1].weight)

        self.gate_logits = nn.Parameter(torch.zeros(planes))
        self.register_buffer("selection_mask", torch.zeros(planes))
        self.register_buffer("protection_mask", torch.zeros(planes))

    @torch.no_grad()
    def set_masks(self, selection: torch.Tensor, protection: torch.Tensor) -> None:
        if selection.shape != self.selection_mask.shape:
            raise ValueError("Selection mask shape mismatch.")
        if protection.shape != self.protection_mask.shape:
            raise ValueError("Protection mask shape mismatch.")
        self.selection_mask.copy_(selection)
        self.protection_mask.copy_(protection)

    def core_region_mask(self) -> torch.Tensor:
        return self.selection_mask * (1.0 - self.protection_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        gate_mask = self.core_region_mask().view(1, -1, 1, 1)
        gates = torch.sigmoid(self.gate_logits).view(1, -1, 1, 1)
        gated = out * (1.0 - gate_mask + gate_mask * gates)
        adapter_delta = self.adapter(out) * gate_mask

        out = gated + adapter_delta + identity
        out = self.relu(out)
        return out

    def protection_regularizer(self) -> torch.Tensor:
        if not torch.any(self.protection_mask):
            return torch.tensor(0.0, device=self.gate_logits.device)
        protect = self.protection_mask.view(1, -1, 1, 1)
        gate_pen = torch.sum((torch.sigmoid(self.gate_logits) - 1.0) ** 2 * self.protection_mask)
        adapter_pen = 0.0
        for layer in self.adapter:
            if isinstance(layer, nn.Conv2d):
                adapter_pen = adapter_pen + torch.sum(layer.weight ** 2)
        adapter_pen = adapter_pen * torch.sum(self.protection_mask)
        return gate_pen + adapter_pen


class CRIResNet(nn.Module):
    def __init__(self, block: type[CRIBlock], layers: List[int], num_classes: int = 10, adapter_rank: int = 16) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], adapter_rank=adapter_rank)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, adapter_rank=adapter_rank)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, adapter_rank=adapter_rank)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, adapter_rank=adapter_rank)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type[CRIBlock], planes: int, blocks: int, stride: int = 1, adapter_rank: int = 16) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, adapter_rank=adapter_rank))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, adapter_rank=adapter_rank))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def cri_blocks(self) -> Iterator[Tuple[str, CRIBlock]]:
        for name, module in self.named_modules():
            if isinstance(module, CRIBlock):
                yield name, module

    def reset_core_masks(self) -> None:
        for _, block in self.cri_blocks():
            block.set_masks(torch.zeros_like(block.selection_mask), torch.zeros_like(block.protection_mask))

    def apply_masks(self, selection: Dict[str, torch.Tensor], protection: Dict[str, torch.Tensor]) -> None:
        for name, block in self.cri_blocks():
            sel = selection.get(name)
            prot = protection.get(name)
            if sel is None or prot is None:
                continue
            block.set_masks(sel.to(next(block.parameters()).device), prot.to(next(block.parameters()).device))

    def protection_regularizer(self) -> torch.Tensor:
        regs = []
        for _, block in self.cri_blocks():
            regs.append(block.protection_regularizer())
        if not regs:
            return torch.tensor(0.0, device=self.fc.weight.device)
        return torch.stack(regs).sum()


def cri_resnet18(num_classes: int = 10, adapter_rank: int = 16) -> CRIResNet:
    return CRIResNet(CRIBlock, [2, 2, 2, 2], num_classes=num_classes, adapter_rank=adapter_rank)

