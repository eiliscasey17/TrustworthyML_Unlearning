from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List

import torch

from .config import ImportanceConfig


@dataclass
class ChannelRecord:
    block: str
    channel: int
    forget_score: float
    retain_score: float
    combined: float


@dataclass
class SelectionResult:
    selection_masks: Dict[str, torch.Tensor]
    protection_masks: Dict[str, torch.Tensor]
    audit_log: List[ChannelRecord]


def _collect_records(
    forget_scores: Dict[str, torch.Tensor],
    retain_scores: Dict[str, torch.Tensor] | None,
    retain_weight: float,
) -> List[ChannelRecord]:
    records: List[ChannelRecord] = []
    for block, scores in forget_scores.items():
        retain_tensor = retain_scores[block] if retain_scores is not None else None
        for idx in range(scores.numel()):
            f_score = float(scores[idx].item())
            r_score = float(retain_tensor[idx].item()) if retain_tensor is not None else 0.0
            combined = f_score - retain_weight * r_score
            records.append(ChannelRecord(block=block, channel=idx, forget_score=f_score, retain_score=r_score, combined=combined))
    return records


def build_masks(
    forget_scores: Dict[str, torch.Tensor],
    retain_scores: Dict[str, torch.Tensor],
    cfg: ImportanceConfig,
) -> SelectionResult:
    records = _collect_records(forget_scores, retain_scores, cfg.retain_weight)

    total_channels = len(records)
    if total_channels == 0:
        raise ValueError("No channels found for selection.")

    select_budget = max(1, int(total_channels * cfg.selection_ratio))
    protection_budget = max(1, int(total_channels * cfg.protection_ratio))

    selection_masks = {block: torch.zeros_like(scores) for block, scores in forget_scores.items()}
    protection_masks = {block: torch.zeros_like(scores) for block, scores in forget_scores.items()}

    records.sort(key=lambda r: r.combined, reverse=True)
    for rec in records[:select_budget]:
        selection_masks[rec.block][rec.channel] = 1.0

    retain_sorted = sorted(records, key=lambda r: r.retain_score, reverse=True)
    for rec in retain_sorted[:protection_budget]:
        protection_masks[rec.block][rec.channel] = 1.0

    return SelectionResult(selection_masks=selection_masks, protection_masks=protection_masks, audit_log=records)


def build_forget_only_masks(
    forget_scores: Dict[str, torch.Tensor],
    selection_ratio: float,
) -> SelectionResult:
    records = _collect_records(forget_scores, None, 0.0)
    total_channels = len(records)
    if total_channels == 0:
        raise ValueError("No channels found for selection.")
    select_budget = max(1, int(total_channels * selection_ratio))

    selection_masks = {block: torch.zeros_like(scores) for block, scores in forget_scores.items()}
    for rec in sorted(records, key=lambda r: r.forget_score, reverse=True)[:select_budget]:
        selection_masks[rec.block][rec.channel] = 1.0

    protection_masks = {block: torch.zeros_like(scores) for block, scores in forget_scores.items()}
    return SelectionResult(selection_masks=selection_masks, protection_masks=protection_masks, audit_log=records)


def build_random_masks(
    forget_scores: Dict[str, torch.Tensor],
    selection_ratio: float,
    seed: int,
    retain_scores: Dict[str, torch.Tensor] | None = None,
) -> SelectionResult:
    records = _collect_records(forget_scores, retain_scores, retain_weight=0.0)
    total_channels = len(records)
    if total_channels == 0:
        raise ValueError("No channels found for selection.")
    select_budget = max(1, int(total_channels * selection_ratio))
    rng = random.Random(seed)
    rng.shuffle(records)
    selected = records[:select_budget]

    selection_masks = {block: torch.zeros_like(scores) for block, scores in forget_scores.items()}
    for rec in selected:
        selection_masks[rec.block][rec.channel] = 1.0

    protection_masks = {block: torch.zeros_like(scores) for block, scores in forget_scores.items()}
    return SelectionResult(selection_masks=selection_masks, protection_masks=protection_masks, audit_log=records)
