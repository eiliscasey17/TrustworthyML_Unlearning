from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from .config import ProjectConfig
from .data import CIFAR10DataModule
from .importance import compute_channel_importance
from .model import cri_resnet18
from .selector import build_masks, build_forget_only_masks, build_random_masks
from .unlearning import run_unlearning, save_selection_artifacts, train_base


def run_full_pipeline(cfg: ProjectConfig, baseline: str, tag: str | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available; please enable a GPU runtime.")

    data_module = CIFAR10DataModule(cfg.data)
    bundle = data_module.setup()
    model = cri_resnet18().to(device)

    if cfg.train.checkpoint_path.exists():
        model.load_state_dict(torch.load(cfg.train.checkpoint_path, map_location=device))
        print(f"Loaded existing checkpoint from {cfg.train.checkpoint_path}")
    else:
        train_base(model, bundle.train, bundle.test, cfg.train, device)
        model = model.to(device)

    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    forget_scores = compute_channel_importance(model, bundle.forget_importance, device)
    retain_scores = compute_channel_importance(model, bundle.retain_importance, device)

    selection = None
    train_all = False
    if baseline == "cri":
        selection = build_masks(forget_scores, retain_scores, cfg.importance)
    elif baseline == "forget_only":
        selection = build_forget_only_masks(forget_scores, cfg.importance.selection_ratio)
    elif baseline == "random_subset":
        selection = build_random_masks(
            forget_scores,
            cfg.importance.selection_ratio,
            seed=data_module.seed,
            retain_scores=retain_scores,
        )
    elif baseline == "full_model":
        train_all = True
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    model_for_unlearning = cri_resnet18()
    model_for_unlearning.load_state_dict(base_state, strict=False)

    # Keep CIFAR-10 artifacts backward compatible; namespace others by dataset.
    base_artifacts = cfg.unlearning.log_dir if cfg.data.dataset.lower() == "cifar10" else cfg.unlearning.log_dir / cfg.data.dataset.lower()
    artifacts_dir = base_artifacts if not tag else base_artifacts / tag
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if selection is not None:
        model_for_unlearning.apply_masks(selection.selection_masks, selection.protection_masks)
        save_selection_artifacts(
            artifacts_dir,
            selection.selection_masks,
            selection.protection_masks,
            [
                {
                    "block": rec.block,
                    "channel": rec.channel,
                    "forget_score": rec.forget_score,
                    "retain_score": rec.retain_score,
                    "combined": rec.combined,
                }
                for rec in selection.audit_log
            ],
        )

    metrics = run_unlearning(
        model_for_unlearning,
        bundle.forget,
        bundle.retain,
        bundle.test,
        cfg.unlearning,
        device,
        train_all=train_all,
    )
    print(f"Final metrics: {metrics}")

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core Region Isolation Unlearning pipeline.")
    parser.add_argument(
        "--baseline",
        choices=["cri", "forget_only", "random_subset", "full_model"],
        default="cri",
        help="Unlearning baseline to run.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional artifact subdirectory name (defaults to baseline name).",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "mnist"],
        default="cifar10",
        help="Dataset to use for unlearning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    cfg.data.dataset = args.dataset
    # Keep separate checkpoints per dataset to avoid reuse across domains.
    if cfg.data.dataset != "cifar10":
        cfg.train.checkpoint_path = cfg.train.checkpoint_path.with_name(f"{cfg.train.checkpoint_path.stem}_{cfg.data.dataset}{cfg.train.checkpoint_path.suffix}")
    tag = args.tag if args.tag is not None else args.baseline
    run_full_pipeline(cfg, args.baseline, tag)


if __name__ == "__main__":
    main()
