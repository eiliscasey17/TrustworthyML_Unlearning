from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch


def _load_metrics(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _plot_metrics(metrics: Dict[str, List[float]], out_path: Path) -> None:
    epochs = range(1, len(next(iter(metrics.values()))) + 1)
    plt.figure(figsize=(8, 4))
    for key, values in metrics.items():
        plt.plot(epochs, values, marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Core Region Unlearning: Forget vs Retain vs Test Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_mask_allocation(masks_path: Path, out_path: Path) -> None:
    if not masks_path.exists():
        print(f"Mask file missing ({masks_path}); skipping mask allocation plot.")
        return
    data = torch.load(masks_path, map_location="cpu")
    selection = data["selection"]
    protection = data["protection"]

    blocks = sorted(selection.keys())
    sel_counts = [int(selection[name].sum().item()) for name in blocks]
    prot_counts = [int(protection[name].sum().item()) for name in blocks]

    x = range(len(blocks))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar([i - width / 2 for i in x], sel_counts, width=width, label="Edited channels")
    plt.bar([i + width / 2 for i in x], prot_counts, width=width, label="Protected channels")
    plt.xticks(x, blocks, rotation=45, ha="right")
    plt.ylabel("Channels")
    plt.title("Channel Allocation Across Residual Blocks")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_importance_scatter(audit_path: Path, out_path: Path) -> None:
    if not audit_path.exists():
        print(f"Audit log missing ({audit_path}); skipping importance scatter plot.")
        return
    with audit_path.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    forget = [rec["forget_score"] for rec in records]
    retain = [rec["retain_score"] for rec in records]
    combined = [rec["combined"] for rec in records]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(forget, retain, c=combined, cmap="coolwarm", s=10, alpha=0.7)
    plt.xlabel("Forget-side importance")
    plt.ylabel("Retain-side importance")
    plt.title("Two-sided Channel Importance Distribution")
    plt.colorbar(scatter, label="Combined score (forget - retain*weight)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_visualization(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(artifacts_dir / "metrics.json")
    _plot_metrics(metrics, artifacts_dir / "metrics_curve.png")
    _plot_mask_allocation(artifacts_dir / "masks.pt", artifacts_dir / "mask_allocation.png")
    _plot_importance_scatter(artifacts_dir / "audit_log.json", artifacts_dir / "importance_scatter.png")
    print(f"Saved plots to {artifacts_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRI unlearning visualizations.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/cri"),
        help="Artifacts directory with masks/logs/metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_visualization(args.artifacts)


if __name__ == "__main__":
    main()
