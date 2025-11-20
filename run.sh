#!/usr/bin/env bash
set -euo pipefail

# Ensure the desired environment is active before invoking this script
# (e.g., `conda activate mod` which already has torch/torchvision with CUDA support).

DATASETS=("cifar10" "mnist")
BASELINES=("cri" "forget_only" "random_subset" "full_model")

for dataset in "${DATASETS[@]}"; do
  for baseline in "${BASELINES[@]}"; do
    TAG="${baseline}"
    if [[ "${dataset}" != "cifar10" ]]; then
      TAG="${dataset}_${baseline}"
    fi
    echo "=== Running baseline: ${baseline} on ${dataset} ==="
    python -m src.criu.pipeline --dataset "${dataset}" --baseline "${baseline}" --tag "${TAG}"
    echo "=== Generating visualizations for: ${baseline} on ${dataset} ==="
    if [[ "${dataset}" == "cifar10" ]]; then
      ART_DIR="artifacts/${baseline}"
    else
      ART_DIR="artifacts/${dataset}/${TAG}"
    fi
    python -m src.criu.visualize --artifacts "${ART_DIR}"
  done
done

echo "All baselines completed. Artifacts saved under artifacts/<dataset>/<tag>/ (or artifacts/<baseline>/ for CIFAR-10)."
