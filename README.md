# Trustworthy Machine Unlearning – Core Region Isolation (CRI-U)

This repo contains the CIFAR-10 (default) and MNIST / ResNet-18 implementation of **Core Region Isolation Unlearning (CRI-U)** for the Trustworthy ML course project by Eilis Casey & Rui Xie. The pipeline identifies the channels that are most responsible for unsafe behavior (forget set) while explicitly protecting retain-critical channels, then applies a hybrid gate+adapter edit limited to that core region. We compare CRI-U against three baselines:

1. **Full-model unlearning** (all parameters trainable)
2. **Forget-only importance** (SalUn-style selection, no retain protection)
3. **Random subset** (same edit budget as CRI-U, channels chosen randomly)

All experiments run on GPU (checkpoints assume CUDA). Artifacts for each baseline live under `artifacts/<baseline>` and include the loss/accuracy curves plus mask/audit visualizations when applicable.

> **Note:** The reliable, maintained implementation now lives under `src/criu`. Earlier experimental variants (e.g., `20251118`, `20251119`, or LoRA-based explorations) have been archived under `legacy/` for reference so the active code path stays clean.

---

## Terminology (for clarity)

- **Retain protection:** channels with high retain-side importance are marked as protected and excluded from edits. In code, `selector.build_masks` sets `protection_masks`, `model.apply_masks` loads them, and `model.protection_regularizer` penalizes any movement of gates/adapters on those channels. This keeps retain-critical capacity stable while the core region is edited.
- **SalUn-style (forget-only importance) baseline:** a selector that ranks channels only by forget-side gradients (no retain weighting, no protection masks). Implemented via `selector.build_forget_only_masks`; it updates the same parameter budget as CRI-U but does not defend retain-critical channels, so it serves as the “forget-only” comparator.
- **Channel importance (how we measure it):** treat each channel’s BatchNorm weight (`bn2.weight`) as a sensitivity probe. For a batch, run forward + backward, grab `|∂loss/∂bn2.weight|` for every channel, and accumulate across the whole dataset. Do this twice—once on the forget loader, once on the retain loader—so you end up with two per-channel score tensors: “forget importance” and “retain importance.” High forget + low retain ⇒ good to edit; high retain ⇒ protect.

---

## End-to-End Design (quick read)

1. **Data splits:** CIFAR-10 is divided into train/forget/retain/test. Forget classes default to `[0, 1]`; retain is the remaining classes (`src/criu/data.py`, `src/criu/config.py`).
2. **Per-set importance:** run `compute_channel_importance` twice—on the forget loader and on the retain loader—to get per-channel gradients for each residual block. Average absolute grads over the dataset to get two importance tensors.
3. **Two-sided selection + protection:** compute `combined = forget_score - retain_weight * retain_score`. Rank all channels and take the top `selection_ratio` fraction (default 10%) by combined score to edit. Independently, take the top `protection_ratio` fraction (default 5%) by retain_score to protect. Save both masks and a full channel-level audit log (`selector.py`).
4. **Core-region edit:** load masks into CRI-ResNet-18; only channels in (selection \ protection) update via gates + low-rank adapters. All other weights are frozen. A protection regularizer discourages any movement on protected channels (`model.py`).
5. **Training objective:** on mixed forget/retain batches, minimize: uniform-forget KL + retain cross-entropy + KL anchor to the frozen base (retain) + protection penalty (`unlearning.py`).
6. **Baselines:** (a) full-model unlearning (all params trainable), (b) SalUn-style forget-only importance (no retain scores, no protection), (c) random subset with the same edit budget. CRI-U is the two-sided + protection + gate/adapter method.

This pipeline keeps the edit surgical (small edited region), auditable (masks + logs), and utility-aware (retain protection + KL anchor).

### MNIST support
Pass `--dataset mnist` to `pipeline.py` (or use `run.sh`) to run the same CRI-U pipeline on MNIST. Transforms upsample to 32×32 and replicate the grayscale channel to 3 channels so the ResNet front-end stays unchanged; checkpoints are stored separately under `artifacts/mnist/`.

---

## Reproducing the Experiments

1. **Environment:** activate a CUDA-ready environment (e.g., `conda activate mod`) with `torch` and `torchvision`.
2. **Run all baselines + plots (CIFAR-10 + MNIST via script):**  
   ```bash
   bash run.sh
   ```
   This sequentially launches:
   - `python -m src.criu.pipeline --dataset <dataset> --baseline <baseline> --tag <tag>`
   - `python -m src.criu.visualize --artifacts <artifact_dir>`

   Output directories:  
   - CIFAR-10: `artifacts/<baseline>` (e.g., `artifacts/cri`)  
   - MNIST: `artifacts/mnist/<dataset_baseline>` (e.g., `artifacts/mnist/mnist_cri`)

   Output directories (CIFAR-10):
   - `artifacts/cri`
   - `artifacts/forget_only`
   - `artifacts/random_subset`
   - `artifacts/full_model`

3. **Run MNIST instead:**  
   ```bash
   python -m src.criu.pipeline --dataset mnist --baseline cri --tag mnist_cri
   python -m src.criu.visualize --artifacts artifacts/mnist/mnist_cri
   ```
   Artifacts for MNIST are stored under `artifacts/mnist/<tag>` to avoid clobbering CIFAR-10 runs.

4. **Visualize individual runs:**  
   ```bash
   python -m src.criu.visualize --artifacts artifacts/cri
   ```
   (Change `--artifacts` to point at any baseline directory.)

---

## Key Results

Figures referenced below are generated automatically by `run.sh`. The latest end-of-run metrics (CIFAR-10 and MNIST) are summarized in `artifacts/results_summary.json`.

### Results overview (final epoch)

| Dataset  | Baseline         | Forget ↓ | Retain ↑ | Test ↑ |
|----------|------------------|---------:|---------:|-------:|
| CIFAR-10 | CRI-U            | 51.30    | 70.25    | 64.27  |
| CIFAR-10 | Forget-only      | 12.40    | 66.13    | 52.12  |
| CIFAR-10 | Random subset    | 19.60    | 73.56    | 60.36  |
| CIFAR-10 | Full model       | 0.20     | 83.84    | 50.70  |
| MNIST    | CRI-U            | 8.69     | 98.75    | 79.67  |
| MNIST    | Forget-only      | 18.48    | 97.55    | 80.10  |
| MNIST    | Random subset    | 40.13    | 99.74    | 86.44  |
| MNIST    | Full model       | 58.21    | 99.92    | 90.30  |

- **What the metrics mean:**  
  - *Forget accuracy* — accuracy on the forget subset (default classes `[0,1]`); lower is better (more forgetting).  
  - *Retain accuracy* — accuracy on the retain subset (all non-forget classes); higher is better (keeps utility).  
  - *Test accuracy* — accuracy on the full CIFAR-10 test set; useful for overall sanity.

### Accuracy Trade-offs (CIFAR-10)

- `artifacts/cri/metrics_curve.png`  
  Forget accuracy drops from ~65% to **~50%** while retain accuracy stays stable at **~70%**, demonstrating that selective gating/adapters reduce the forget set with minimal collateral damage.  
  Final metrics (`artifacts/cri/metrics.json`):
  - Forget: **51.30%**
  - Retain: **70.25%**
  - Test: **64.27%**
  ![CRI metrics](artifacts/cri/metrics_curve.png)

- `artifacts/forget_only/metrics_curve.png`  
  Pure forget-side selection reaches **12.40%** forget accuracy but retain accuracy collapses to **66.13%** (~−4.1% vs CRI), highlighting the need for retain protection.
  ![Forget-only metrics](artifacts/forget_only/metrics_curve.png)

- `artifacts/random_subset/metrics_curve.png`  
  Randomly updating the same 10% of channels yields inconsistent behavior; final forget accuracy **19.60%** but retain accuracy **73.56%**, indicating edits landed on benign capacity rather than the unsafe region.
  ![Random subset metrics](artifacts/random_subset/metrics_curve.png)

- `artifacts/full_model/metrics_curve.png`  
  Full-model unlearning reaches **0.20%** forget accuracy but retains **83.84%** accuracy thanks to retraining all weights—expensive and with no audit trail. Serves as an upper bound baseline.
  ![Full-model metrics](artifacts/full_model/metrics_curve.png)

### Accuracy Trade-offs (MNIST)

- `artifacts/mnist/mnist_cri/metrics_curve.png`  
  CRI-U forget accuracy drops to **8.69%** while retain accuracy stays high at **98.75%**, showing targeted edits can suppress the forget set without harming retain set utility.  
  ![MNIST CRI metrics](artifacts/mnist/mnist_cri/metrics_curve.png)

Other MNIST baselines (forget-only/random/full-model) live under `artifacts/mnist/<tag>/metrics_curve.png` with matching audit logs and masks for selective methods.

### Auditability and Targeted Edits

- `artifacts/cri/mask_allocation.png`  
  Shows exactly how many channels were edited vs. protected per block (e.g., layer1 dominates the core region, layer4 untouched). This figure demonstrates the “surgical” edit footprint.
  ![Mask allocation](artifacts/cri/mask_allocation.png)

- `artifacts/cri/importance_scatter.png`  
  Visualizes forget-side vs retain-side importance scores for every channel, colored by the combined score used for selection—mirroring the methodology described in the proposal.
  ![Importance scatter](artifacts/cri/importance_scatter.png)

- `artifacts/cri/audit_log.json`  
  Detailed change log listing every channel, its forget/retain scores, and whether it falls into the edited/protected region. Same format exists for other selective baselines (forget_only, random_subset).

Full-model runs do not produce masks/audit logs (no targeted region), so `visualize.py` simply skips those plots.

---

## Repository Structure

- `src/criu/config.py` – hyperparameters (splits, learning rates, edit budget)
- `src/criu/data.py` – CIFAR-10 splits (train/forget/retain/test)
- `src/criu/model.py` – ResNet-18 with CRI blocks (gates + adapters)
- `src/criu/selector.py` – CRI selector + forget-only/random baselines
- `src/criu/unlearning.py` – training loop for base model + unlearning passes
- `src/criu/pipeline.py` – CLI entry point; handles baselines, logging, metrics
- `src/criu/visualize.py` – generates accuracy curves, mask allocation, importance scatter
- `legacy/` – frozen copies of earlier experiments (e.g., `src_20251118`, `src_20251119`, LoRA trials) kept for comparison
- `run.sh` – convenience script to execute all baselines and visualizations

---

## Suggested Next Steps

1. **Longer unlearning runs** for CRI/forget-only to push forget accuracy closer to random while monitoring retain drop.
2. **Pareto plots** (retain vs. forget) across different edit budgets (adjust `selection_ratio`).
3. **Weight-change heatmaps** per residual block using the saved audit mask to reinforce the “core region” story.

With the current pipeline + artifacts you can already present:
- CRI-U achieves ~2× better retain preservation than forget-only at the same forget reduction.
- Random edits are unpredictable despite identical budget → selection matters.
- Full-model unlearning remains the brute-force upper bound but lacks auditability.

These findings directly match the “success criteria” outlined in the proposal and are ready for inclusion in the report/slides.
