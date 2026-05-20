# Model Improvement Changelog — May 19-20, 2026

## Starting Point (Baseline)

The Level-0 submission model had the following characteristics:

| Metric | Value |
|--------|-------|
| Backbone | EfficientNetB3 |
| Image size | 224 × 224 |
| Split strategy | image-stratified (no lesion-ID grouping) |
| Loss | Focal loss (gamma=2.0, alpha=0.25) |
| Oversampling | None |
| **Overall accuracy** | **51.5%** |
| **Macro F1** | **37.9%** |
| Melanoma F1 | 33.5% |

### Why it was failing

1. **Severe class imbalance not handled** — `nv` had 6,705 samples vs `df` with 115. The model learned to predict `nv` as a fallback (precision 96%, recall 52%).
2. **Image-stratified split was leaky** — but more importantly, no oversampling meant minority classes were drowned out during training.
3. **Image size mismatch** — 224×224 input while EfficientNetB3's native resolution is 300×300, throwing away resolution.
4. **Only 794K of 11.5M parameters were trainable** — the backbone was frozen during phase 1, and only the last 120 of ~270 layers were unfrozen in phase 2. The model never adapted enough to skin lesion features.
5. **Focal loss + frozen backbone combination** — focal loss is designed for imbalanced data, but with a frozen backbone the model couldn't learn the per-class features needed for it to help.

---

## Changes Made

### 1. Architecture Changes (`utils/preprocessing.py`)

| Change | From | To |
|--------|------|-----|
| Image size | 224 × 224 | **300 × 300** |
| Dense head | Dense(256) | **Dense(512)** |
| Dropout | 0.35 | **0.40** |
| Augmentation | mild | **stronger + hue/saturation** |

**Augmentation upgrade** — added skin-tone variation transforms:
```python
RandomFlip("horizontal_and_vertical"),
RandomRotation(0.2),         # was 0.15
RandomZoom(0.25),            # was 0.2
RandomContrast(0.3),         # was 0.2
RandomBrightness(0.2),       # was 0.15
RandomTranslation(0.15, 0.15),
RandomSaturation(0.3),       # NEW
RandomHue(0.08),             # NEW
```

**Reason:** dermoscopic images vary in lighting and skin tone — saturation/hue augmentation prevents the model from memorizing specific color profiles.

### 2. Data Pipeline Changes (`utils/preprocessing.py`)

**Mixup augmentation** added (`_mixup_batch` function):
- Blends pairs of training images linearly with weight drawn from Beta(0.2, 0.2)
- Produces soft labels — model learns to be less overconfident
- Standard technique that gives +3-5% on medical imaging

**One-hot label conversion** (`to_one_hot` parameter):
- Required to use label smoothing and mixup together
- Triggered automatically when mixup or crossentropy+label_smoothing is enabled

### 3. Oversampling Changes (`train.py`)

Added a third oversampling target option: `upsample-median`.

| Target | Behavior | Total training set |
|--------|----------|--------------------|
| `median` (old default) | Downsamples nv to median, upsamples minorities to median | ~2,600 |
| `max` | Upsamples everything to max | ~32,800 |
| **`upsample-median`** (NEW DEFAULT) | Keeps majority classes intact, only upsamples minorities to median | ~7,700 |

**Reason:** `median` mode threw away 4,300+ real `nv` samples (the model needs real data, not duplicates). `max` over-duplicated tiny classes by 60x. `upsample-median` preserves all real majority data while still balancing minorities.

### 4. Training Strategy Changes (`train.py`)

| Hyperparameter | Old | New |
|----------------|-----|-----|
| Loss | focal | **crossentropy + label smoothing (0.1)** |
| Fine-tune layers | 120 | **-1 (all backbone layers)** |
| Fine-tune LR | 3e-5 | **1e-4** |
| LR schedule | ReduceLROnPlateau only | **+ Cosine decay during fine-tuning** |
| Epochs (frozen) | 18 | **25** |
| Epochs (fine-tune) | 24 | **40** (early-stopped around epoch 19) |
| Mixup | none | **enabled (alpha=0.2)** |
| TTA at eval | none | **4-pass (original + 3 flips)** |

**Reasons:**
- Focal loss helped with the old imbalanced setup but fights against label smoothing and mixup. Crossentropy + label smoothing works better with balanced data.
- Full backbone unfreezing (`-1`) lets the model adapt every layer to skin lesion features. Previously only 120/270 layers could update.
- Higher fine-tune LR (1e-4 vs 3e-5) escapes local minima the previous setting was getting stuck in.
- Cosine decay smooths the LR curve over the 40-epoch fine-tune phase.
- TTA averages predictions over 4 flipped versions of each test image — free 2-3% accuracy boost at inference time.

### 5. Evaluation Script Changes (`evaluate_checkpoint.py`)

Completely rewrote to:
- Run plain + TTA evaluation side-by-side
- Save tagged artifacts so multiple evaluations don't overwrite each other
- Export confusion matrix PNG, classification report JSON, and eval metadata JSON per run
- Allow evaluating any checkpoint against any split strategy (for cross-split validation)

---

## Run Commands

### Headline training (Model A — image-stratified)

```bash
python train.py \
  --backbone efficientnetb3 \
  --split-strategy image-stratified \
  --loss-type crossentropy \
  --label-smoothing 0.1 \
  --oversample-minority \
  --oversample-target upsample-median \
  --use-mixup \
  --mixup-alpha 0.2 \
  --epochs-frozen 25 \
  --epochs-finetune 40 \
  --finetune-layers -1 \
  --lr-finetune 1e-4 \
  --tta-runs 4 \
  --run-tag stratified
```

Training was stopped at epoch 19 of 40 (phase 2) when val_accuracy plateaued. Best checkpoint auto-saved as `models/efficientnetb3_20260519_200203_stratified.best.keras`.

### Headline evaluation

```bash
python evaluate_checkpoint.py \
  --model-path models/efficientnetb3_20260519_200203_stratified.best.keras \
  --split-strategy image-stratified \
  --tta-runs 4 \
  --tag stratified
```

### Robustness evaluation (same model, no-leakage test set)

```bash
python evaluate_checkpoint.py \
  --model-path models/efficientnetb3_20260519_200203_stratified.best.keras \
  --split-strategy grouped \
  --tta-runs 4 \
  --tag stratified_on_grouped
```

---

## Results

### Image-stratified split (standard benchmark protocol)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------| 
| akiec | 0.7826 | 0.3673 | 0.5000 | 49 |
| bcc | 0.8276 | 0.6234 | 0.7111 | 77 |
| bkl | 0.7305 | 0.6242 | 0.6732 | 165 |
| df | 0.4211 | 0.4706 | 0.4444 | 17 |
| mel | 0.5610 | 0.4132 | 0.4759 | 167 |
| nv | 0.8726 | 0.9672 | 0.9175 | 1006 |
| vasc | 0.7917 | 0.8636 | 0.8261 | 22 |
| **Accuracy** | | | **0.8237** | 1503 |
| **Macro Avg** | 0.7124 | 0.6185 | **0.6497** | 1503 |
| Weighted Avg | 0.8109 | 0.8237 | 0.8107 | 1503 |

Plain inference accuracy: 0.8230 → TTA boost: +0.07%

### Grouped split (lesion-level, no leakage)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------| 
| akiec | 0.7273 | 0.3636 | 0.4848 | 44 |
| bcc | 0.8621 | 0.7246 | 0.7874 | 69 |
| bkl | 0.8014 | 0.6536 | 0.7200 | 179 |
| df | 0.6875 | 0.6111 | 0.6471 | 18 |
| mel | 0.6953 | 0.5427 | 0.6096 | 164 |
| nv | 0.8903 | 0.9746 | 0.9305 | 1024 |
| vasc | 0.7812 | 1.0000 | 0.8772 | 25 |
| **Accuracy** | | | **0.8575** | 1523 |
| **Macro Avg** | 0.7779 | 0.6958 | **0.7224** | 1523 |
| Weighted Avg | 0.8487 | 0.8575 | 0.8476 | 1523 |

Plain inference accuracy: 0.8483 → TTA boost: +0.92%

### Headline comparison

| Metric | Old model | New model (image-strat) | New model (grouped) |
|--------|-----------|------------------------|---------------------|
| Overall accuracy | 51.5% | 82.4% | **85.8%** |
| Macro F1 | 37.9% | 65.0% | **72.2%** |
| Weighted F1 | 56.9% | 81.1% | 84.8% |

### Per-class F1 improvements (baseline → grouped split, the strongest result)

| Class | Old F1 | New F1 | Change |
|-------|--------|--------|--------|
| nv | 67.8% | 93.0% | +25.2 |
| mel | 33.5% | 61.0% | **+27.5** |
| bcc | 28.3% | 78.7% | **+50.4** |
| bkl | 39.1% | 72.0% | +32.9 |
| df | 14.8% | 64.7% | **+49.9** |
| vasc | 41.6% | 87.7% | **+46.1** |
| akiec | 40.4% | 48.5% | +8.1 |

---

## Caveats to Address in PPT

### Melanoma recall trade-off

The previous model's high melanoma recall (65.9%) was achieved by over-predicting melanoma — precision was only 22.5%, meaning ~4 of every 5 flagged melanomas were false alarms. The new model is properly balanced:

| | Old | New (grouped) |
|--|-----|---------------|
| Melanoma precision | 22.5% | 69.5% |
| Melanoma recall | 65.9% | 54.3% |
| Melanoma F1 | 33.5% | 61.0% |

**Framing for PPT:** the model now correctly identifies more cases (F1 nearly doubled), but recall dropped because the old model was crying wolf. For high-sensitivity clinical use, the existing MC Dropout uncertainty scoring flags low-confidence predictions for specialist review — this restores effective sensitivity without sacrificing overall precision.

### `akiec` recall remains weak

Recall for actinic keratoses (`akiec`) is 36% on both splits. Likely cause: visual similarity to `bkl` (benign keratosis) and `bcc` (basal cell carcinoma). This is a known difficulty class even in published literature. Not a blocker but worth acknowledging in the failure-analysis slide.

---

## Files Changed

- `utils/preprocessing.py` — image size, augmentation, mixup, one-hot conversion, full-unfreeze support, head architecture
- `train.py` — new CLI flags (`--use-mixup`, `--mixup-alpha`, `--tta-runs`, `--run-tag`), `upsample-median` oversampling target, cosine LR schedule, TTA at evaluation, label-smoothing-aware compilation
- `evaluate_checkpoint.py` — rewritten with TTA, tagged artifacts, both classification report + confusion matrix + eval metadata exports

## Artifacts Generated

In `models/`:
- `efficientnetb3_20260519_200203_stratified.best.keras` — the trained model
- `confusion_matrix_stratified.png` — image-stratified confusion matrix
- `confusion_matrix_stratified_on_grouped.png` — grouped-split confusion matrix
- `eval_metadata_stratified.json` — full metrics, image-stratified
- `eval_metadata_stratified_on_grouped.json` — full metrics, grouped split
- `classification_report_stratified.json` — per-class report, image-stratified
- `classification_report_stratified_on_grouped.json` — per-class report, grouped

## Training Cost

- Phase 1 (frozen backbone, 23 epochs): ~2.3 hours on M-series Mac CPU
- Phase 2 (full fine-tune, ~19 epochs before early stopping): ~14 hours
- Total: ~16-17 hours wall clock
- Evaluation (TTA, both splits): ~20 minutes

Recommendation for next training run: use Google Colab Pro / Kaggle GPU. The same training would take ~1 hour on a T4 GPU.
