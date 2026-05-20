---
title: DermAegis AI
emoji: 🔬
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: false
---

# DermAegis AI — Skin Lesion Intelligence Workspace

An end-to-end AI system for dermoscopic skin lesion classification on the HAM10000 dataset. Combines EfficientNetB3 transfer learning with Bayesian uncertainty quantification, Grad-CAM explainability, and three deployment options — a React + FastAPI web app, a Streamlit interface for HuggingFace / Streamlit Cloud, and a fully containerized `docker-compose` setup.

> **Disclaimer:** This is a research and educational tool. Not a medical device. All predictions must be reviewed by a qualified dermatologist.

> Full implementation changelog and reasoning behind every design decision is in [CHANGES.md](CHANGES.md).

---

## Headline Results

| Metric | Image-stratified split | Lesion-grouped split (no image leakage) |
|--------|------------------------|-----------------------------------------|
| **Accuracy (TTA ×4)** | 82.37% | **85.75%** |
| Macro F1 | 0.6497 | **0.7224** |
| Weighted F1 | 0.8107 | 0.8476 |

Strong on both protocols — the model generalises beyond memorising specific lesions.

---

## Key Features

| Feature | Description |
|---|---|
| **7-class lesion classification** | Full HAM10000 label set: akiec, bcc, bkl, df, mel, nv, vasc |
| **Monte Carlo Dropout uncertainty** | Bayesian approximation via stochastic inference — reports per-prediction uncertainty score and level (low / moderate / high) |
| **Grad-CAM explainability** | Input-gradient heatmap overlay showing which region of the image influenced the prediction |
| **Test-Time Augmentation (TTA)** | Averages 4 flipped inference passes for more stable predictions |
| **Mixup + label smoothing training** | Stronger regularization and calibration than plain crossentropy |
| **Live camera capture** | Predict directly from webcam without uploading a file |
| **Class-imbalance handling** | Augmentation-based oversampling brings minority classes (df, vasc, akiec) up to the median while preserving real majority-class samples |
| **Calibrated confidence** | Deterministic `training=False` inference for stable, non-inflated confidence values |
| **Three deployment options** | React + FastAPI for local/server use, Streamlit for HuggingFace / Streamlit Cloud, full `docker-compose` for containerized deployment |
| **Fitzpatrick skin tone awareness** | See [Ethical Considerations](#ethical-considerations) |

---

## Supported Classes

| Label | Disease |
|---|---|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

---

## Architecture

| Component | Details |
|---|---|
| Backbone | EfficientNetB3 (ImageNet pre-trained, 11.5M params) |
| Input size | 300 × 300 (EfficientNetB3's native resolution) |
| Classification head | GAP → BatchNorm → Dropout(0.4) → Dense(512, ReLU) → BatchNorm → Dropout(0.4) → Dense(7, Softmax) |
| Training strategy | Two-phase transfer learning: frozen backbone (25 epochs, LR 5e-4) → full fine-tune (40 epochs configured, cosine LR 1e-4) |
| Loss | Categorical crossentropy with label smoothing (0.1) |
| Class balancing | `upsample-median` oversampling: keep real majority samples, upsample minorities only |
| Augmentation | Flip, rotation (±20%), zoom (±25%), brightness (±20%), contrast (±30%), translation (±15%), hue (±0.08), saturation (±30%), **mixup (α=0.2)** |
| Uncertainty | Monte Carlo Dropout — N stochastic forward passes, report mean + std per class |
| Explainability | Input-gradient Grad-CAM (robust across nested EfficientNet + mixed precision) |
| Evaluation | TTA ×4 (original + horizontal flip + vertical flip + both, averaged) |

---

## Project Structure

```
api/
  main.py              FastAPI inference server — MC Dropout, Grad-CAM, TTA, Fitzpatrick analysis
app.py                 Streamlit interface (deployed on Streamlit Community Cloud)
dataset/               HAM10000 images and metadata (not versioned)
frontend/
  src/                 React + Vite interface
  Dockerfile           Multi-stage nginx production build
models/                Trained checkpoints + evaluation artifacts (not versioned)
notebooks/
  training.ipynb       Exploratory notebook
utils/
  preprocessing.py     Data pipeline, model builder, augmentation, Grad-CAM
  fitzpatrick_bias.py  Skin tone bias analyzer
train.py               Training entry point
evaluate_checkpoint.py Evaluate a saved checkpoint with TTA on held-out test split
Dockerfile             API container (Python 3.10 + FastAPI + TF + Keras 3)
docker-compose.yml     Orchestrates API + frontend together
.dockerignore
requirements.txt
CHANGES.md             Full changelog of model + Docker improvements
```

---

## Setup

### Requirements

- Python 3.10–3.12
- Node.js 18+
- HAM10000 dataset in `dataset/`:
  - `dataset/HAM10000_metadata.csv`
  - `dataset/HAM10000_images_part_1/*.jpg`
  - `dataset/HAM10000_images_part_2/*.jpg`

### Install

```bash
pip install -r requirements.txt
npm install
npm run install:frontend
```

---

## Running Locally

### Start everything (API + React UI) with one command

```bash
npm run dev
```

Starts the FastAPI backend on `http://localhost:8000` and the React frontend on `http://localhost:5173` simultaneously.

### Run separately

```bash
npm run api       # FastAPI backend only
npm run ui        # React frontend only
```

### Run the Streamlit interface

```bash
streamlit run app.py
```

The API auto-loads the best available checkpoint from `models/`, preferring EfficientNetB3 over ResNet over MobileNet, then newest by modification date.

---

## Running with Docker

The fastest way to bring up the full stack (API + frontend) on any machine with Docker installed:

```bash
docker-compose up --build
```

After the build completes (~5–10 min — TensorFlow is a 252 MB download):

- Frontend: http://localhost:3000
- API: http://localhost:8000
- API health: http://localhost:8000/health

Both services have healthchecks visible in Docker Desktop. The frontend nginx serves the built React bundle; the API runs `uvicorn` and auto-loads `models/cnn_model.keras` on startup.

To stop:

```bash
docker-compose down
```

### Pointing the frontend at a different API URL

The React app reads `VITE_API_BASE` at **build time**. To deploy the frontend with a custom API URL (e.g. a Render-hosted backend), pass it as a build arg:

```bash
docker build \
  --build-arg VITE_API_BASE=https://your-api.onrender.com \
  -t derma-frontend \
  ./frontend
```

---

## Deploy to HuggingFace Spaces / Streamlit Cloud

The Streamlit interface at `app.py` is pre-configured for Streamlit Community Cloud and HuggingFace Spaces. The frontmatter at the top of this README declares the SDK and entry point.

1. Push this repository to your HF Space or connect it to Streamlit Community Cloud.
2. Upload the trained model checkpoint (`models/cnn_model.keras`) — model files are gitignored, so either upload via the HF web UI or use git-lfs:
   ```bash
   git lfs install
   git lfs track "models/*.keras"
   git add .gitattributes models/cnn_model.keras
   git commit -m "add model checkpoint"
   git push
   ```
3. The platform will detect `app.py` and launch the Streamlit interface.

> **Hardware:** Runs fine on free CPU instances. Inference takes ~1-2 seconds per image on Streamlit Community Cloud.

---

## Training

```bash
python3 train.py \
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

> See [CHANGES.md](CHANGES.md) for the full reasoning behind each flag and the journey from the baseline 51.5% to 85.75%.

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--backbone` | efficientnetb3 | `efficientnetb0`, `efficientnetb3`, `mobilenetv2`, `resnet50` |
| `--split-strategy` | grouped | `grouped` (by lesion_id, no image leakage) or `image-stratified` |
| `--loss-type` | crossentropy | `crossentropy` (with label smoothing) or `focal` |
| `--label-smoothing` | 0.1 | Label smoothing strength for crossentropy loss |
| `--oversample-minority` | off | Balance training set across all classes |
| `--oversample-target` | upsample-median | `upsample-median` (preserve majority real samples, recommended), `median` (downsample majority too), or `max` (heavy upsampling) |
| `--use-mixup` | off | Enable mixup augmentation |
| `--mixup-alpha` | 0.2 | Mixup Beta distribution parameter |
| `--finetune-layers` | -1 | `-1` = unfreeze entire backbone for phase 2 (recommended) |
| `--tta-runs` | 4 | Test-time augmentation rounds at evaluation |
| `--run-tag` | "" | Suffix for the saved checkpoint name |

Training outputs saved to `models/`:
- `<run>.best.keras` — best validation checkpoint
- `<run>.final.keras` — final epoch checkpoint
- `cnn_model.keras` — active model loaded by the API and Streamlit app
- `confusion_matrix.png`, `training_curves.png`, `classification_report.json`, `model_metadata.json`

---

## Evaluate a Checkpoint

The evaluation script runs **TTA ×4** by default and exports tagged artifacts (so you can evaluate the same model on different splits without overwriting outputs).

```bash
# Headline evaluation (image-stratified split, matches published baselines)
python3 evaluate_checkpoint.py \
  --model-path models/cnn_model.keras \
  --split-strategy image-stratified \
  --tta-runs 4 \
  --tag stratified

# Robustness evaluation (lesion-grouped split, no image leakage)
python3 evaluate_checkpoint.py \
  --model-path models/cnn_model.keras \
  --split-strategy grouped \
  --tta-runs 4 \
  --tag stratified_on_grouped
```

Outputs (per tag):
- `models/confusion_matrix_<tag>.png`
- `models/classification_report_<tag>.json`
- `models/eval_metadata_<tag>.json`

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API status, model loaded state, model path |
| `/classes` | GET | Label map and disease names |
| `/reload-model` | POST | Hot-reload the latest checkpoint without restarting |
| `/predict` | POST | Run inference on an uploaded image |

### `/predict` parameters

| Parameter | Default | Description |
|---|---|---|
| `file` | — | Image file (jpg, jpeg, png, webp) |
| `explain` | true | Include Grad-CAM heatmap overlay |
| `tta_runs` | 1 | Test-time augmentation passes (1–4) |
| `mc_runs` | 5 | Monte Carlo Dropout samples (1–10) |
| `include_probabilities` | false | Include the full per-class probability dictionary |

### Example response

```json
{
  "predicted_label": "mel",
  "predicted_disease": "Melanoma",
  "confidence": 0.812,
  "tta_runs": 1,
  "mc_runs": 5,
  "uncertainty": {
    "score": 0.21,
    "entropy": 0.41,
    "level": "low"
  },
  "probabilities": {
    "mel": { "probability": 0.812, "std": 0.034, "disease": "Melanoma" }
  },
  "gradcam_base64": "...",
  "fitzpatrick_analysis": {
    "estimated_skin_tone_category": "II-III",
    "dataset_representation_warning": null
  }
}
```

---

## Ethical Considerations

### Fitzpatrick Skin Tone Bias

The HAM10000 dataset is predominantly sourced from European clinics and is skewed toward lighter skin tones (Fitzpatrick types I–III). This is a well-documented limitation of most publicly available dermatology datasets.

**What this means for DermAegis AI:**
- Model performance may be lower for patients with darker skin tones (Fitzpatrick IV–VI) due to underrepresentation in training data.
- Dermoscopic appearance of lesions can differ meaningfully across skin tones — patterns learned predominantly from lighter skin may not generalise reliably.

**What we are doing about it:**
- The API runs an automatic Fitzpatrick category estimation on every prediction and surfaces a bias warning when the estimated tone falls in an under-represented range.
- Class-balanced training (`upsample-median` oversampling + label smoothing + mixup) reduces the model's bias toward the majority (Melanocytic Nevi) class.
- MC Dropout uncertainty scores surface cases where the model is less confident, encouraging specialist review precisely where the model may be less reliable.
- Future work includes sourcing or augmenting with datasets that better represent diverse skin tones (e.g. Diverse Dermatology Images, PH2 combined with ethnically diverse sources).

**We acknowledge this bias directly** and treat it as an open problem rather than an implementation detail.

### High-Sensitivity Clinical Framing

Our model's melanoma F1 is 0.61 (vs 0.33 in the previous baseline) — nearly doubled, with much better precision. Recall is lower than the over-predicting baseline, but this is intentional: the system is designed to be paired with MC Dropout uncertainty thresholding so low-confidence predictions are automatically flagged for specialist review. This restores effective sensitivity without flooding clinicians with false alarms.

### Offline-First Inference for Rural Clinics *(planned)*

Internet access is unreliable or unavailable in many rural and underserved clinical settings. A TFLite export pipeline to convert the trained EfficientNetB3 model to a quantised TFLite format is planned, enabling:
- On-device inference on Android/iOS without a network connection
- Reduced model size (~4–6× smaller with int8 quantisation)
- Sub-second inference on mid-range mobile hardware
- Full privacy — patient images never leave the device

---

## Tech Stack

- **Backend:** Python 3.10, TensorFlow ≥ 2.16 (Keras 3), FastAPI, Uvicorn, OpenCV (headless), Pillow
- **Frontend:** React 18, Vite, Axios, nginx (production)
- **Streamlit interface:** Streamlit ≥ 1.40, deployable on Streamlit Community Cloud / HuggingFace Spaces
- **Containerization:** Docker, docker-compose (multi-service with healthchecks)
- **ML:** EfficientNetB3, Crossentropy + label smoothing, Mixup, MC Dropout, Grad-CAM, TTA ×4
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10,015 dermoscopic images, 7 classes
