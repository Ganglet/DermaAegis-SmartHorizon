---
title: DermAegis AI
emoji: 🔬
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# DermAegis AI — Skin Lesion Intelligence Workspace

An end-to-end AI system for dermoscopic skin lesion classification built on the HAM10000 dataset. Combines EfficientNetB3 transfer learning with Bayesian uncertainty quantification, Grad-CAM explainability, and dual interfaces — a real-time React web app and a Gradio interface deployable on HuggingFace Spaces.

> **Disclaimer:** This is a research and educational tool. Not a medical device. All predictions must be reviewed by a qualified dermatologist.

---

## Key Features

| Feature | Description |
|---|---|
| **7-class lesion classification** | Full HAM10000 label set: akiec, bcc, bkl, df, mel, nv, vasc |
| **Monte Carlo Dropout uncertainty** | Bayesian approximation via stochastic inference — reports per-prediction uncertainty score and level (low / moderate / high) |
| **Grad-CAM explainability** | Input-gradient heatmap overlay showing which region of the image influenced the prediction |
| **Test-Time Augmentation (TTA)** | Averages multiple augmented inference passes for more stable predictions |
| **Live camera capture** | Predict directly from webcam without uploading a file |
| **Class-imbalance handling** | Focal loss (γ=2.5, α=0.25) + intelligent oversampling balances training across all 7 classes |
| **Calibrated confidence** | Deterministic `training=False` inference for stable, non-inflated confidence values |
| **Dual interface** | React + FastAPI for local/server use; Gradio `app.py` for HuggingFace Spaces |
| **Offline-first inference** *(in progress)* | TFLite export for edge deployment in rural clinics with no reliable internet — being developed by the mobile team |
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
| Backbone | EfficientNetB3 (ImageNet pre-trained) |
| Training strategy | Two-phase transfer learning: frozen backbone → selective fine-tuning |
| Loss | Focal loss (γ=2.5, α=0.25) to penalise confident wrong predictions |
| Class balancing | Minority class oversampling + class-weight equalization |
| Augmentation | Flip, rotation (±15°), zoom (±20%), brightness (±15%), contrast (±20%), translation |
| Uncertainty | Monte Carlo Dropout — N stochastic forward passes, report mean + std per class |
| Explainability | Input-gradient Grad-CAM (robust across nested EfficientNet + mixed precision) |
| Inference | TTA + deterministic probability calibration |

---

## Project Structure

```
api/
  main.py              FastAPI inference server — MC Dropout, Grad-CAM, TTA
app.py                 Gradio interface for HuggingFace Spaces deployment
dataset/               HAM10000 images and metadata (not versioned)
frontend/
  src/                 React + Vite interface
models/                Trained checkpoints (not versioned)
notebooks/
  training.ipynb       Exploratory notebook
utils/
  preprocessing.py     Data pipeline, model builder, augmentation, Grad-CAM
train.py               Training entry point
evaluate_checkpoint.py Evaluate saved checkpoint on held-out test split
requirements.txt
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

### Run Gradio interface

```bash
python app.py
```

Opens the Gradio UI at `http://localhost:7860`.

The API auto-loads the best available checkpoint from `models/`, preferring EfficientNetB3 over ResNet over MobileNet, then newest by modification date.

---

## Deploy to HuggingFace Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces), select **Gradio** as the SDK.
2. Push this repository to the Space (or clone and push):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. Upload your trained model checkpoint. Because model files are gitignored, upload manually via the HF web UI or git-lfs:
   ```bash
   # Option A — HF web UI: go to Files → Upload file → models/cnn_model.keras
   # Option B — git-lfs
   git lfs install
   git lfs track "models/*.keras"
   git add models/cnn_model.keras
   git commit -m "add model checkpoint"
   git push space main
   ```
4. The Space will detect `app.py` and launch the Gradio interface automatically.

> **Hardware:** The Space runs fine on a CPU instance. For faster inference, upgrade to a T4 GPU Space.

---

## Training

```bash
python3 train.py \
  --backbone efficientnetb3 \
  --epochs-frozen 20 \
  --epochs-finetune 30 \
  --loss-type focal \
  --focal-gamma 2.5 \
  --focal-alpha 0.25 \
  --oversample-minority \
  --oversample-target max \
  --mixed-precision
```

> Always use `python3`. Models are saved with the system TensorFlow — running with a different environment (e.g. Anaconda) causes a Keras version mismatch.

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--backbone` | efficientnetb3 | `efficientnetb0`, `efficientnetb3`, `mobilenetv2`, `resnet50` |
| `--split-strategy` | grouped | `grouped` (by lesion_id, lower leakage) or `image-stratified` |
| `--loss-type` | crossentropy | `crossentropy` or `focal` |
| `--oversample-minority` | off | Balance training set across all classes |
| `--oversample-target` | median | `median` (capped balance) or `max` (upsample to majority class size) |
| `--mixed-precision` | off | float16 training for faster runs |

Training outputs saved to `models/`:
- `<run>.best.keras` — best validation checkpoint
- `<run>.final.keras` — final epoch checkpoint
- `cnn_model.keras` — active model loaded by the API and Gradio app
- `confusion_matrix.png`, `training_curves.png`, `classification_report.json`, `model_metadata.json`

---

## Evaluate a Checkpoint

```bash
# Latest checkpoint
python3 evaluate_checkpoint.py

# Specific checkpoint
python3 evaluate_checkpoint.py --model-path models/efficientnetb3_20260411_114526.best.keras
```

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
  "gradcam_base64": "..."
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
- Class-weighted focal loss and oversampling reduce the model's bias toward the majority (Melanocytic Nevi) class, improving recall on rarer lesion types.
- Model uncertainty scores surface cases where the model is less confident, encouraging specialist review precisely where the model may be less reliable.
- The system explicitly shows uncertainty level (low / moderate / high) so clinicians are never presented with a false sense of precision.
- Future work includes sourcing or augmenting with datasets that better represent diverse skin tones (e.g. Diverse Dermatology Images, PH2 combined with ethnically diverse sources).

**We acknowledge this bias directly** and treat it as an open problem rather than an implementation detail.

### Offline-First Inference for Rural Clinics *(in progress)*

Internet access is unreliable or unavailable in many rural and underserved clinical settings. The mobile team is developing a **TFLite export pipeline** to convert the trained EfficientNetB3 model to a quantised TFLite format, enabling:
- On-device inference on Android/iOS without a network connection
- Reduced model size (~4–6× smaller with int8 quantisation)
- Sub-second inference on mid-range mobile hardware
- Full privacy — patient images never leave the device

This feature is under active development and will be integrated as a companion mobile app.

---

## Tech Stack

- **Backend:** Python 3.12, TensorFlow 2.21, FastAPI, OpenCV, Pillow
- **Frontend:** React 18, Vite, Axios
- **Gradio interface:** Gradio 4+, deployable on HuggingFace Spaces
- **ML:** EfficientNetB3, Focal Loss, MC Dropout, Grad-CAM, TTA
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) — 10,015 dermoscopic images, 7 classes
