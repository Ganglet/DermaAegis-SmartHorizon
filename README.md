# DermAegis AI — Unified Skin Lesion Intelligence Workspace

**Advanced AI-powered dermoscopic skin lesion classification system combining the best features from both projects.**

> **Clinical Disclaimer:** This is a research and educational tool. Not a medical device. All predictions must be reviewed by a qualified dermatologist.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| **7-class lesion classification** | Full HAM10000 label set: akiec, bcc, bkl, df, mel, nv, vasc |
| **Monte Carlo Dropout uncertainty** | Bayesian approximation via stochastic inference — reports per-prediction uncertainty score and level (low / moderate / high) |
| **Grad-CAM explainability** | Input-gradient heatmap overlay showing which region of the image influenced the prediction |
| **Test-Time Augmentation (TTA)** | Averages multiple augmented inference passes for more stable predictions (1-8 runs) |
| **Fitzpatrick skin tone bias analysis** | Automatic detection and confidence adjustment based on training data representation |
| **Live camera capture** | Predict directly from webcam without uploading a file |
| **Class-imbalance handling** | Focal loss (γ=2.5, α=0.25) + intelligent oversampling balances training across all 7 classes |
| **Calibrated confidence** | Deterministic `training=False` inference for stable, non-inflated confidence values |
| **Triple interface** | React + FastAPI for local/server use; Gradio `app.py` for HuggingFace Spaces; Streamlit support |
| **Multi-backbone support** | EfficientNetB0, EfficientNetB3, MobileNetV2, ResNet50 |

---

## 📊 Supported Classes

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

## 🏗️ Architecture

| Component | Details |
|---|---|
| Backbones | EfficientNetB0, EfficientNetB3, MobileNetV2, ResNet50 (ImageNet pre-trained) |
| Training strategy | Two-phase transfer learning: frozen backbone → selective fine-tuning |
| Loss | Focal loss (γ=2.5, α=0.25) to penalise confident wrong predictions |
| Class balancing | Minority class oversampling + class-weight equalization |
| Augmentation | Flip, rotation (±15°), zoom (±20%), brightness (±15%), contrast (±20%), translation |
| Uncertainty | Monte Carlo Dropout — N stochastic forward passes, report mean + std per class |
| Explainability | Input-gradient Grad-CAM (robust across nested EfficientNet + mixed precision) |
| Inference | TTA + deterministic probability calibration + MC Dropout uncertainty |
| Bias Mitigation | Fitzpatrick skin tone detection + confidence adjustment + transparency warnings |

---

## 📁 Project Structure

```
dermAegis-unified/
├── api/
│   └── main.py              # FastAPI inference server — MC Dropout, Grad-CAM, TTA, Bias Analysis
├── app.py                   # Gradio interface for HuggingFace Spaces deployment
├── app/
│   └── streamlit_app.py     # Streamlit interface (optional)
├── dataset/                 # HAM10000 images and metadata (not versioned)
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # React + Vite interface with camera capture
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── models/                  # Trained checkpoints (not versioned)
│   └── cnn_model.keras
├── notebooks/
│   └── training.ipynb       # Exploratory notebook
├── utils/
│   ├── preprocessing.py     # Data pipeline, model builder, augmentation, Grad-CAM
│   ├── fitzpatrick_bias.py  # Skin tone detection and bias mitigation
│   └── __init__.py
├── train.py                 # Training entry point
├── evaluate_checkpoint.py   # Evaluate saved checkpoint on held-out test split
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### Requirements

- Python 3.10–3.12
- Node.js 18+
- HAM10000 dataset in `dataset/`:
  - `dataset/HAM10000_metadata.csv`
  - `dataset/HAM10000_images_part_1/*.jpg`
  - `dataset/HAM10000_images_part_2/*.jpg`

### Install

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

---

## 🎯 Running Locally

### Option 1: Full Stack (API + React UI)

```bash
# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend
cd frontend
npm run dev
```

Access the React UI at `http://localhost:5173`

### Option 2: Gradio Interface

```bash
python app.py
```

Opens the Gradio UI at `http://localhost:7860`

### Option 3: Streamlit Interface (Optional)

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Training

```bash
python train.py \
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

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--backbone` | efficientnetb0 | `efficientnetb0`, `efficientnetb3`, `mobilenetv2`, `resnet50` |
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

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API status, model loaded state, model path, metrics |
| `/classes` | GET | Label map and disease names |
| `/reload-model` | POST | Hot-reload the latest checkpoint without restarting |
| `/predict` | POST | Run inference on an uploaded image |

### `/predict` parameters

| Parameter | Default | Description |
|---|---|---|
| `file` | — | Image file (jpg, jpeg, png, webp) |
| `explain` | true | Include Grad-CAM heatmap overlay |
| `tta_runs` | 1 | Test-time augmentation passes (1–8) |
| `mc_runs` | 5 | Monte Carlo Dropout samples (1–10) |
| `confidence_temperature` | 1.0 | Probability sharpening/smoothing (0.1–2.0) |
| `include_probabilities` | true | Return full class distribution |
| `include_bias_analysis` | true | Include Fitzpatrick skin tone bias analysis |

### Example response

```json
{
  "predicted_label": "mel",
  "predicted_disease": "Melanoma",
  "confidence": 0.812,
  "tta_runs": 1,
  "mc_runs": 5,
  "confidence_temperature": 1.0,
  "uncertainty": {
    "score": 0.21,
    "entropy": 0.41,
    "level": "low"
  },
  "probabilities": {
    "mel": { "probability": 0.812, "std": 0.034, "disease": "Melanoma" }
  },
  "fitzpatrick_analysis": {
    "skin_tone_category": 2,
    "skin_tone_name": "Type III-IV (Medium)",
    "training_representation": "20.0%",
    "reliability_level": "MODERATE",
    "original_confidence": "81.2%",
    "adjusted_confidence": "73.1%",
    "bias_warning": "⚠️ Medium skin tone detected...",
    "recommendation": "Exercise additional caution..."
  },
  "gradcam_base64": "..."
}
```

---

## 🌍 Ethical Considerations

### Fitzpatrick Skin Tone Bias

The HAM10000 dataset is predominantly sourced from European clinics and is skewed toward lighter skin tones (Fitzpatrick types I–III).

**What this means:**
- Model performance may be lower for patients with darker skin tones (Fitzpatrick IV–VI)
- Dermoscopic appearance of lesions can differ meaningfully across skin tones

**What we're doing about it:**
- ✅ Automatic skin tone detection using ITA (Individual Typology Angle)
- ✅ Confidence adjustment based on training data representation:
  - Light skin (75% training data): No adjustment (1.0x)
  - Medium skin (20% training data): 10% reduction (0.9x)
  - Dark skin (5% training data): 25% reduction (0.75x)
- ✅ Explicit warnings when predictions may be less reliable
- ✅ Clinical recommendations based on skin tone representation
- ✅ Transparency about dataset limitations

**Bias Mitigation Strategy:**
1. **Transparency**: Explicitly detect and report skin tone category
2. **Confidence Adjustment**: Lower confidence scores for underrepresented skin tones
3. **Clinical Warnings**: Provide clear warnings when predictions may be less reliable
4. **Recommendations**: Offer specific guidance based on skin tone representation

See `FITZPATRICK_BIAS_MITIGATION.md` for detailed documentation.

---

## 🎓 Tech Stack

- **Backend:** Python 3.12, TensorFlow 2.13-2.21, FastAPI, OpenCV, Pillow
- **Frontend:** React 18, Vite, Axios
- **Gradio interface:** Gradio 4+, deployable on HuggingFace Spaces
- **ML:** EfficientNetB0/B3, MobileNetV2, ResNet50, Focal Loss, MC Dropout, Grad-CAM, TTA
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) — 10,015 dermoscopic images, 7 classes

---

## 🚢 Deploy to HuggingFace Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces), select **Gradio** as the SDK.
2. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. Upload your trained model checkpoint via the HF web UI or git-lfs:
   ```bash
   git lfs install
   git lfs track "models/*.keras"
   git add models/cnn_model.keras
   git commit -m "add model checkpoint"
   git push space main
   ```

---

## 🔬 Evaluation

```bash
# Latest checkpoint
python evaluate_checkpoint.py

# Specific checkpoint
python evaluate_checkpoint.py --model-path models/efficientnetb3_20260411_114526.best.keras
```

---

## 🛠️ Troubleshooting

### API returns 404 on expected routes
- Another project may already use port 8000
- Start this API on 8001: `uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload`
- Set `VITE_API_BASE=http://127.0.0.1:8001` before starting frontend

### Model not loading
- Ensure at least one model exists: `models/cnn_model.keras` or `models/*.best.keras`
- Call `POST /reload-model` after adding checkpoints

### Frontend cannot connect
- Confirm API host/port
- Confirm `VITE_API_BASE` matches API URL

### Grad-CAM missing
- Prediction can still succeed if Grad-CAM fails
- Check `gradcam_error` in API response

---

## 📚 References

### Academic Research
1. **Daneshjou, R., et al. (2022)** - "Disparities in dermatology AI performance on a diverse, curated clinical image set" - *Science Advances*
2. **Groh, M., et al. (2021)** - "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset" - *CVPR*
3. **Kinyanjui, N. M., et al. (2020)** - "Fairness of Classifiers Across Skin Tones in Dermatology" - *MICCAI*
4. **Adamson, A. S., & Smith, A. (2018)** - "Machine Learning and Health Care Disparities in Dermatology" - *JAMA Dermatology*

### Datasets
- **HAM10000**: Training dataset (biased toward light skin)
- **Fitzpatrick17k**: Diverse skin tone dataset for validation
- **DDI (Diverse Dermatology Images)**: Balanced dataset across skin tones

---

## 🙏 Acknowledgments

1. HAM10000 dataset creators and contributors
2. TensorFlow, FastAPI, React, and Gradio open-source communities
3. Research community working on fairness in medical AI

---

## 📄 License and Usage

Use this project for education, experimentation, and research. Ensure compliance with HAM10000 licensing and attribution requirements before redistribution or commercial use.

---

## 🔮 Future Enhancements

### Short-term
- [ ] Validate bias detection accuracy on diverse test set
- [ ] Fine-tune adjustment factors based on clinical data
- [ ] Add confidence intervals
- [ ] Implement A/B testing

### Long-term
- [ ] Train on diverse datasets (Fitzpatrick17k, DDI)
- [ ] Implement fairness-aware loss functions
- [ ] Use domain adaptation techniques
- [ ] Conduct clinical validation studies
- [ ] Partner with diverse dermatology clinics
- [ ] TFLite export for offline mobile inference

---

**Status**: ✅ Fully Integrated and Tested  
**Date**: April 2026  
**Version**: 2.0.0 (Unified)
