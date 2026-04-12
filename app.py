"""
DermAegis AI — Unified Gradio interface for HuggingFace Spaces deployment.

Features:
- Monte Carlo Dropout uncertainty quantification
- Grad-CAM explainability
- Fitzpatrick skin tone bias analysis
- Support for EfficientNetB0, EfficientNetB3, MobileNetV2, ResNet50

Run locally:  python app.py
HF Spaces:    push repo + model checkpoint, Spaces picks up app.py automatically.

Model placement:
  - Put your trained checkpoint at models/cnn_model.keras   (preferred)
  - Or any *.best.keras file inside models/                  (auto-selected)
"""

import os

import cv2
import numpy as np
import gradio as gr
from PIL import Image

from utils.preprocessing import (
    DISEASE_NAMES,
    LABEL_NAMES,
    find_last_conv_layer,
    load_trained_model,
    make_gradcam_heatmap,
    prepare_inference_image,
)
from utils.fitzpatrick_bias import (
    estimate_skin_tone_category,
    generate_bias_report,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_last_conv_layer = None


def _resolve_model_path() -> str | None:
    preferred = os.path.join(PROJECT_ROOT, "models", "cnn_model.keras")
    if os.path.isfile(preferred):
        return preferred

    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.isdir(models_dir):
        return None

    candidates = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".best.keras")
    ]
    if not candidates:
        return None

    backbone_priority = {"efficientnetb3": 4, "efficientnet": 3, "resnet": 2, "mobilenet": 1}

    def _score(path: str):
        name = os.path.basename(path).lower()
        priority = next((v for k, v in backbone_priority.items() if k in name), 0)
        return priority, os.path.getmtime(path)

    return max(candidates, key=_score)


def _load_model() -> bool:
    global _model, _last_conv_layer
    path = _resolve_model_path()
    if path is None:
        print("No model checkpoint found in models/")
        return False
    print(f"Loading model from {path}")
    _model = load_trained_model(path)
    _last_conv_layer = find_last_conv_layer(_model)
    print("Model loaded successfully.")
    return True


_load_model()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _mc_dropout_std(model_input: np.ndarray, runs: int = 5) -> np.ndarray:
    """Monte Carlo Dropout for uncertainty estimation."""
    import tensorflow as tf
    replicated = np.repeat(model_input, runs, axis=0)
    all_preds = _model(replicated, training=True).numpy()
    return all_preds.std(axis=0)


def _predictive_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of the predictive distribution."""
    clipped = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _overlay_gradcam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    img = image_np.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    merged = np.clip((1 - alpha) * img + alpha * colored, 0, 1)
    return np.uint8(merged * 255)


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(image: np.ndarray):
    if _model is None:
        return (
            "**Model not loaded.** Place a trained checkpoint at `models/cnn_model.keras` and restart.",
            None,
            None,
        )
    if image is None:
        return "Upload a dermoscopic skin lesion image to begin.", None, None

    pil_image = Image.fromarray(image).convert("RGB")
    model_input = prepare_inference_image(pil_image)

    # Calibrated probabilities (training=False for deterministic inference)
    probs = _model(model_input, training=False).numpy()[0]
    
    # Monte Carlo Dropout for uncertainty
    mc_std = _mc_dropout_std(model_input, runs=5)

    top_idx = int(np.argmax(probs))
    top_label = LABEL_NAMES[top_idx]
    confidence = float(probs[top_idx])

    # Uncertainty quantification
    entropy = _predictive_entropy(probs)
    uncertainty_score = float(np.clip(entropy / np.log(len(LABEL_NAMES)), 0.0, 1.0))
    uncertainty_level = (
        "High" if uncertainty_score > 0.70 else
        "Moderate" if uncertainty_score > 0.45 else
        "Low"
    )

    # Fitzpatrick bias analysis
    bias_lines = []
    try:
        skin_tone_category = estimate_skin_tone_category(model_input)
        bias_report = generate_bias_report(skin_tone_category, confidence, dataset_name="ham10000")
        
        bias_lines.append("\n---\n### 🌍 Fitzpatrick Skin Tone Bias Analysis\n")
        bias_lines.append(f"**Detected Skin Tone:** {bias_report['skin_tone_name']}  ")
        bias_lines.append(f"**Training Data Representation:** {bias_report['training_representation']}  ")
        bias_lines.append(f"**Model Reliability:** {bias_report['reliability_level']}  ")
        bias_lines.append(f"**Original Confidence:** {bias_report['original_confidence']}  ")
        bias_lines.append(f"**Bias-Adjusted Confidence:** {bias_report['adjusted_confidence']}  ")
        
        if bias_report.get('bias_warning'):
            bias_lines.append(f"\n> {bias_report['bias_warning']}")
        
        bias_lines.append(f"\n**Recommendation:** {bias_report['recommendation']}  ")
        bias_lines.append(f"\n*{bias_report['dataset_bias_note']}*")
    except Exception as e:
        bias_lines.append(f"\n*Bias analysis unavailable: {str(e)}*")

    # Prediction summary
    lines = [
        f"## {DISEASE_NAMES[top_label]} (`{top_label}`)",
        f"**Confidence:** {confidence * 100:.1f}%  ",
        f"**Model Uncertainty:** {uncertainty_score * 100:.1f}% — {uncertainty_level}",
    ]
    if uncertainty_level == "High":
        lines.append("\n> ⚠️ High uncertainty — specialist review strongly recommended.")

    lines.append("\n---\n### Class Probabilities\n")
    sorted_probs = sorted(
        [
            (LABEL_NAMES[i], DISEASE_NAMES[LABEL_NAMES[i]], float(probs[i]), float(mc_std[i]))
            for i in range(len(probs))
        ],
        key=lambda x: -x[2],
    )
    for label, disease, prob, std in sorted_probs:
        filled = round(prob * 20)
        bar = "█" * filled + "░" * (20 - filled)
        lines.append(f"`{bar}` **{disease}** — {prob * 100:.1f}% ±{std * 100:.1f}%  ")

    # Add bias analysis
    lines.extend(bias_lines)

    lines.append(
        "\n---\n> *Clinical Safety Notice: Research tool only. Not a medical device. "
        "Final diagnosis must be performed by a qualified dermatologist.*"
    )

    result_md = "\n".join(lines)

    # Grad-CAM overlay
    gradcam_img = None
    try:
        heatmap = make_gradcam_heatmap(model_input.astype(np.float32), _model, _last_conv_layer)
        gradcam_img = Image.fromarray(_overlay_gradcam(model_input[0], heatmap))
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")

    return result_md, pil_image, gradcam_img


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="DermAegis AI - Unified",
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="red"),
) as demo:
    gr.Markdown(
        """
# DermAegis AI — Unified Skin Lesion Intelligence Workspace

Advanced AI-powered dermoscopic lesion classification across 7 HAM10000 lesion types.

**Key Features:**
- 🧠 **Monte Carlo Dropout** uncertainty quantification
- 🔍 **Grad-CAM** explainability heatmaps
- 🌍 **Fitzpatrick skin tone bias** analysis and mitigation
- 🎯 **Multi-backbone support** (EfficientNetB0/B3, MobileNetV2, ResNet50)

> ⚠️ **Clinical Safety Notice:** Research and educational tool only. Not a medical device.
> All predictions must be reviewed by a qualified dermatologist.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Dermoscopic Image",
                type="numpy",
                sources=["upload", "webcam"],
            )
            submit_btn = gr.Button("Run Prediction", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Prediction Results")

    with gr.Row():
        original_out = gr.Image(label="Input Image", type="pil")
        gradcam_out = gr.Image(label="Grad-CAM Explainability", type="pil")

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[result_output, original_out, gradcam_out],
    )

    gr.Markdown(
        """
---
**Supported classes:** Actinic Keratoses · Basal Cell Carcinoma · Benign Keratosis ·
Dermatofibroma · Melanoma · Melanocytic Nevi · Vascular Lesions

**Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

**Features:**
- Test-Time Augmentation (TTA) for robust predictions
- Bayesian uncertainty via Monte Carlo Dropout
- Fitzpatrick skin tone bias detection and confidence adjustment
- Input-gradient Grad-CAM for all backbone architectures
        """
    )


if __name__ == "__main__":
    demo.launch()
