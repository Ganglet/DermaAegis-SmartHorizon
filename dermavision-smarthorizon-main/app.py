"""
DermAegis AI — Gradio interface for HuggingFace Spaces deployment.
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
    print("Model loaded.")
    return True


_load_model()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _mc_dropout_std(model_input: np.ndarray, runs: int = 5) -> np.ndarray:
    import tensorflow as tf
    replicated = np.repeat(model_input, runs, axis=0)
    all_preds = _model(replicated, training=True).numpy()
    return all_preds.std(axis=0)


def _predictive_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _overlay_gradcam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
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

    # Calibrated probabilities
    probs = _model(model_input, training=False).numpy()[0]
    mc_std = _mc_dropout_std(model_input)

    top_idx = int(np.argmax(probs))
    top_label = LABEL_NAMES[top_idx]
    confidence = float(probs[top_idx])

    entropy = _predictive_entropy(probs)
    uncertainty_score = float(np.clip(entropy / np.log(len(LABEL_NAMES)), 0.0, 1.0))
    uncertainty_level = (
        "High" if uncertainty_score > 0.70 else
        "Moderate" if uncertainty_score > 0.45 else
        "Low"
    )

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
    except Exception:
        pass

    return result_md, pil_image, gradcam_img


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="DermAegis AI",
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="red"),
) as demo:
    gr.Markdown(
        """
# DermAegis AI — Skin Lesion Intelligence Workspace

AI-powered dermoscopic lesion classification across 7 HAM10000 lesion types.
Combines **EfficientNetB3** transfer learning with **Monte Carlo Dropout** uncertainty
quantification and **Grad-CAM** explainability.

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
            result_output = gr.Markdown(label="Prediction")

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
        """
    )


if __name__ == "__main__":
    demo.launch()
