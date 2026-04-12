import base64
import io
import json
import os
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_model.keras")
MODEL_META_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

app = FastAPI(title="DermAegis AI - Unified Skin Cancer Detection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: tf.keras.Model | None = None
_last_conv_layer: str | None = None
_active_model_path: str | None = None


def _extract_quality_metrics(meta: dict[str, Any]) -> dict[str, float]:
    metrics = meta.get("metrics") or {}
    if not isinstance(metrics, dict):
        return {}

    quality = {}
    if "accuracy" in metrics:
        quality["accuracy"] = float(metrics["accuracy"])
    if "top2_acc" in metrics:
        quality["top2_acc"] = float(metrics["top2_acc"])
    if "loss" in metrics:
        quality["loss"] = float(metrics["loss"])
    return quality


def _resolve_model_path() -> str | None:
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH

    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.isdir(models_dir):
        return None

    candidates = [
        os.path.join(models_dir, name)
        for name in os.listdir(models_dir)
        if name.endswith(".best.keras")
    ]
    if not candidates:
        return None

    # Prefer historically stronger backbones first, then newest checkpoint in that tier.
    backbone_priority = {"efficientnetb3": 4, "efficientnet": 3, "resnet": 2, "mobilenet": 1}

    def _score(path: str) -> tuple[int, float]:
        name = os.path.basename(path).lower()
        priority = 0
        for key, value in backbone_priority.items():
            if key in name:
                priority = value
                break
        return priority, os.path.getmtime(path)

    return max(candidates, key=_score)


def _load_latest_model() -> bool:
    global _model
    global _last_conv_layer
    global _active_model_path

    resolved_path = _resolve_model_path()
    if resolved_path is None:
        return False

    _model = load_trained_model(resolved_path)
    _active_model_path = resolved_path
    _last_conv_layer = find_last_conv_layer(_model)
    return True


@app.on_event("startup")
def load_model_on_startup() -> None:
    _load_latest_model()


def _get_model() -> tf.keras.Model:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not found. Train the model first.")
    return _model


def _overlay_gradcam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    image_np = image_np.astype(np.float32)
    if image_np.max() > 1:
        image_np = image_np / 255.0

    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    merged = np.clip((1 - alpha) * image_np + alpha * heatmap_color, 0, 1)

    encoded_ok, encoded = cv2.imencode(".png", np.uint8(merged * 255))
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _load_model_meta() -> dict[str, Any]:
    if not os.path.isfile(MODEL_META_PATH):
        return {}
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _tta_augment(image_batch: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    tensor = tf.image.random_flip_left_right(tensor)
    tensor = tf.image.random_flip_up_down(tensor)
    tensor = tf.image.random_brightness(tensor, max_delta=0.06)
    tensor = tf.image.random_contrast(tensor, lower=0.9, upper=1.1)
    return tf.clip_by_value(tensor, 0.0, 255.0).numpy()


def _apply_confidence_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if np.isclose(temperature, 1.0):
        return probs

    clipped = np.clip(probs.astype(np.float64), 1e-8, 1.0)
    sharpened = np.power(clipped, 1.0 / temperature)
    normalized = sharpened / np.sum(sharpened)
    return normalized.astype(np.float32)


def _mc_dropout_std(model: tf.keras.Model, model_input: np.ndarray, runs: int) -> np.ndarray:
    """Run Monte Carlo Dropout inference to estimate per-class std (uncertainty only).
    Uses training=True to activate dropout randomness. Replicates input N times for
    a single batched forward pass — much faster than N sequential calls on CPU."""
    replicated = np.repeat(model_input, runs, axis=0)     # (runs, H, W, 3)
    all_preds = model(replicated, training=True).numpy()  # (runs, 7)
    return all_preds.std(axis=0)


def _predictive_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of the mean predictive distribution — scalar uncertainty score."""
    clipped = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


@app.get("/health")
def health() -> dict[str, Any]:
    meta = _load_model_meta()
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": _active_model_path or MODEL_PATH,
        "model_metrics": _extract_quality_metrics(meta),
    }


@app.get("/classes")
def classes() -> dict[str, Any]:
    return {
        "labels": LABEL_NAMES,
        "disease_names": DISEASE_NAMES,
        "metadata": _load_model_meta(),
    }


@app.post("/reload-model")
def reload_model() -> dict[str, Any]:
    loaded = _load_latest_model()
    return {
        "reloaded": loaded,
        "model_loaded": _model is not None,
        "model_path": _active_model_path or MODEL_PATH,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(default=True),
    tta_runs: int = Query(default=1, ge=1, le=8),
    mc_runs: int = Query(default=5, ge=1, le=10),
    confidence_temperature: float = Query(default=1.0, gt=0.0, le=2.0),
    include_probabilities: bool = Query(default=True),
    include_bias_analysis: bool = Query(default=True),
) -> dict[str, Any]:
    model = _get_model()

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    model_input = prepare_inference_image(image)

    # 1. Deterministic probabilities with training=False — calibrated confidence
    base_probs = model(model_input, training=False).numpy()[0]

    # 2. TTA (reuse base_probs as the first pass)
    if tta_runs > 1:
        tta_preds = [base_probs]
        for _ in range(tta_runs - 1):
            augmented = _tta_augment(model_input)
            tta_preds.append(model(augmented, training=False).numpy()[0])
        probs = np.mean(np.stack(tta_preds, axis=0), axis=0)
    else:
        probs = base_probs

    # 3. Apply confidence temperature if specified
    if not np.isclose(confidence_temperature, 1.0):
        probs = _apply_confidence_temperature(probs, temperature=confidence_temperature)

    # 4. MC Dropout for per-class std (uncertainty display)
    mc_std = _mc_dropout_std(model, model_input, runs=mc_runs)

    top_idx = int(np.argmax(probs))
    top_label = LABEL_NAMES[top_idx]
    entropy = _predictive_entropy(probs)
    uncertainty_score = float(np.clip(entropy / np.log(len(LABEL_NAMES)), 0.0, 1.0))

    response: dict[str, Any] = {
        "predicted_label": top_label,
        "predicted_disease": DISEASE_NAMES[top_label],
        "confidence": float(probs[top_idx]),
        "tta_runs": int(tta_runs),
        "mc_runs": int(mc_runs),
        "confidence_temperature": float(confidence_temperature),
        "uncertainty": {
            "score": round(uncertainty_score, 4),
            "entropy": round(entropy, 4),
            "level": "high" if uncertainty_score > 0.70 else "moderate" if uncertainty_score > 0.45 else "low",
        },
    }

    # 5. Fitzpatrick bias analysis
    if include_bias_analysis:
        try:
            skin_tone_category = estimate_skin_tone_category(model_input)
            bias_report = generate_bias_report(
                skin_tone_category,
                float(probs[top_idx]),
                dataset_name="ham10000"
            )
            response["fitzpatrick_analysis"] = bias_report
        except Exception as exc:
            response["fitzpatrick_analysis_error"] = str(exc)

    # 6. Full probability breakdown
    if include_probabilities:
        response["probabilities"] = {
            LABEL_NAMES[i]: {
                "probability": float(probs[i]),
                "std": float(mc_std[i]),
                "disease": DISEASE_NAMES[LABEL_NAMES[i]],
            }
            for i in range(len(probs))
        }

    # 7. Grad-CAM explainability
    if explain:
        try:
            last_conv = _last_conv_layer or find_last_conv_layer(model)
            input_f32 = model_input.astype(np.float32)
            heatmap = make_gradcam_heatmap(input_f32, model, last_conv)
            response["gradcam_base64"] = _overlay_gradcam(input_f32[0], heatmap)
        except Exception as exc:
            response["gradcam_error"] = str(exc)

    return response
