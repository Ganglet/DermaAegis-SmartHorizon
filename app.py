"""
DermAegis AI — Streamlit interface for HuggingFace Spaces deployment.
Run locally:  streamlit run app.py
"""

import base64
import io
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.preprocessing import (
    DISEASE_NAMES,
    LABEL_NAMES,
    find_last_conv_layer,
    load_trained_model,
    make_gradcam_heatmap,
    prepare_inference_image,
)
from utils.fitzpatrick_bias import estimate_skin_tone_category, generate_bias_report

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DermAegis AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — mirrors React design
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;700;800&family=IBM+Plex+Mono:wght@400;700&display=swap');

:root {
  --bg: #f4efe4;
  --ink: #1f2a21;
  --ink-soft: #425145;
  --accent: #da4f38;
  --accent-2: #f3a949;
  --card: rgba(255,255,255,0.72);
  --line: rgba(31,42,33,0.14);
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 10% 20%, #fef3c0 0%, transparent 30%),
    radial-gradient(circle at 90% 10%, #ffd8b8 0%, transparent 36%),
    linear-gradient(155deg, #f0ecdf 0%, #f7f4ea 40%, #fef8f0 100%) !important;
  font-family: 'Bricolage Grotesque', sans-serif !important;
  color: #1f2a21 !important;
}

[data-testid="stHeader"] { background: transparent !important; }

.hero-title {
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 800;
  letter-spacing: -0.01em;
  margin: 0 0 4px;
  color: #1f2a21;
}

.hero-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #425145;
  margin: 0 0 10px;
}

.hero-desc {
  color: #425145;
  max-width: 760px;
  margin-bottom: 20px;
}

.status-card {
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(31,42,33,0.14);
  border-radius: 12px;
  padding: 10px 14px;
}

.status-kicker {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  color: #425145;
  margin: 0;
  text-transform: uppercase;
}

.status-ok { color: #1f7447; font-weight: 700; margin: 4px 0 0; font-size: 0.95rem; }
.status-bad { color: #8f2213; font-weight: 700; margin: 4px 0 0; font-size: 0.95rem; }
.status-path { color: #425145; font-size: 0.78rem; margin: 4px 0 0; word-break: break-all; }

.panel {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(31,42,33,0.14);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 12px 30px rgba(29,40,32,0.14);
  backdrop-filter: blur(8px);
  margin-top: 16px;
}

.card {
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(31,42,33,0.14);
  border-radius: 14px;
  padding: 14px;
}

.card-title {
  font-size: 1rem;
  font-weight: 700;
  margin: 0 0 12px;
  color: #1f2a21;
}

.pred-label {
  font-size: 1.2rem;
  font-weight: 800;
  margin: 0 0 6px;
}

.pred-meta {
  color: #425145;
  margin: 0 0 4px;
  font-size: 0.9rem;
}

.pred-band {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  color: #6d2d00;
  background: rgba(255,196,135,0.35);
  margin-bottom: 12px;
}

.prob-row {
  margin-bottom: 8px;
}

.prob-head {
  display: flex;
  justify-content: space-between;
  font-size: 0.84rem;
  margin-bottom: 3px;
}

.bar-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(31,42,33,0.12);
  overflow: hidden;
}

.bar-fill-inner {
  height: 100%;
  background: linear-gradient(90deg, #d44d33, #efaa4f);
  border-radius: 999px;
}

.u-box { border-radius: 12px; padding: 10px 12px; margin: 10px 0; border: 1px solid transparent; }
.u-low  { background: rgba(31,116,71,0.08);  border-color: rgba(31,116,71,0.2); }
.u-mod  { background: rgba(243,169,73,0.12); border-color: rgba(243,169,73,0.3); }
.u-high { background: rgba(143,34,19,0.08);  border-color: rgba(143,34,19,0.25); }

.u-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #425145;
  margin: 0 0 6px;
}

.u-bar-track {
  height: 6px;
  border-radius: 999px;
  background: rgba(31,42,33,0.12);
  overflow: hidden;
  margin-bottom: 6px;
}

.u-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #1f7447, #f3a949, #da4f38);
}

.u-meta    { font-size: 0.82rem; color: #425145; margin: 0; }
.u-warning { font-size: 0.82rem; font-weight: 600; color: #8f2213; margin: 6px 0 0; }

.notice {
  margin-top: 16px;
  border: 1px solid rgba(150,56,22,0.2);
  background: rgba(255,240,232,0.85);
  border-radius: 12px;
  padding: 10px 14px;
  color: #6b2d10;
  font-size: 0.88rem;
}

/* Override Streamlit button */
div.stButton > button {
  background: linear-gradient(120deg, #da4f38, #f3a949) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  padding: 10px 20px !important;
  width: 100% !important;
  transition: filter 180ms ease !important;
}

div.stButton > button:hover { filter: brightness(1.05) !important; }

[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed #425145 !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,0.6) !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached so it runs once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    preferred = os.path.join(PROJECT_ROOT, "models", "cnn_model.keras")
    if os.path.isfile(preferred):
        path = preferred
    else:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if not os.path.isdir(models_dir):
            return None, None, None
        backbone_priority = {"efficientnetb3": 4, "efficientnet": 3, "resnet": 2, "mobilenet": 1}
        candidates = [
            os.path.join(models_dir, f)
            for f in os.listdir(models_dir)
            if f.endswith(".best.keras")
        ]
        if not candidates:
            return None, None, None

        def _score(p):
            name = os.path.basename(p).lower()
            priority = next((v for k, v in backbone_priority.items() if k in name), 0)
            return priority, os.path.getmtime(p)

        path = max(candidates, key=_score)

    model = load_trained_model(path)
    last_conv = find_last_conv_layer(model)
    return model, last_conv, path


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def mc_dropout_std(model, model_input: np.ndarray, runs: int = 5) -> np.ndarray:
    replicated = np.repeat(model_input, runs, axis=0)
    all_preds = model(replicated, training=True).numpy()
    return all_preds.std(axis=0)


def predictive_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def overlay_gradcam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = image_np.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    merged = np.clip((1 - alpha) * img + alpha * colored, 0, 1)
    return np.uint8(merged * 255)


def run_predict(model, last_conv, pil_image: Image.Image):
    model_input = prepare_inference_image(pil_image)
    probs = model(model_input, training=False).numpy()[0]
    mc_std = mc_dropout_std(model, model_input)

    top_idx = int(np.argmax(probs))
    top_label = LABEL_NAMES[top_idx]
    confidence = float(probs[top_idx])

    entropy = predictive_entropy(probs)
    uncertainty_score = float(np.clip(entropy / np.log(len(LABEL_NAMES)), 0.0, 1.0))
    uncertainty_level = (
        "High" if uncertainty_score > 0.70 else
        "Moderate" if uncertainty_score > 0.45 else
        "Low"
    )

    sorted_probs = sorted(
        [(LABEL_NAMES[i], DISEASE_NAMES[LABEL_NAMES[i]], float(probs[i]), float(mc_std[i]))
         for i in range(len(probs))],
        key=lambda x: -x[2],
    )

    bias_info = None
    try:
        skin_tone_category = estimate_skin_tone_category(model_input)
        bias_info = generate_bias_report(skin_tone_category, confidence, dataset_name="ham10000")
    except Exception:
        pass

    gradcam_img = None
    try:
        heatmap = make_gradcam_heatmap(model_input.astype(np.float32), model, last_conv)
        gradcam_img = Image.fromarray(overlay_gradcam(model_input[0], heatmap))
    except Exception:
        pass

    return {
        "disease": DISEASE_NAMES[top_label],
        "label": top_label,
        "confidence": confidence,
        "uncertainty_score": uncertainty_score,
        "uncertainty_level": uncertainty_level,
        "sorted_probs": sorted_probs,
        "bias_info": bias_info,
        "gradcam_img": gradcam_img,
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def bar_html(pct: float, color_grad="linear-gradient(90deg, #d44d33, #efaa4f)") -> str:
    return f"""
<div class="bar-track">
  <div class="bar-fill-inner" style="width:{pct*100:.1f}%;background:{color_grad}"></div>
</div>"""


def uncertainty_html(score: float, level: str) -> str:
    cls = {"Low": "u-low", "Moderate": "u-mod", "High": "u-high"}.get(level, "u-mod")
    warning = '<p class="u-warning">High uncertainty — specialist review strongly recommended.</p>' if level == "High" else ""
    return f"""
<div class="u-box {cls}">
  <p class="u-label">Model Uncertainty</p>
  <div class="u-bar-track">
    <div class="u-bar-fill" style="width:{score*100:.1f}%"></div>
  </div>
  <p class="u-meta">{score*100:.1f}% — {level} uncertainty</p>
  {warning}
</div>"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

model, last_conv, model_path = load_model()
model_loaded = model is not None

# Hero
st.markdown("""
<div class="hero-title">DERMAEGIS AI</div>
<div class="hero-sub">Skin Lesion Intelligence Workspace</div>
<div class="hero-desc">Upload a dermoscopic image, run your trained model, inspect probabilities, and verify attention focus with Grad-CAM.</div>
""", unsafe_allow_html=True)

# Status strip
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.markdown(f"""
<div class="status-card">
  <p class="status-kicker">Model</p>
  <p class="{'status-ok' if model_loaded else 'status-bad'}">{'Loaded' if model_loaded else 'Not loaded'}</p>
</div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
<div class="status-card">
  <p class="status-kicker">Backend</p>
  <p class="status-ok">Ready</p>
</div>""", unsafe_allow_html=True)

with c3:
    path_display = model_path if model_path else "No checkpoint found in models/"
    st.markdown(f"""
<div class="status-card">
  <p class="status-kicker">Model Path</p>
  <p class="status-path">{path_display}</p>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)

# Upload + camera
up_col, cam_col = st.columns([3, 2])
with up_col:
    uploaded = st.file_uploader("Choose image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
with cam_col:
    camera_shot = st.camera_input("Or use camera")

# Resolve image source
pil_image = None
if camera_shot is not None:
    pil_image = Image.open(camera_shot).convert("RGB")
elif uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

run_btn = st.button("Run Prediction", disabled=(pil_image is None or not model_loaded))

if not model_loaded:
    st.error("No model checkpoint found. Place a trained `.keras` file in `models/` and restart.")

# Results
left, right = st.columns(2)

with left:
    st.markdown('<div class="card"><div class="card-title">Input</div>', unsafe_allow_html=True)
    if pil_image:
        st.image(pil_image, use_container_width=True)
    else:
        st.markdown("<p style='color:#425145'>No image selected.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">Prediction</div>', unsafe_allow_html=True)

    if run_btn and pil_image and model_loaded:
        with st.spinner("Analyzing…"):
            res = run_predict(model, last_conv, pil_image)
            st.session_state["result"] = res

    res = st.session_state.get("result")

    if res:
        conf_pct = res["confidence"] * 100
        band = (
            "High confidence" if res["confidence"] >= 0.70 else
            "Moderate confidence" if res["confidence"] >= 0.45 else
            "Low confidence"
        )
        st.markdown(f"""
<p class="pred-label">{res['disease']}</p>
<p class="pred-meta">Label: {res['label']}</p>
<p class="pred-meta">Confidence: {conf_pct:.1f}%</p>
<span class="pred-band">{band}</span>
{uncertainty_html(res['uncertainty_score'], res['uncertainty_level'])}
""", unsafe_allow_html=True)

        # Probability bars
        for label, disease, prob, std in res["sorted_probs"]:
            pct_str = f"{prob*100:.1f}%"
            std_str = f" ±{std*100:.1f}%" if std else ""
            st.markdown(f"""
<div class="prob-row">
  <div class="prob-head"><span>{disease}</span><span>{pct_str}{std_str}</span></div>
  {bar_html(prob)}
</div>""", unsafe_allow_html=True)

        # Fitzpatrick bias
        if res.get("bias_info"):
            b = res["bias_info"]
            with st.expander("Fitzpatrick Skin Tone Analysis"):
                st.markdown(f"""
**Detected:** {b['skin_tone_name']}
**Training representation:** {b['training_representation']} of HAM10000
**Reliability:** {b['reliability_level']}
**Adjusted confidence:** {b['adjusted_confidence']}
""")
                if b.get("bias_warning"):
                    st.warning(b["bias_warning"])
    else:
        st.markdown("<p style='color:#425145'>Run inference to view results.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Grad-CAM full-width
st.markdown('<div class="card" style="margin-top:12px"><div class="card-title">Grad-CAM Explainability</div>', unsafe_allow_html=True)
res = st.session_state.get("result")
if res and res.get("gradcam_img"):
    st.image(res["gradcam_img"], use_container_width=True)
elif res:
    st.markdown("<p style='color:#425145'>Grad-CAM unavailable for this image.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color:#425145'>Heatmap will appear after prediction.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
  <strong>Clinical Safety Notice:</strong> This tool is for research and educational purposes only.
  It is not a medical device. Final diagnosis must always be performed by a qualified dermatologist.
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
