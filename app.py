"""
DermAegis AI — Streamlit interface for HuggingFace Spaces deployment.
Run locally:  streamlit run app.py
"""

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
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..60,400;12..60,700;12..60,800&family=IBM+Plex+Mono:wght@400;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:
    radial-gradient(circle at 10% 20%, #fef3c0 0%, transparent 30%),
    radial-gradient(circle at 90% 10%, #ffd8b8 0%, transparent 36%),
    linear-gradient(155deg, #f0ecdf 0%, #f7f4ea 40%, #fef8f0 100%) !important;
  font-family: 'Bricolage Grotesque', sans-serif !important;
}

[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer         { visibility: hidden; }

/* ── typography ── */
.da-title {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.01em;
  color: #1f2a21;
  margin: 0 0 2px;
  line-height: 1.05;
}
.da-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #425145;
  margin: 0 0 8px;
}
.da-desc { color: #425145; margin: 0 0 20px; font-size: 0.95rem; }

/* ── status cards ── */
.sc {
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(31,42,33,0.14);
  border-radius: 12px;
  padding: 10px 14px;
  height: 100%;
}
.sc-kicker {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #425145;
  margin: 0 0 4px;
}
.sc-val-ok  { color: #1f7447; font-weight: 700; font-size: 0.95rem; margin: 0; }
.sc-val-bad { color: #8f2213; font-weight: 700; font-size: 0.95rem; margin: 0; }
.sc-path    { color: #425145; font-size: 0.76rem; margin: 0; word-break: break-all; line-height: 1.4; }

/* ── section label ── */
.sec-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #425145;
  margin: 0 0 8px;
  font-weight: 700;
}

/* ── prediction card ── */
.pred-name {
  font-size: 1.25rem;
  font-weight: 800;
  color: #1f2a21;
  margin: 0 0 4px;
}
.pred-meta { color: #425145; font-size: 0.88rem; margin: 0 0 4px; }
.pred-band {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  color: #6d2d00;
  background: rgba(255,196,135,0.38);
  margin-bottom: 14px;
}

/* ── uncertainty ── */
.u-box { border-radius: 10px; padding: 10px 12px; margin: 0 0 14px; border: 1px solid transparent; }
.u-low  { background: rgba(31,116,71,0.08);  border-color: rgba(31,116,71,0.22); }
.u-mod  { background: rgba(243,169,73,0.12); border-color: rgba(243,169,73,0.32); }
.u-high { background: rgba(143,34,19,0.08);  border-color: rgba(143,34,19,0.25); }
.u-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: #425145; margin: 0 0 6px;
}
.u-track { height: 6px; border-radius: 999px; background: rgba(31,42,33,0.12); overflow: hidden; margin-bottom: 5px; }
.u-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #1f7447, #f3a949, #da4f38); }
.u-meta    { font-size: 0.81rem; color: #425145; margin: 0; }
.u-warning { font-size: 0.81rem; font-weight: 600; color: #8f2213; margin: 6px 0 0; }

/* ── probability bars ── */
.prob-row { margin-bottom: 9px; }
.prob-head { display: flex; justify-content: space-between; font-size: 0.82rem; color: #1f2a21; margin-bottom: 3px; }
.bar-track { height: 7px; border-radius: 999px; background: rgba(31,42,33,0.11); overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #d44d33, #efaa4f); }

/* ── notice ── */
.notice {
  border: 1px solid rgba(150,56,22,0.2);
  background: rgba(255,240,232,0.85);
  border-radius: 12px;
  padding: 10px 14px;
  color: #6b2d10;
  font-size: 0.86rem;
  margin-top: 18px;
}

/* ── streamlit widget overrides ── */
div.stButton > button {
  background: linear-gradient(120deg, #da4f38, #f3a949) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  padding: 12px 24px !important;
  width: 100% !important;
  letter-spacing: 0.01em !important;
}
div.stButton > button:hover   { filter: brightness(1.06) !important; }
div.stButton > button:disabled { opacity: 0.55 !important; }

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.65) !important;
  border: 1.5px dashed rgba(66,81,69,0.5) !important;
  border-radius: 12px !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.55) !important;
  border-radius: 10px !important;
  gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: 8px !important;
  font-weight: 600 !important;
}

section[data-testid="stCameraInputToolbar"] { display: none !important; }

/* image output — no extra borders */
[data-testid="stImage"] img {
  border-radius: 10px !important;
  border: 1px solid rgba(31,42,33,0.12) !important;
  background: #fff;
}

/* divider */
hr { border-color: rgba(31,42,33,0.1) !important; }

/* expander */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.6) !important;
  border: 1px solid rgba(31,42,33,0.12) !important;
  border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading
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
# Inference
# ---------------------------------------------------------------------------

def mc_dropout_std(model, model_input: np.ndarray, runs: int = 5) -> np.ndarray:
    replicated = np.repeat(model_input, runs, axis=0)
    return model(replicated, training=True).numpy().std(axis=0)


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
    return np.uint8(np.clip((1 - alpha) * img + alpha * colored, 0, 1) * 255)


def run_predict(model, last_conv, pil_image: Image.Image) -> dict:
    model_input = prepare_inference_image(pil_image)
    probs = model(model_input, training=False).numpy()[0]
    mc_std = mc_dropout_std(model, model_input)

    top_idx = int(np.argmax(probs))
    top_label = LABEL_NAMES[top_idx]
    confidence = float(probs[top_idx])

    entropy = predictive_entropy(probs)
    u_score = float(np.clip(entropy / np.log(len(LABEL_NAMES)), 0.0, 1.0))
    u_level = "High" if u_score > 0.70 else "Moderate" if u_score > 0.45 else "Low"

    sorted_probs = sorted(
        [(LABEL_NAMES[i], DISEASE_NAMES[LABEL_NAMES[i]], float(probs[i]), float(mc_std[i]))
         for i in range(len(probs))],
        key=lambda x: -x[2],
    )

    bias_info = None
    try:
        bias_info = generate_bias_report(
            estimate_skin_tone_category(model_input), confidence, dataset_name="ham10000"
        )
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
        "u_score": u_score,
        "u_level": u_level,
        "sorted_probs": sorted_probs,
        "bias_info": bias_info,
        "gradcam_img": gradcam_img,
    }


# ---------------------------------------------------------------------------
# HTML render helpers
# ---------------------------------------------------------------------------

def _bar(pct: float, grad="linear-gradient(90deg,#d44d33,#efaa4f)") -> str:
    return (
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct*100:.1f}%;background:{grad}"></div>'
        f'</div>'
    )

def _uncertainty_block(score: float, level: str) -> str:
    cls = {"Low": "u-low", "Moderate": "u-mod", "High": "u-high"}.get(level, "u-mod")
    warn = f'<p class="u-warning">High uncertainty — specialist review strongly recommended.</p>' if level == "High" else ""
    return f"""
<div class="u-box {cls}">
  <p class="u-label">Model Uncertainty</p>
  <div class="u-track"><div class="u-fill" style="width:{score*100:.1f}%"></div></div>
  <p class="u-meta">{score*100:.1f}% — {level} uncertainty</p>
  {warn}
</div>"""


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

model, last_conv, model_path = load_model()
model_loaded = model is not None

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="da-title">DERMAEGIS AI</p>
<p class="da-sub">Skin Lesion Intelligence Workspace</p>
<p class="da-desc">Upload a dermoscopic image, run your trained model, inspect probabilities, and verify attention focus with Grad-CAM.</p>
""", unsafe_allow_html=True)

# ── Status strip ────────────────────────────────────────────────────────────
s1, s2, s3 = st.columns([1, 1, 3])
with s1:
    val_cls = "sc-val-ok" if model_loaded else "sc-val-bad"
    val_txt = "Loaded" if model_loaded else "Not loaded"
    st.markdown(f'<div class="sc"><p class="sc-kicker">Model</p><p class="{val_cls}">{val_txt}</p></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="sc"><p class="sc-kicker">Backend</p><p class="sc-val-ok">Ready</p></div>', unsafe_allow_html=True)
with s3:
    path_txt = model_path or "No checkpoint found in models/"
    st.markdown(f'<div class="sc"><p class="sc-kicker">Model Path</p><p class="sc-path">{path_txt}</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not model_loaded:
    st.error("No model checkpoint found. Place a trained `.keras` file in `models/` and restart.")

# ── Input panel ─────────────────────────────────────────────────────────────
st.markdown('<p class="sec-label">Image Input</p>', unsafe_allow_html=True)

tab_upload, tab_camera = st.tabs(["Upload Image", "Use Camera"])
pil_image = None

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a dermoscopic image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")

with tab_camera:
    camera_shot = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_shot:
        pil_image = Image.open(camera_shot).convert("RGB")

st.markdown("<br>", unsafe_allow_html=True)

# ── Run button ───────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 3])
with btn_col:
    run_btn = st.button(
        "Analyzing…" if st.session_state.get("running") else "Run Prediction",
        disabled=(pil_image is None or not model_loaded),
    )

if run_btn and pil_image and model_loaded:
    with st.spinner("Running inference…"):
        st.session_state["result"] = run_predict(model, last_conv, pil_image)
    st.session_state["result_image"] = pil_image

st.markdown("<hr>", unsafe_allow_html=True)

# ── Results grid ─────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="medium")
res = st.session_state.get("result")
res_img = st.session_state.get("result_image")

with left:
    st.markdown('<p class="sec-label">Input</p>', unsafe_allow_html=True)
    if res_img:
        st.image(res_img, use_container_width=True)
    elif pil_image:
        st.image(pil_image, use_container_width=True)
    else:
        st.markdown("<p style='color:#425145;font-size:0.9rem'>No image selected.</p>", unsafe_allow_html=True)

with right:
    st.markdown('<p class="sec-label">Prediction</p>', unsafe_allow_html=True)
    if res:
        conf_pct = res["confidence"] * 100
        band = (
            "High confidence" if res["confidence"] >= 0.70 else
            "Moderate confidence" if res["confidence"] >= 0.45 else
            "Low confidence"
        )
        st.markdown(f"""
<p class="pred-name">{res['disease']}</p>
<p class="pred-meta">Label: <strong>{res['label']}</strong></p>
<p class="pred-meta">Confidence: <strong>{conf_pct:.1f}%</strong></p>
<span class="pred-band">{band}</span>
{_uncertainty_block(res['u_score'], res['u_level'])}
""", unsafe_allow_html=True)

        # Probability bars
        html_bars = ""
        for label, disease, prob, std in res["sorted_probs"]:
            std_str = f" ±{std*100:.1f}%" if std else ""
            html_bars += f"""
<div class="prob-row">
  <div class="prob-head"><span>{disease}</span><span>{prob*100:.1f}%{std_str}</span></div>
  {_bar(prob)}
</div>"""
        st.markdown(html_bars, unsafe_allow_html=True)

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
        st.markdown("<p style='color:#425145;font-size:0.9rem'>Run inference to view results.</p>", unsafe_allow_html=True)

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="sec-label">Grad-CAM Explainability</p>', unsafe_allow_html=True)

if res and res.get("gradcam_img"):
    gcol, _ = st.columns([2, 1])
    with gcol:
        st.image(res["gradcam_img"], use_container_width=True)
elif res:
    st.markdown("<p style='color:#425145;font-size:0.9rem'>Grad-CAM unavailable for this image.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color:#425145;font-size:0.9rem'>Heatmap will appear after prediction.</p>", unsafe_allow_html=True)

# ── Notice ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
  <strong>Clinical Safety Notice:</strong> This tool is for research and educational purposes only.
  It is not a medical device. Final diagnosis must always be performed by a qualified dermatologist.
</div>
""", unsafe_allow_html=True)
