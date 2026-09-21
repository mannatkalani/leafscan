"""
app.py
------
LeafScan — a Streamlit front end for the CNN trained in CNN_Project.ipynb
(New Plant Diseases Dataset, 38 classes, 128x128 input).

Run with:  streamlit run app.py

The app looks for a saved Keras model at one of MODEL_SEARCH_PATHS. If none
is found it still runs, in a clearly-labelled "no model loaded" state, so the
UI itself is always inspectable even before you've exported weights.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time

import numpy as np
import streamlit as st
from PIL import Image

from classes import CLASS_NAMES, get_disease_info, parse_class_name

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMG_SIZE = 128
MODEL_SEARCH_PATHS = [
    "model/plant_disease_model.h5",
    "model/plant_disease_model.keras",
    "plant_disease_model.h5",
    "plant_disease_model.keras",
]
HISTORY_PATH = "model/history.json"

# Static architecture summary — mirrors the Sequential CNN built in the
# notebook (Conv32 -> Conv64 -> Conv128 -> Dense128 -> Dense(num_classes)).
# Used as a fallback when no model is loaded, and to label the live model.
ARCHITECTURE = [
    ("Input", f"{IMG_SIZE} x {IMG_SIZE} x 3", "-"),
    ("Conv2D (32, 3x3, relu)", "126 x 126 x 32", "896"),
    ("MaxPooling2D (2x2)", "63 x 63 x 32", "0"),
    ("Conv2D (64, 3x3, relu)", "61 x 61 x 64", "18,496"),
    ("MaxPooling2D (2x2)", "30 x 30 x 64", "0"),
    ("Conv2D (128, 3x3, relu)", "28 x 28 x 128", "73,856"),
    ("MaxPooling2D (2x2)", "14 x 14 x 128", "0"),
    ("Flatten", "25,088", "0"),
    ("Dense (128, relu)", "128", "3,211,392"),
    ("Dropout (0.5)", "128", "0"),
    ("Dense (softmax)", f"{len(CLASS_NAMES)}", f"{128 * len(CLASS_NAMES) + len(CLASS_NAMES):,}"),
]
FALLBACK_TOTAL_PARAMS = 3_309_542

st.set_page_config(
    page_title="LeafScan | Plant Pathology AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling — deep-forest / lab-scanner theme (green, not the Streamlit default)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root{
        --bg:            #081A12;
        --panel:         #0F2B1D;
        --panel-2:       #123524;
        --line:          #1F4430;
        --leaf:          #3FA34D;
        --leaf-bright:   #63D678;
        --lime:          #C6FF6B;
        --amber:         #E3A857;
        --text:          #EAF3EA;
        --text-muted:    #8FB39D;
        --mono: 'JetBrains Mono', monospace;
        --display: 'Space Grotesk', sans-serif;
        --body: 'Inter', sans-serif;
    }

    html, body, [class*="css"]{ font-family: var(--body); color: var(--text); }
    .stApp{
        background:
            radial-gradient(circle at 12% -10%, rgba(63,163,77,0.16), transparent 45%),
            radial-gradient(circle at 90% 0%, rgba(198,255,107,0.07), transparent 40%),
            var(--bg);
    }
    #MainMenu, footer, header{ visibility: hidden; }
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #0A2216 0%, #071A12 100%);
        border-right: 1px solid var(--line);
    }
    .block-container{ padding-top: 2rem; max-width: 1180px; }

    /* ---------- Hero ---------- */
    .hero{
        display:flex; align-items:center; justify-content:space-between;
        gap: 1.5rem; flex-wrap:wrap;
        padding: 1.6rem 1.8rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(120deg, rgba(63,163,77,0.10), rgba(18,53,36,0.6));
        margin-bottom: 1.6rem;
    }
    .hero h1{
        font-family: var(--display); font-weight:700; font-size: 2.15rem;
        margin:0; letter-spacing:-0.01em; color: var(--text);
        line-height:1.15;
    }
    .hero h1 span{ color: var(--leaf-bright); }
    .hero p{ font-family: var(--body); color: var(--text-muted); margin:0.35rem 0 0 0; font-size:0.98rem; max-width:46ch; }
    .status-chip{
        font-family: var(--mono); font-size:0.78rem; letter-spacing:0.02em;
        padding: 0.45rem 0.85rem; border-radius: 999px; white-space:nowrap;
        border:1px solid; display:inline-flex; align-items:center; gap:0.45rem;
    }
    .status-chip.online{ color:#B9F6CA; border-color:#2E6B44; background:rgba(63,163,77,0.14); }
    .status-chip.offline{ color:#F3C089; border-color:#6B5230; background:rgba(227,168,87,0.12); }
    .status-chip .dot{ width:7px; height:7px; border-radius:50%; background:currentColor; box-shadow:0 0 8px currentColor; }

    /* ---------- Panel labels (scanner-readout style section headers) ---------- */
    .panel-label{
        font-family: var(--mono); font-size:0.72rem; letter-spacing:0.14em;
        color: var(--text-muted); text-transform:uppercase;
        border-bottom: 1px dashed var(--line); padding-bottom:0.5rem; margin-bottom:0.9rem;
    }

    /* ---------- Specimen frame ---------- */
    .specimen-frame{
        position:relative; border-radius:14px; overflow:hidden;
        border: 1px solid var(--line); background: var(--panel);
        min-height: 280px; display:flex; align-items:center; justify-content:center;
    }
    .specimen-frame img{ width:100%; display:block; }
    .specimen-frame.locked{ box-shadow: 0 0 0 1px var(--leaf), 0 0 24px rgba(63,163,77,0.35) inset; }
    .scan-sweep{
        position:absolute; left:0; right:0; height:3px;
        background: linear-gradient(90deg, transparent, var(--lime), transparent);
        box-shadow: 0 0 14px 3px rgba(198,255,107,0.7);
        animation: sweep 1.4s ease-in-out infinite;
    }
    @keyframes sweep{
        0%{ top:0%; opacity:0; } 10%{ opacity:1; } 90%{ opacity:1; } 100%{ top:100%; opacity:0; }
    }
    .empty-state{
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:0.5rem; color: var(--text-muted); padding: 3rem 1rem; text-align:center;
    }
    .empty-state .glyph{ font-size:2rem; opacity:0.7; }
    .empty-state .msg{ font-family: var(--mono); font-size:0.82rem; }

    /* ---------- Diagnosis card ---------- */
    .diag-card{ border:1px solid var(--line); border-radius:14px; background: var(--panel); padding:1.4rem 1.5rem; }
    .diag-top{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
    .diag-crop{ font-family: var(--mono); font-size:0.76rem; color: var(--text-muted); letter-spacing:0.08em; text-transform:uppercase; }
    .diag-condition{ font-family: var(--display); font-size:1.5rem; font-weight:700; margin:0.15rem 0 0 0; }
    .pill{ font-family: var(--mono); font-size:0.78rem; padding:0.35rem 0.75rem; border-radius:999px; border:1px solid; white-space:nowrap; }
    .pill.healthy{ color:#B9F6CA; border-color:#2E6B44; background:rgba(63,163,77,0.16); }
    .pill.diseased{ color:#F3C089; border-color:#6B5230; background:rgba(227,168,87,0.14); }
    .diag-cause{ color: var(--text-muted); font-size:0.92rem; margin: 0.9rem 0 1.1rem 0; }

    .meter-row{ margin-bottom:0.65rem; }
    .meter-label{ display:flex; justify-content:space-between; font-family: var(--mono); font-size:0.76rem; color:var(--text-muted); margin-bottom:0.3rem; }
    .meter-track{ height:8px; border-radius:5px; background: var(--panel-2); border:1px solid var(--line); overflow:hidden; }
    .meter-fill{ height:100%; border-radius:5px; background: linear-gradient(90deg, var(--leaf), var(--leaf-bright)); }

    .tips{ margin-top:1.1rem; padding-top:1rem; border-top:1px dashed var(--line); }
    .tips-label{ font-family: var(--mono); font-size:0.72rem; letter-spacing:0.1em; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.5rem; }
    .tips ul{ margin:0; padding-left:1.1rem; color: var(--text); font-size:0.92rem; }
    .tips li{ margin-bottom:0.3rem; }

    /* ---------- Stat chips (sidebar + insights) ---------- */
    .stat-row{ display:flex; justify-content:space-between; font-family: var(--mono); font-size:0.82rem; padding:0.4rem 0; border-bottom:1px solid var(--line); }
    .stat-row span:first-child{ color: var(--text-muted); }
    .stat-row span:last-child{ color: var(--text); }
    .chip-group{ display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.6rem; }
    .tech-chip{ font-family: var(--mono); font-size:0.72rem; color: var(--text-muted); border:1px solid var(--line); border-radius:6px; padding:0.25rem 0.55rem; }

    /* ---------- Architecture table ---------- */
    .arch-table{ width:100%; border-collapse:collapse; font-family: var(--mono); font-size:0.82rem; }
    .arch-table th{ text-align:left; color:var(--text-muted); font-weight:500; padding:0.5rem 0.4rem; border-bottom:1px solid var(--line); font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase; }
    .arch-table td{ padding:0.45rem 0.4rem; border-bottom:1px solid rgba(31,68,48,0.5); }
    .arch-table tr:last-child td{ border-bottom:none; font-weight:600; color: var(--leaf-bright); }

    /* ---------- Buttons / tabs / uploader (theme the default widgets) ---------- */
    .stButton > button{
        background: linear-gradient(120deg, var(--leaf), #2E8B41);
        color: #06170E; border:none; border-radius:10px; font-weight:600;
        padding:0.6rem 1.1rem; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover{ transform: translateY(-1px); box-shadow:0 6px 18px rgba(63,163,77,0.35); }
    .stButton > button:disabled{ background: var(--panel-2); color: var(--text-muted); }

    .stTabs [data-baseweb="tab-list"]{ gap: 1.5rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"]{ font-family: var(--mono); font-size:0.85rem; color: var(--text-muted); }
    .stTabs [aria-selected="true"]{ color: var(--leaf-bright) !important; }

    [data-testid="stFileUploaderDropzone"]{
        background: var(--panel); border:1px dashed var(--line); border-radius:12px;
    }

    .about-lead{ font-size:1.02rem; color: var(--text); line-height:1.6; }
    .about-lead b{ color: var(--leaf-bright); }
    .pipeline-step{ display:flex; gap:0.8rem; padding:0.6rem 0; border-bottom:1px dashed var(--line); }
    .pipeline-step .n{ font-family: var(--mono); color: var(--leaf-bright); font-size:0.85rem; width:1.6rem; }
    .pipeline-step .t{ font-size:0.92rem; color: var(--text); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the first available trained model, or return None."""
    try:
        from tensorflow.keras.models import load_model as keras_load_model
    except ImportError:
        return None, None, "TensorFlow is not installed in this environment."

    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                return keras_load_model(path), path, None
            except Exception as exc:  # noqa: BLE001 - surface any load error to the UI
                return None, None, f"Found a model at '{path}' but failed to load it: {exc}"

    return None, None, None


model, model_path, model_error = load_model()
model_online = model is not None


def preprocess(img: Image.Image) -> np.ndarray:
    resized = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(resized).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(img: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
    preds = model.predict(preprocess(img), verbose=0)[0]
    order = np.argsort(preds)[::-1][:top_k]
    return [(CLASS_NAMES[i], float(preds[i])) for i in order]


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def meter_html(label: str, pct: float) -> str:
    return f"""
    <div class="meter-row">
        <div class="meter-label"><span>{label}</span><span>{pct * 100:.1f}%</span></div>
        <div class="meter-track"><div class="meter-fill" style="width:{max(pct * 100, 2):.1f}%"></div></div>
    </div>
    """


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
            <span style="font-size:1.6rem;">🌿</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.25rem;">LeafScan</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.74rem; color:#8FB39D; margin-bottom:1.2rem;">
            Plant pathology, by CNN
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel-label">System Status</div>', unsafe_allow_html=True)
    if model_online:
        st.markdown('<span class="status-chip online"><span class="dot"></span>Model online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-chip offline"><span class="dot"></span>No model loaded</span>', unsafe_allow_html=True)

    total_params = f"{model.count_params():,}" if model_online else f"~{FALLBACK_TOTAL_PARAMS:,}"
    st.markdown(
        f"""
        <div style="margin-top:0.9rem;">
            <div class="stat-row"><span>Classes</span><span>{len(CLASS_NAMES)}</span></div>
            <div class="stat-row"><span>Input size</span><span>{IMG_SIZE}x{IMG_SIZE}</span></div>
            <div class="stat-row"><span>Parameters</span><span>{total_params}</span></div>
            <div class="stat-row"><span>Architecture</span><span>Sequential CNN</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel-label" style="margin-top:1.4rem;">Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chip-group">
            <span class="tech-chip">TensorFlow / Keras</span>
            <span class="tech-chip">OpenCV</span>
            <span class="tech-chip">NumPy</span>
            <span class="tech-chip">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_online:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("How to connect your model"):
            st.markdown(
                f"""
Save your trained model at the end of the notebook:

```python
model.save("model/plant_disease_model.h5")
```

Then drop the `model/` folder next to `app.py`. LeafScan checks:
{chr(10).join(f"- `{p}`" for p in MODEL_SEARCH_PATHS)}
                """
            )
            if model_error:
                st.caption(f"Last load error: {model_error}")

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
status_html = (
    '<span class="status-chip online"><span class="dot"></span>Model online</span>'
    if model_online
    else '<span class="status-chip offline"><span class="dot"></span>Demo mode — no model loaded</span>'
)
st.markdown(
    f"""
    <div class="hero">
        <div>
            <h1>Plant pathology, <span>read by a CNN.</span></h1>
            <p>Upload a leaf photo and a convolutional network trained on 38 crop-disease
            classes identifies the condition and suggests next steps.</p>
        </div>
        {status_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

tab_scan, tab_insights, tab_about = st.tabs(["Scan", "Insights", "About"])

# --------------------------------------------------------------------------
# Scan tab
# --------------------------------------------------------------------------
with tab_scan:
    left, right = st.columns([0.44, 0.56], gap="large")

    with left:
        st.markdown('<div class="panel-label">Specimen</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        frame_slot = st.empty()
        current_img = None

        if uploaded is not None:
            current_img = Image.open(uploaded)
            file_id = f"{uploaded.name}-{uploaded.size}"
            if file_id != st.session_state.last_file_id:
                st.session_state.last_file_id = file_id
                st.session_state.diagnosis = None
            frame_slot.markdown(
                f"""<div class="specimen-frame locked"><img src="{image_to_data_uri(current_img)}"/></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.session_state.last_file_id = None
            st.session_state.diagnosis = None
            frame_slot.markdown(
                """
                <div class="specimen-frame">
                    <div class="empty-state">
                        <div class="glyph">🍃</div>
                        <div class="msg">awaiting specimen — jpg or png</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        run_clicked = st.button(
            "Run diagnosis",
            use_container_width=True,
            disabled=(current_img is None or not model_online),
        )
        if not model_online:
            st.caption("Connect a model (see sidebar) to enable diagnosis.")

    with right:
        st.markdown('<div class="panel-label">Diagnostic readout</div>', unsafe_allow_html=True)

        if run_clicked and current_img is not None and model_online:
            frame_slot.markdown(
                f"""<div class="specimen-frame locked"><img src="{image_to_data_uri(current_img)}"/><div class="scan-sweep"></div></div>""",
                unsafe_allow_html=True,
            )
            progress = st.progress(0, text="Calibrating sensors...")
            steps = [
                "Calibrating sensors...",
                "Extracting leaf features...",
                "Cross-referencing pathology classes...",
                "Finalizing diagnosis...",
            ]
            for i, step_text in enumerate(steps):
                time.sleep(0.3)
                progress.progress(int((i + 1) / len(steps) * 100), text=step_text)

            top3 = run_inference(current_img, top_k=3)
            progress.empty()
            frame_slot.markdown(
                f"""<div class="specimen-frame locked"><img src="{image_to_data_uri(current_img)}"/></div>""",
                unsafe_allow_html=True,
            )

            best_class, best_conf = top3[0]
            st.session_state.diagnosis = {"top3": top3, "class": best_class, "conf": best_conf}

        diag = st.session_state.diagnosis
        if diag is None:
            st.markdown(
                """
                <div class="diag-card">
                    <div class="empty-state" style="padding:2.2rem 0.5rem;">
                        <div class="glyph">📟</div>
                        <div class="msg">no readout yet — upload a leaf and run diagnosis</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            parsed = parse_class_name(diag["class"])
            info = get_disease_info(diag["class"])
            pill_class = "healthy" if parsed.is_healthy else "diseased"
            pill_text = "Healthy" if parsed.is_healthy else "Disease detected"

            meters = "".join(meter_html(parse_class_name(c).condition, p) for c, p in diag["top3"])

            st.markdown(
                f"""
                <div class="diag-card">
                    <div class="diag-top">
                        <div>
                            <div class="diag-crop">{parsed.crop}</div>
                            <div class="diag-condition">{parsed.condition}</div>
                        </div>
                        <span class="pill {pill_class}">{pill_text}</span>
                    </div>
                    <div class="diag-cause">{info['cause']}</div>
                    {meters}
                    <div class="tips">
                        <div class="tips-label">Recommended actions</div>
                        <ul>{"".join(f"<li>{tip}</li>" for tip in info['tips'])}</ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if parsed.is_healthy:
                st.balloons()

# --------------------------------------------------------------------------
# Insights tab  (kept to a single optional chart, per design brief)
# --------------------------------------------------------------------------
with tab_insights:
    col1, col2 = st.columns([0.55, 0.45], gap="large")

    with col1:
        st.markdown('<div class="panel-label">Model architecture</div>', unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td>{layer}</td><td>{shape}</td><td>{params}</td></tr>"
            for layer, shape, params in ARCHITECTURE
        )
        st.markdown(
            f"""
            <table class="arch-table">
                <tr><th>Layer</th><th>Output shape</th><th>Params</th></tr>
                {rows}
                <tr><td>Total</td><td>—</td><td>{total_params}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="panel-label">Training summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stat-row"><span>Optimizer</span><span>Adam</span></div>
            <div class="stat-row"><span>Loss</span><span>Categorical cross-entropy</span></div>
            <div class="stat-row"><span>Augmentation</span><span>Rotation, zoom, h-flip</span></div>
            <div class="stat-row"><span>Batch size</span><span>32</span></div>
            """,
            unsafe_allow_html=True,
        )

        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH) as f:
                    hist = json.load(f)
                import pandas as pd

                epochs = range(1, len(hist.get("accuracy", [])) + 1)
                df = pd.DataFrame(
                    {
                        "train_acc": hist.get("accuracy", []),
                        "val_acc": hist.get("val_accuracy", []),
                    },
                    index=list(epochs),
                )
                st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
                st.line_chart(df)
            except Exception:
                st.caption("history.json found but couldn't be parsed.")
        else:
            st.caption(
                "Export training history (accuracy per epoch) to model/history.json "
                "to see the learning curve here."
            )

# --------------------------------------------------------------------------
# About tab
# --------------------------------------------------------------------------
with tab_about:
    col1, col2 = st.columns([0.6, 0.4], gap="large")

    with col1:
        st.markdown('<div class="panel-label">Project</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="about-lead">
            LeafScan is an end-to-end plant disease classifier: a convolutional
            neural network trained from scratch on the <b>New Plant Diseases
            Dataset</b> (an augmented version of PlantVillage), served through
            this interface. Upload a photo of a leaf and the model identifies
            the crop, flags whether it's healthy, and — when it isn't —
            names the likely condition with practical next steps.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="panel-label" style="margin-top:1.6rem;">Pipeline</div>', unsafe_allow_html=True)
        steps = [
            "Download & inspect the dataset (38 classes, 14 crop species) via the Kaggle API",
            "Augment training data — rotation, zoom, horizontal flip — to reduce overfitting",
            "Train a 3-block Conv2D/MaxPooling CNN with dropout, Adam optimizer",
            "Export the trained model and serve predictions through this Streamlit app",
        ]
        st.markdown(
            "".join(
                f'<div class="pipeline-step"><div class="n">{i+1:02d}</div><div class="t">{s}</div></div>'
                for i, s in enumerate(steps)
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="panel-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="stat-row"><span>Source</span><span>Kaggle</span></div>
            <div class="stat-row"><span>Classes</span><span>38</span></div>
            <div class="stat-row"><span>Crop species</span><span>14</span></div>
            <div class="stat-row"><span>Image size</span><span>128x128</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="panel-label" style="margin-top:1.6rem;">Built by</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="about-lead" style="font-size:0.92rem;">
            Add your name, portfolio link, and GitHub repo here —
            this panel is designed to be the first thing a recruiter reads.
            </div>
            """,
            unsafe_allow_html=True,
        )
