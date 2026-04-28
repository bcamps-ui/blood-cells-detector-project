"""
app.py — Streamlit web app for the Blood Cell Detector.

Upload a peripheral blood smear image (or pick a built-in sample) and the
YOLO26 model will detect and classify cells into 7 classes:
RBC, Platelets, Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil.
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
SAMPLE_DIR = ROOT / "test_images"

# ── class colours (BGR for cv2, converted to RGB for display) ────────────
WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# Colours in RGB for both annotation and UI badges
CLASS_COLORS_RGB = {
    "RBC":        (220, 60, 60),
    "Platelets":  (50, 190, 80),
    "Neutrophil": (50, 120, 255),
    "Lymphocyte": (50, 120, 255),
    "Monocyte":   (50, 120, 255),
    "Eosinophil": (50, 120, 255),
    "Basophil":   (50, 120, 255),
}

# Emoji for each class (used in the results panel)
CLASS_EMOJI = {
    "RBC":        "🔴",
    "Platelets":  "🟢",
    "Neutrophil": "🔵",
    "Lymphocyte": "🔵",
    "Monocyte":   "🔵",
    "Eosinophil": "🔵",
    "Basophil":   "🔵",
}


# ── page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── inject custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero ─────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(230, 57, 70, 0.25);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(230,57,70,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(90deg, #E63946, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}
.hero p {
    color: #b0b8c8;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0;
    position: relative;
}
.hero .badge {
    display: inline-block;
    background: rgba(230,57,70,0.15);
    color: #E63946;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(230,57,70,0.3);
    letter-spacing: 0.5px;
}

/* ── Cards ────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #1A1D23, #22252B);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.metric-card .label {
    font-size: 0.8rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #fafafa;
}

/* ── Results table ────────────────────────────────────────── */
.cell-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.cell-table th {
    background: #1A1D23;
    color: #8892a4;
    padding: 0.75rem 1rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.cell-table td {
    padding: 0.65rem 1rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    color: #e0e0e0;
    font-size: 0.95rem;
}
.cell-table tr:nth-child(even) td {
    background: rgba(255,255,255,0.02);
}
.cell-table .dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Disclaimer ───────────────────────────────────────────── */
.disclaimer {
    background: rgba(230, 57, 70, 0.08);
    border-left: 3px solid #E63946;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #c0c0c0;
    margin-top: 1rem;
}

/* ── Sample-image buttons ─────────────────────────────────── */
.sample-btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

/* ── Sidebar polish ───────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141a 0%, #0E1117 100%);
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06);
}

/* ── Upload area ──────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(230,57,70,0.3);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(230,57,70,0.6);
}
</style>
""", unsafe_allow_html=True)


# ── model loading (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLO model once and cache it across reruns."""
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    # Warm-up run to compile / initialise lazy layers
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model.predict(dummy, conf=0.5, imgsz=64, verbose=False)
    return model


# ── annotation drawing ───────────────────────────────────────────────────
def annotate_image(
    img_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confs: np.ndarray,
    names: dict,
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an RGB image and return the annotated copy."""
    out = img_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(img_rgb.shape[1], img_rgb.shape[0]) / 1600)
    thickness = max(1, line_width)

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, classes, confs):
        name = names[int(cls_id)]
        color = CLASS_COLORS_RGB.get(name, (200, 200, 200))
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Draw box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if show_labels or show_conf:
            parts = []
            if show_labels:
                parts.append(name)
            if show_conf:
                parts.append(f"{conf:.2f}")
            label = " ".join(parts)

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            # Label background
            label_y = y1 - 4
            if label_y - th - 4 < 0:
                label_y = y2 + th + 6
            cv2.rectangle(
                out,
                (x1, label_y - th - 4),
                (x1 + tw + 6, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                out, label, (x1 + 3, label_y - 2),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return out


# ── run inference ─────────────────────────────────────────────────────────
def run_detection(
    model,
    img_rgb: np.ndarray,
    conf: float,
    iou: float,
) -> dict:
    """Run YOLO prediction and return structured results."""
    results = model.predict(
        img_rgb,
        conf=conf,
        iou=iou,
        imgsz=640,
        device="cpu",
        verbose=False,
        max_det=300,
    )
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    names = r.names
    counts = Counter(names[int(c)] for c in cls)
    return {
        "boxes": boxes,
        "classes": cls,
        "confs": confs,
        "names": names,
        "counts": counts,
        "total": len(boxes),
    }


# ── helper: list sample images ───────────────────────────────────────────
def get_sample_images() -> list[Path]:
    if SAMPLE_DIR.exists():
        return sorted(
            p for p in SAMPLE_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    return []


# ── helper: image → downloadable bytes ───────────────────────────────────
def image_to_bytes(img_rgb: np.ndarray, fmt: str = "JPEG") -> bytes:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=92)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown("---")

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence for a detection to be shown. Lower → more detections (including faint cells); higher → fewer false positives.",
    )
    iou_threshold = st.slider(
        "IoU threshold (NMS)",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression. Higher → allows more overlapping boxes.",
    )

    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    show_labels = st.toggle("Show class labels", value=True)
    show_conf = st.toggle("Show confidence scores", value=True)

    st.markdown("---")
    st.markdown("### 📋 Class Legend")
    legend_md = """
| Color | Class |
|-------|-------|
| 🔴 | RBC |
| 🟢 | Platelets |
| 🔵 | Neutrophil |
| 🔵 | Lymphocyte |
| 🔵 | Monocyte |
| 🔵 | Eosinophil |
| 🔵 | Basophil |
"""
    st.markdown(legend_md)

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Research use only.</strong> '
        "This model has not been validated as a medical device. "
        "Do not use its outputs to inform diagnosis or treatment.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🧬 YOLO26 · 7 CLASSES · 640 × 640</div>
    <h1>🔬 Blood Cell Detector</h1>
    <p>
        Upload a peripheral blood smear image and the AI model will detect
        and classify <strong>Red Blood Cells</strong>, <strong>Platelets</strong>,
        and <strong>White Blood Cell subtypes</strong> (Neutrophil, Lymphocyte,
        Monocyte, Eosinophil, Basophil) in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Image source selection ────────────────────────────────────────────────
tab_upload, tab_sample = st.tabs(["📤 Upload Image", "🖼️ Sample Images"])

input_image: np.ndarray | None = None
image_name: str = ""

with tab_upload:
    uploaded = st.file_uploader(
        "Drag and drop a blood smear image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Max 10 MB.",
    )
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        input_image = np.array(pil)
        image_name = uploaded.name

with tab_sample:
    samples = get_sample_images()
    if samples:
        st.markdown("Pick one of the built-in test images:")
        cols = st.columns(min(len(samples), 3))
        for idx, sample_path in enumerate(samples):
            col = cols[idx % 3]
            with col:
                thumb = Image.open(sample_path).convert("RGB")
                thumb.thumbnail((200, 200))
                st.image(thumb, caption=sample_path.stem, width="stretch")
                if st.button(f"Use {sample_path.stem}", key=f"sample_{idx}", use_container_width=True):
                    pil = Image.open(sample_path).convert("RGB")
                    input_image = np.array(pil)
                    image_name = sample_path.name
    else:
        st.info("No sample images found in `test_images/` directory.")

# ── Run detection ─────────────────────────────────────────────────────────
if input_image is not None:
    st.markdown("---")

    # Load model with a nice spinner
    with st.spinner("🔄 Loading model…"):
        model = load_model()

    with st.spinner("🔍 Running detection…"):
        det = run_detection(model, input_image, conf_threshold, iou_threshold)

    annotated = annotate_image(
        input_image,
        det["boxes"],
        det["classes"],
        det["confs"],
        det["names"],
        show_labels=show_labels,
        show_conf=show_conf,
    )

    # ── Metric cards ──────────────────────────────────────────────────────
    st.markdown("### 📊 Detection Summary")
    m_cols = st.columns(4)

    rbc_count = det["counts"].get("RBC", 0)
    plt_count = det["counts"].get("Platelets", 0)
    wbc_count = sum(det["counts"].get(w, 0) for w in WBC_SUBTYPES)

    with m_cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="label">Total Cells</div>'
            f'<div class="value">{det["total"]}</div></div>',
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            f'<div class="metric-card"><div class="label">🔴 RBC</div>'
            f'<div class="value">{rbc_count}</div></div>',
            unsafe_allow_html=True,
        )
    with m_cols[2]:
        st.markdown(
            f'<div class="metric-card"><div class="label">🟢 Platelets</div>'
            f'<div class="value">{plt_count}</div></div>',
            unsafe_allow_html=True,
        )
    with m_cols[3]:
        st.markdown(
            f'<div class="metric-card"><div class="label">🔵 WBC</div>'
            f'<div class="value">{wbc_count}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # ── Side-by-side images ───────────────────────────────────────────────
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("#### Original")
        st.image(input_image, width="stretch")
    with col_det:
        st.markdown("#### Detected")
        st.image(annotated, width="stretch")

    # ── Download button ───────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download annotated image",
        data=image_to_bytes(annotated),
        file_name=f"detected_{image_name}",
        mime="image/jpeg",
        use_container_width=True,
    )

    # ── Detailed class breakdown ──────────────────────────────────────────
    if det["total"] > 0:
        st.markdown("### 🧬 Detailed Class Breakdown")

        all_classes = ["RBC", "Platelets", "Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
        rows = []
        for cls_name in all_classes:
            cnt = det["counts"].get(cls_name, 0)
            if cnt > 0:
                r, g, b = CLASS_COLORS_RGB[cls_name]
                pct = cnt / det["total"] * 100
                rows.append(
                    f'<tr>'
                    f'<td><span class="dot" style="background:rgb({r},{g},{b})"></span>{cls_name}</td>'
                    f'<td style="text-align:right;font-weight:600">{cnt}</td>'
                    f'<td style="text-align:right">{pct:.1f}%</td>'
                    f'</tr>'
                )

        table_html = (
            '<table class="cell-table">'
            "<thead><tr><th>Class</th><th style='text-align:right'>Count</th>"
            "<th style='text-align:right'>%</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        # ── WBC differential (if WBCs detected) ──────────────────────────
        if wbc_count > 0:
            st.markdown("### 🩸 WBC Differential")
            st.caption("Proportion of each WBC subtype among all detected white blood cells.")
            diff_cols = st.columns(5)
            for i, wbc_name in enumerate(["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]):
                cnt = det["counts"].get(wbc_name, 0)
                pct = cnt / wbc_count * 100 if wbc_count else 0
                with diff_cols[i]:
                    st.metric(label=wbc_name, value=f"{cnt}", delta=f"{pct:.1f}%")

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("")
    col_empty = st.columns([1, 2, 1])[1]
    with col_empty:
        st.markdown(
            """
            <div style="text-align:center; padding:3rem 1rem; opacity:0.6;">
                <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
                <h3 style="color:#8892a4; font-weight:500;">No image selected</h3>
                <p style="color:#6b7280;">Upload a blood smear image or choose a sample to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
