import streamlit as st
import numpy as np
from PIL import Image
import cv2


# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="uACR Dipstick Analyzer",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #F5F4F0 !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 680px !important; }

/* ── Page title ── */
.page-title {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #1A1A1A;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E0D8;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #AAA;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Radio buttons ── */
.stRadio > label { display: none; }
.stRadio > div { gap: 0.5rem; }
.stRadio [data-baseweb="radio"] > label {
    background: #F5F4F0;
    border: 1px solid #E2E0D8;
    border-radius: 20px;
    padding: 6px 18px;
    font-size: 0.85rem;
    color: #555;
    cursor: pointer;
    transition: all 0.15s;
}
.stRadio [data-baseweb="radio"][aria-checked="true"] > label {
    background: #1A1A1A;
    border-color: #1A1A1A;
    color: #FFF;
}

/* ── Buttons ── */
.stButton > button {
    background: #1A1A1A !important;
    color: #FFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.2rem !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #F5F4F0;
    border: 1px solid #E2E0D8;
    border-radius: 10px;
    padding: 0.8rem 1rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #888 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #1A1A1A !important;
}

/* ── Status banners ── */
.status-normal   { background:#EDFAF1; border-left:3px solid #27AE60; border-radius:6px; padding:0.8rem 1rem; margin-bottom: 1.5rem; }
.status-moderate { background:#FFF8E6; border-left:3px solid #F39C12; border-radius:6px; padding:0.8rem 1rem; margin-bottom: 1.5rem; }
.status-severe   { background:#FEECEC; border-left:3px solid #E74C3C; border-radius:6px; padding:0.8rem 1rem; margin-bottom: 1.5rem; }
.status-label    { font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:500; }
.status-sub      { font-size:0.83rem; color:#555; margin-top:0.3rem; }

/* ── Detection badge ── */
.detect-badge {
    display:inline-block;
    font-family:'DM Mono',monospace;
    font-size:0.68rem;
    letter-spacing:0.08em;
    padding:3px 10px;
    border-radius:20px;
    margin-right:6px;
    margin-bottom:4px;
}
.badge-found   { background:#EDFAF1; color:#27AE60; border:1px solid #A9DFB8; }
.badge-missing { background:#FEECEC; color:#E74C3C; border:1px solid #F4AEAE; }

/* ── Divider ── */
hr { border:none; border-top:1px solid #E2E0D8; margin:1rem 0; }

/* ── Results uACR display ── */
.results-uacr-display {
    text-align: center;
    padding: 1.5rem 0;
    margin: 1rem 0;
}

.results-uacr-value {
    font-family: 'DM Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    color: #1A1A1A;
    line-height: 1;
}

.results-uacr-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Force text color ── */
p, span, div, label { color: #1A1A1A; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def detect_dipstick_regions(image) -> dict:
    """
    Detect all regions with a black outline on a urine dipstick.
    Returns success=True if 6+ regions found.
    """
    img = np.array(image)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img_area = img.shape[0] * img.shape[1]

    # Detect black outlines
    outline_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    kernel = np.ones((3, 3), np.uint8)
    outline_mask = cv2.erode(cv2.dilate(outline_mask, kernel, iterations=3), kernel, iterations=1)

    all_contours, hierarchy = cv2.findContours(outline_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    potential_regions = []
    for i, contour in enumerate(all_contours):
        if hierarchy[0][i][3] == -1:  # Skip outermost outline
            continue

        area = cv2.contourArea(contour)
        if not (img_area * 0.0001 < area < img_area * 0.5):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        peri = cv2.arcLength(contour, True)
        if len(cv2.approxPolyDP(contour, 0.06 * peri, True)) < 3:
            continue

        interior_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.rectangle(interior_mask, (x, y), (x + w, y + h), 255, -1)
        interior_pixels = bgr[interior_mask == 255]
        if len(interior_pixels) == 0:
            continue

        avg_bgr = np.mean(interior_pixels, axis=0)
        avg_hsv = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

        if avg_hsv[1] > 15 and avg_hsv[2] > 30:
            potential_regions.append({
                'bounds': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': area,
            })

    # Sort top-to-bottom
    potential_regions.sort(key=lambda r: r['center'][1])

    all_regions = []
    for i, region in enumerate(potential_regions):
        all_regions.append({
            'name':  f"Region {i + 1}",
            'bounds': region['bounds'],
            'center': region['center'],
            'area':   region['area'],
        })

    success = len(all_regions) >= 6

    return {
        'success': success,
        'regions': all_regions,
    }


def visualize_detection_streamlit(image, detection_results: dict):
    """
    Create visualization for Streamlit display.
    """
    img = np.array(image)
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img.copy()

    region_names = {
        "Region 1": "Blue Ref",
        "Region 2": "Red Ref",
        "Region 3": "Green Ref",
        "Region 4": "Albumin",
        "Region 5": "Creatinine",
        "Region 6": "pH",
    }

    palette = [
        (0, 255, 0),    (255, 0, 0),   (0, 0, 255),
        (0, 255, 255),  (255, 0, 255), (255, 165, 0),
    ]

    for i, region in enumerate(detection_results['regions'][:6]):
        x, y, w, h = region['bounds']
        color = palette[i % len(palette)]

        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 3)

        label = region_names.get(region['name'], region['name'])
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        cv2.rectangle(display_img,
                      (x, y - text_size[1] - 8),
                      (x + text_size[0] + 8, y - 3),
                      color, -1)
        cv2.putText(display_img, label, (x + 4, y - 5),
                    font, font_scale, (0, 0, 0), thickness)

    return cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="page-title">uACR Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Dipstick · 6-Point Detection (3 References + 3 Test Pads)</p>', unsafe_allow_html=True)

# ── Step 1: Image input ────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">Step 01 — Capture</div>', unsafe_allow_html=True)

method = st.radio("", ["📷  Camera", "📁  Upload"], horizontal=True, label_visibility="collapsed")
img_file = None

if "Camera" in method:
    img_file = st.camera_input("Position dipstick so all regions are visible",
                                label_visibility="visible")
else:
    img_file = st.file_uploader("Upload dipstick image (JPG or PNG)",
                                 type=["jpg", "jpeg", "png"],
                                 label_visibility="visible")

st.markdown("</div>", unsafe_allow_html=True)

# ── Step 2: Detect & preview ───────────────────────────────────────────────
if img_file:
    image = Image.open(img_file).convert("RGB")

    st.markdown('<div class="card"><div class="card-label">Step 02 — Detect Regions</div>', unsafe_allow_html=True)

    col_orig, col_ann = st.columns(2)

    with col_orig:
        st.caption("Original")
        st.image(image)

    # Run detection
    detection_results = detect_dipstick_regions(image)
    annotated = visualize_detection_streamlit(image, detection_results)

    with col_ann:
        st.caption("Detected regions")
        st.image(annotated)

    # Map region names
    region_display_names = {
        "Region 1": "Blue Ref",
        "Region 2": "Red Ref",
        "Region 3": "Green Ref",
        "Region 4": "Albumin",
        "Region 5": "Creatinine",
        "Region 6": "pH",
    }

    found_names = [r['name'] for r in detection_results['regions']]

    # Detection status badges
    badge_html = '<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0;">'
    for region_key, display_name in region_display_names.items():
        if region_key in found_names:
            badge_html += f'<span class="detect-badge badge-found">✓ {display_name}</span>'
        else:
            badge_html += f'<span class="detect-badge badge-missing">✗ {display_name}</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    if not detection_results['success']:
        st.warning(f"Only {len(detection_results['regions'])}/6 regions detected.")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 3: Analyze ───────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">Step 03 — Analyze</div>', unsafe_allow_html=True)

    if st.button("🔬 Run Full Analysis", use_container_width=True, disabled=(not detection_results['success'])):
        with st.spinner("Calculating uACR…"):
            # Generate random uACR score
            uacr = np.random.uniform(10, 500)
            albumin = np.random.uniform(5, 150)
            creatinine = np.random.uniform(20, 100)
            
            # Store results
            st.session_state.results = {
                "uacr": uacr,
                "albumin": albumin,
                "creatinine": creatinine,
                "annotated": annotated,
            }
            st.rerun()

    if not detection_results['success']:
        st.caption("⚠️ Analysis disabled until all 6 regions are detected.")

    st.markdown("</div>", unsafe_allow_html=True)



# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card" style="margin-top:1.5rem;">
<div class="card-label">How to use</div>
<ul style="font-size:0.85rem; color:#555; margin:0; padding-left:1.2rem; line-height:1.8;">
  <li>Place dipstick on a <strong>flat, well-lit</strong> surface</li>
  <li>Ensure all <strong>6 regions</strong> are visible and have clear black outlines</li>
  <li>Avoid glare — diffuse natural light works best</li>
  <li>Hold camera <strong>directly above</strong> the strip</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Research prototype — not for clinical use.")