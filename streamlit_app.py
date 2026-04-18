import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import colorsys

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
.status-normal   { background:#EDFAF1; border-left:3px solid #27AE60; border-radius:6px; padding:0.8rem 1rem; }
.status-moderate { background:#FFF8E6; border-left:3px solid #F39C12; border-radius:6px; padding:0.8rem 1rem; }
.status-severe   { background:#FEECEC; border-left:3px solid #E74C3C; border-radius:6px; padding:0.8rem 1rem; }
.status-label    { font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:500; }
.status-sub      { font-size:0.83rem; color:#555; margin-top:0.3rem; }

/* ── Color swatches ── */
.swatch-row { display:flex; gap:1rem; margin-top:0.6rem; }
.swatch-item { flex:1; text-align:center; }
.swatch-box  { height:40px; border-radius:6px; border:1px solid rgba(0,0,0,0.08); margin-bottom:0.4rem; }
.swatch-name { font-family:'DM Mono',monospace; font-size:0.7rem; color:#888; }
.swatch-rgb  { font-family:'DM Mono',monospace; font-size:0.65rem; color:#AAA; }

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

def find_reference_strips(image):
    """
    Find RGB reference strips using HSV color masking and contour analysis.
    
    Args:
        image: PIL Image or numpy array (RGB format)
    
    Returns:
        dict with keys 'Red', 'Green', 'Blue' containing detection info
    """
    # ── 1. Normalize input ───────────────────────────────────────────────────
    img = np.array(image)
    
    # Handle both RGB (PIL) and BGR (cv2.imread) inputs.
    # Convert RGB → BGR → HSV, which is the correct OpenCV pipeline.
    # If your image is already BGR, replace cvtColor line with just:
    #   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img  # assume BGR already

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    img_area = img.shape[0] * img.shape[1]

    # ── 2. Relaxed area bounds ────────────────────────────────────────────────
    # Lowered min to 0.0005 (half a percent), raised max to 0.5 (half the image)
    # Reference strips can be large color blocks
    min_strip_area = img_area * 0.0005
    max_strip_area = img_area * 0.5

    # ── 3. Widened HSV ranges ─────────────────────────────────────────────────
    # Saturation/value floors lowered from 30 → 50 for hue but 40 minimum
    # for value to reject near-black without losing dark-ish saturated colors.
    color_ranges = {
        "Red": [
            ([0,   80, 50], [10,  255, 255]),   # lower red hue
            ([165, 80, 50], [180, 255, 255]),   # upper red hue (wraps around 180)
        ],
        "Green": [
            ([40, 60, 50], [90, 255, 255]),     # pure green range
        ],
        "Blue": [
            ([95, 60, 50], [135, 255, 255]),    # pure blue range
        ],
    }

    refs = {}

    for name, ranges in color_ranges.items():

        # ── 4. Build combined mask ────────────────────────────────────────────
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # ── 5. Morphological cleanup ──────────────────────────────────────────
        # Use a modest kernel; too large and you'll bridge adjacent color regions
        kernel_size = max(3, min(img.shape[0], img.shape[1]) // 150)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── 6. Filter contours ────────────────────────────────────────────────
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_strip_area or area > max_strip_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # FIX: horizontal strips are wide, so aspect_ratio > 1.
            # Old ceiling of 3.0 rejected 400×60px strips (ratio = 6.7).
            # New range: 0.2 (tall thin) to 20.0 (very wide strip)
            if not (0.2 < aspect_ratio < 20.0):
                continue

            # Require roughly rectangular shape
            peri  = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) >= 4:
                valid_contours.append((area, x, y, w, h, contour))

        if not valid_contours:
            continue

        # ── 7. Score and pick best candidate ─────────────────────────────────
        # Sort by area descending; consider top 5 not just top 3
        valid_contours.sort(key=lambda c: c[0], reverse=True)

        best, best_score = None, -1
        for area, x, y, w, h, contour in valid_contours[:5]:
            # Reward aspect ratios near a wide horizontal strip (w > h)
            ar = w / h if h > 0 else 1.0
            # Score peaks at ar=2 (2:1 wide strip), tolerant either side
            aspect_score = max(0.0, 1.0 - abs(ar - 2.0) / 5.0)
            area_score   = min(area / max_strip_area, 1.0)
            score        = aspect_score * 0.6 + area_score * 0.4

            if score > best_score:        # FIX: only update best_score here
                best_score = score
                best = (area, x, y, w, h, contour)

        if best:
            area, x, y, w, h, contour = best
            refs[name] = {
                "center":       (x + w // 2, y + h // 2),
                "bounds":       (x, y, w, h),
                "area":         area,
                "aspect_ratio": round(w / h, 3),
                "score":        round(best_score, 3),
            }

    return refs


def draw_strip_highlights(image: Image.Image, refs: dict) -> Image.Image:
    """
    Draw labelled bounding boxes around detected reference strips.
    Returns an annotated RGB PIL image.
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    COLOR_MAP = {
        "Red":   (0,   0,   220),   # BGR
        "Green": (0,   180,  30),
        "Blue":  (210,  80,   0),
    }
    LABEL_BG = {
        "Red":   (0,   0,   220),
        "Green": (0,   180,  30),
        "Blue":  (210,  80,   0),
    }

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.45, img_cv.shape[0] / 2000)
    thickness  = max(2, img_cv.shape[0] // 400)

    for name, info in refs.items():
        x, y, w, h = info["bounds"]
        bgr         = COLOR_MAP.get(name, (200, 200, 200))
        label_bgr   = LABEL_BG.get(name, (200, 200, 200))

        # Bounding box
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), bgr, thickness)

        # Corner accents
        corner_len = max(8, min(w, h) // 5)
        pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for cx, cy in pts:
            dx = corner_len if cx == x else -corner_len
            dy = corner_len if cy == y else -corner_len
            cv2.line(img_cv, (cx, cy), (cx + dx, cy), bgr, thickness + 1)
            cv2.line(img_cv, (cx, cy), (cx, cy + dy), bgr, thickness + 1)

        # Label pill
        label       = f" {name} "
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        pad         = 4
        lx          = x
        ly          = max(y - th - pad * 2, 0)
        cv2.rectangle(img_cv, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2), label_bgr, -1)
        cv2.putText(img_cv, label, (lx + pad, ly + th + pad - 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Score badge (small, bottom-right of box)
        score_txt     = f"{info['score']:.2f}"
        (sw, sh), _   = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_PLAIN, 0.85, 1)
        cv2.rectangle(img_cv, (x + w - sw - 6, y + h - sh - 4),
                      (x + w, y + h), bgr, -1)
        cv2.putText(img_cv, score_txt, (x + w - sw - 3, y + h - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def estimate_pads(refs: dict) -> dict:
    """
    Estimate positions of Albumin and Creatinine pads by interpolating
    from reference strip geometry (simple linear model).
    """
    if len(refs) < 2:
        return {}

    sorted_refs = sorted(refs.items(), key=lambda item: item[1]["center"][1])
    # Use the vertical span of the refs to define a coordinate system
    top_y    = sorted_refs[0][1]["center"][1]
    bot_y    = sorted_refs[-1][1]["center"][1]
    span     = bot_y - top_y if bot_y != top_y else 1

    avg_x    = int(np.mean([v["center"][0] for _, v in refs.items()]))
    avg_w    = int(np.mean([v["bounds"][2]  for _, v in refs.items()]))
    avg_h    = int(np.mean([v["bounds"][3]  for _, v in refs.items()]))

    # Place pads above the top reference at equal spacing
    pad_gap  = max(avg_h + 4, span // max(len(refs), 1))
    pads     = {}
    for i, pad_name in enumerate(["Albumin", "Creatinine"]):
        cy     = top_y - pad_gap * (i + 1)
        cx     = avg_x
        pads[pad_name] = {
            "center": (cx, cy),
            "bounds": (cx - avg_w // 2, cy - avg_h // 2, avg_w, avg_h),
        }
    return pads


def measure_colors(image: Image.Image, regions: dict) -> dict:
    """Sample mean RGB inside each region's bounding box."""
    img = np.array(image)
    colors = {}
    h_img, w_img = img.shape[:2]

    for name, info in regions.items():
        x, y, w, h = info["bounds"]
        # Clamp to image bounds
        x1 = max(0, x);       y1 = max(0, y)
        x2 = min(w_img, x+w); y2 = min(h_img, y+h)
        if x2 > x1 and y2 > y1:
            patch  = img[y1:y2, x1:x2]
            colors[name] = patch.mean(axis=(0, 1))[:3]
        else:
            colors[name] = np.array([128.0, 128.0, 128.0])

    return colors


def rgb_to_conc(rgb: np.ndarray, analyte: str) -> float:
    """
    Demo conversion: maps RGB → concentration using a simple heuristic.
    Replace with your calibrated model.
    """
    r, g, b = rgb
    if analyte == "albumin":
        # Darker / more blue → higher albumin
        brightness = (r + g + b) / 3.0
        return max(0.0, (255 - brightness) / 255 * 500)
    elif analyte == "creatinine":
        # Greener → higher creatinine
        greenness = g - (r + b) / 2.0
        return max(1.0, 50 + greenness * 2)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="page-title">uACR Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Dipstick · 3-Point Reference Calibration</p>', unsafe_allow_html=True)

# ── Step 1: Image input ────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">Step 01 — Capture</div>', unsafe_allow_html=True)

method  = st.radio("", ["📷  Camera", "📁  Upload"], horizontal=True, label_visibility="collapsed")
img_file = None

if "Camera" in method:
    img_file = st.camera_input("Position dipstick so all 3 reference strips and detection pads are visible",
                                label_visibility="visible")
else:
    img_file = st.file_uploader("Upload dipstick image (JPG or PNG)",
                                 type=["jpg", "jpeg", "png"],
                                 label_visibility="visible")

st.markdown("</div>", unsafe_allow_html=True)

# ── Step 2: Detect & preview ───────────────────────────────────────────────
if img_file:
    image = Image.open(img_file).convert("RGB")

    st.markdown('<div class="card"><div class="card-label">Step 02 — Detect Strips</div>', unsafe_allow_html=True)

    col_orig, col_ann = st.columns(2)

    with col_orig:
        st.caption("Original")
        st.image(image)

    # Run detection immediately for the preview
    refs = find_reference_strips(image)
    annotated = draw_strip_highlights(image, refs)

    with col_ann:
        st.caption("Detected strips")
        st.image(annotated)

    # Detection status badges
    found_set   = set(refs.keys())
    all_colors  = ["Red", "Green", "Blue"]
    badge_html  = ""
    for c in all_colors:
        if c in found_set:
            badge_html += f'<span class="detect-badge badge-found">✓ {c}</span>'
        else:
            badge_html += f'<span class="detect-badge badge-missing">✗ {c}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    if len(refs) < 3:
        st.warning(f"Only {len(refs)}/3 reference strips detected. "
                   "Try better lighting or reposition the dipstick.")
    else:
        strip_info = "  |  ".join(
            f"**{n}** — aspect {v['aspect_ratio']:.2f}, score {v['score']:.2f}"
            for n, v in refs.items()
        )
        st.caption(strip_info)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 3: Analyze ───────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">Step 03 — Analyze</div>', unsafe_allow_html=True)

    if st.button("🔬  Run Full Analysis", use_container_width=True, disabled=(len(refs) < 3)):
        with st.spinner("Measuring pad colors and calculating concentrations…"):
            pads         = estimate_pads(refs)
            colors       = measure_colors(image, pads)
            alb_conc     = rgb_to_conc(colors.get("Albumin",    np.array([128,128,128])), "albumin")
            creat_conc   = rgb_to_conc(colors.get("Creatinine", np.array([128,128,128])), "creatinine")
            uacr         = alb_conc / (creat_conc / 100) if creat_conc > 0 else 0

            st.session_state.results = {
                "uacr":        uacr,
                "albumin":     alb_conc,
                "creatinine":  creat_conc,
                "colors":      colors,
                "refs":        refs,
                "annotated":   annotated,
            }
            st.rerun()

    if len(refs) < 3:
        st.caption("⚠️ Analysis disabled until all 3 reference strips are detected.")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results

    st.markdown('<div class="card"><div class="card-label">Results</div>', unsafe_allow_html=True)

    # Annotated image (re-show in results section)
    if "annotated" in res:
        st.image(res["annotated"], caption="Detected reference strips", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Albumin",    f"{res['albumin']:.0f} mg/dL")
    c2.metric("Creatinine", f"{res['creatinine']:.0f} mg/dL")
    c3.metric("uACR",       f"{res['uacr']:.1f} mg/g")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Color swatches for measured pads
    if res["colors"]:
        st.markdown('<div class="card-label" style="margin-bottom:0.4rem;">Measured Pad Colors</div>', unsafe_allow_html=True)
        swatch_html = '<div class="swatch-row">'
        for name, rgb in res["colors"].items():
            hex_c = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            swatch_html += f"""
            <div class="swatch-item">
                <div class="swatch-box" style="background:{hex_c};"></div>
                <div class="swatch-name">{name}</div>
                <div class="swatch-rgb">{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}</div>
            </div>"""
        swatch_html += "</div>"
        st.markdown(swatch_html, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Interpretation
    uacr = res["uacr"]
    if uacr < 30:
        st.markdown("""
        <div class="status-normal">
            <div class="status-label">✅ Normal</div>
            <div class="status-sub">uACR &lt; 30 mg/g — within normal range</div>
        </div>""", unsafe_allow_html=True)
    elif uacr < 300:
        st.markdown("""
        <div class="status-moderate">
            <div class="status-label">⚠️ Moderately Increased</div>
            <div class="status-sub">uACR 30–300 mg/g — Microalbuminuria detected</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-severe">
            <div class="status-label">🔴 Severely Increased</div>
            <div class="status-sub">uACR &gt; 300 mg/g — Macroalbuminuria detected</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  New Analysis", use_container_width=True):
        st.session_state.results = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card" style="margin-top:1.5rem;">
<div class="card-label">How to use</div>
<ul style="font-size:0.85rem; color:#555; margin:0; padding-left:1.2rem; line-height:1.8;">
  <li>Place dipstick on a <strong>flat, well-lit</strong> surface</li>
  <li>Ensure all <strong>3 reference strips</strong> (Red, Green, Blue) are in frame</li>
  <li>Avoid glare — diffuse natural light works best</li>
  <li>Hold camera <strong>directly above</strong> the strip, not at an angle</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Research prototype — not for clinical use. Consult a healthcare provider for medical decisions.")