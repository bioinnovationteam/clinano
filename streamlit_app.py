import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import colorsys
from typing import List, Dict, Tuple, Optional


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
def detect_dipstick_regions(image) -> Dict:
    """
    Detect all regions with a black outline on a urine dipstick in top-to-bottom order.
    
    Args:
        image: PIL Image or numpy array (RGB format)
    
    Returns:
        Dictionary with:
        - 'success': bool
        - 'regions': List of dicts with keys: 'name', 'type', 'bounds', 'center', 'aspect_ratio'
        - 'reference_strips': Subset for just reference strips
        - 'test_pads': Subset for just test pads
    """
    # ── 1. Normalize input ───────────────────────────────────────────────────
    img = np.array(image)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img_area = img.shape[0] * img.shape[1]

        # ── 2. Detect black outlines ─────────────────────────────────────────────
    outline_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    kernel = np.ones((3, 3), np.uint8)
    outline_mask = cv2.erode(cv2.dilate(outline_mask, kernel, iterations=3), kernel, iterations=1)

    all_contours, hierarchy = cv2.findContours(outline_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # ── 3. Filter contours to valid outlined regions ─────────────────────────
    potential_regions = []
    for i, contour in enumerate(all_contours):
        # Only keep child contours (those with a parent) — skips the outer holder
        if hierarchy[0][i][3] == -1:  # -1 means no parent = outermost = holder outline
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

        # Discard empty/black-filled regions — only keep colored interiors
        if avg_hsv[1] > 15 and avg_hsv[2] > 30:
            potential_regions.append({
                'bounds': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': area,
                'aspect_ratio': round(w / h, 3) if h > 0 else 0,
                'avg_color_bgr': avg_bgr,
                'avg_color_hsv': avg_hsv,
            })

    # ── 4. Sort top-to-bottom and assign generic names ───────────────────────
    potential_regions.sort(key=lambda r: r['center'][1])

    all_regions = []
    for i, region in enumerate(potential_regions):
        r, g, b = region['avg_color_bgr'][2], region['avg_color_bgr'][1], region['avg_color_bgr'][0]
        h_hsv = region['avg_color_hsv']
        all_regions.append({
            'name':         f"Region {i + 1}",
            'type':         'unknown',
            'bounds':       region['bounds'],
            'center':       region['center'],
            'area':         region['area'],
            'aspect_ratio': region['aspect_ratio'],
            'color_rgb':    (float(r), float(g), float(b)),
            'color_hsv':    (float(h_hsv[0]), float(h_hsv[1]), float(h_hsv[2])),
        })

    # ── 5. Return using same output structure ────────────────────────────────
    found_names = [r['name'] for r in all_regions]
    success = len(all_regions) > 0

    if not success:
        print("Warning: No outlined regions detected.")
    else:
        print(f"Found {len(all_regions)} outlined regions: {found_names}")

    return {
        'success':          success,
        'regions':          all_regions,
        'reference_strips': [],   # populated by caller once regions are confirmed
        'test_pads':        [],   # populated by caller once regions are confirmed
        'order_verified':   False
    }



def visualize_detection_streamlit(image, detection_results: Dict):
    """
    Create visualization for Streamlit display (returns image array).
    """
    img = np.array(image)
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img.copy()

    region_names = {
        "Region 1": "Blue Ref",
        "Region 2": "Red Ref",
        "Region 3": "Green Ref",
        "Region 4": "Albumin Pad",
        "Region 5": "Creatinine Pad",
        "Region 6": "pH Pad",
    }

    palette = [
        (0, 255, 0),    (255, 0, 0),   (0, 0, 255),
        (0, 255, 255),  (255, 0, 255), (255, 165, 0),
        (128, 0, 128),  (0, 128, 255), (0, 255, 128),
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



def process_regions_average_explicit(regions, black_threshold=30, color_order='RGB'):
    """
    Takes 6 detected regions, removes strong black pixels, and outputs average RGB values.
    
    Parameters:
    - regions: List of 6 image regions (numpy arrays)
    - black_threshold: Pixels with all RGB values <= this threshold are "strong black"
    - color_order: 'RGB' or 'BGR' - specifies input color order
    
    Returns:
    - List of 6 tuples (R_avg, G_avg, B_avg)
    """
    avg_rgb_values = []
    
    for idx, region in enumerate(regions):
        # Convert to numpy array if needed
        if not isinstance(region, np.ndarray):
            region = np.array(region)
        
        # Convert to RGB if necessary
        if color_order.upper() == 'BGR' and region.shape[2] == 3:
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        else:
            region_rgb = region.copy()
        
        # Create mask for non-black pixels
        # A pixel is considered black if all channels <= threshold
        is_black = np.all(region_rgb <= black_threshold, axis=2)
        non_black_mask = ~is_black
        
        # Calculate average of non-black pixels
        if np.any(non_black_mask):
            # Extract non-black pixels
            non_black = region_rgb[non_black_mask]
            # Calculate average
            avg = np.mean(non_black, axis=0)
            avg_rgb = tuple(avg.astype(int))
        else:
            # No valid pixels found
            avg_rgb = (0, 0, 0)
        
        avg_rgb_values.append(avg_rgb)
    
    return avg_rgb_values



def calculate_albumin_concentration(pad_rgb, refs):
    """
    Calculate albumin concentration based on pad RGB and reference colors.
    This is a placeholder - replace with your actual calibration logic.
    """
    # Example: Use green reference for intensity normalization
    green_ref = refs['Green']['color_rgb']
    
    # Simple intensity calculation (placeholder)
    pad_intensity = np.mean(pad_rgb)
    ref_intensity = np.mean(green_ref)
    
    # Normalize
    normalized = pad_intensity / ref_intensity if ref_intensity > 0 else 0
    
    # Map to concentration (calibration curve needed)
    # This is just an example - replace with your actual calibration
    concentration = normalized * 300  # 0-300 mg/dL range
    
    return np.clip(concentration, 0, 300)



def calculate_creatinine_concentration(pad_rgb, refs):
    """
    Calculate creatinine concentration based on pad RGB and reference colors.
    Placeholder - replace with actual calibration.
    """
    # Similar to albumin but with different calibration
    red_ref = refs['Red']['color_rgb']
    
    pad_intensity = np.mean(pad_rgb)
    ref_intensity = np.mean(red_ref)
    
    normalized = pad_intensity / ref_intensity if ref_intensity > 0 else 0
    
    # Creatinine range typically 0-300 mg/dL
    concentration = normalized * 200
    
    return np.clip(concentration, 0, 200)



def interpret_pad_3(pad_rgb, refs):
    """
    Interpret pH pad (Pad 3) based on color.
    Placeholder - replace with actual interpretation logic.
    """
    # pH typically ranges from 5.0 to 8.5
    # Colors might range from orange (acidic) to green/blue (basic)
    
    # Example: Use red-green ratio for pH estimation
    r, g, b = pad_rgb
    
    if r > g and r > b:
        return "Acidic (pH 5.0-6.0)"
    elif g > r and g > b:
        return "Neutral (pH 6.5-7.5)"
    elif b > r and b > g:
        return "Alkaline (pH 8.0-8.5)"
    else:
        return "Mixed - check manually"

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
    img_file = st.camera_input("Position dipstick so all 3 reference strips and 3 test pads are visible",
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

    # Run unified detection (all 6 regions)
    detection_results = detect_dipstick_regions(image)
    annotated = visualize_detection_streamlit(image, detection_results)

    with col_ann:
        st.caption("Detected regions")
        st.image(annotated)

    # Map generic Region N names to friendly display names
    region_display_names = {
        "Region 1": "Blue Ref",
        "Region 2": "Red Ref",
        "Region 3": "Green Ref",
        "Region 4": "Albumin Pad",
        "Region 5": "Creatinine Pad",
        "Region 6": "pH Pad",
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
        st.warning(f"Only {len(detection_results['regions'])}/6 regions detected. "
                   "Try better lighting, reposition the dipstick, or ensure all regions have clear black outlines.")

        missing = [display_name for region_key, display_name in region_display_names.items()
                   if region_key not in found_names]
        if missing:
            st.info(f"Missing: {', '.join(missing)}")
    else:
        with st.expander("### Detected Regions (top to bottom)"):
            for region in detection_results['regions']:
                display_name = region_display_names.get(region['name'], region['name'])
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{display_name}**")
                with col2:
                    st.write(f"Aspect: {region['aspect_ratio']:.2f}")
                with col3:
                    st.write(f"Area: {region['area']:.0f} px")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 3: Analyze ───────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">Step 03 — Analyze</div>', unsafe_allow_html=True)

    if st.button("🔬 Run Full Analysis", use_container_width=True, disabled=(not detection_results['success'])):
            with st.spinner("Measuring pad colors and calculating concentrations…"):
                regions = detection_results['regions']
                
                # Extract the actual image data from each detected region
                # Note: you need to store the original image crop when detecting regions.
                # If your detection doesn't save the cropped image, you must modify detect_dipstick_regions
                # to include an 'image_crop' key. For now, assuming you have region['image'].
                # If not, you'll need to extract crops from the original image using region['bounds'].
                
                # Get the 6 region images (you must have stored them during detection)
                region_images = [region['image'] for region in regions]  # ← you need to add this during detection
                
                # Remove strong black pixels and get average RGB for each region
                processed_colors = process_regions_average_explicit(region_images, black_threshold=30, color_order='RGB')
                
                # Split into refs (first 3) and pads (last 3)
                ref_colors = processed_colors[:3]   # Blue, Red, Green
                pad_colors = processed_colors[3:]   # Albumin, Creatinine, pH
                
                # Build refs dictionary with cleaned colors
                refs = {}
                ref_names = ["Blue", "Red", "Green"]
                for i, name in enumerate(ref_names):
                    refs[name] = {
                        'center':       regions[i]['center'],
                        'bounds':       regions[i]['bounds'],
                        'area':         regions[i]['area'],
                        'aspect_ratio': regions[i]['aspect_ratio'],
                        'color_rgb':    ref_colors[i],   # cleaned RGB
                        'score':        1.0,
                    }
                
                # Build colors dictionary with cleaned pad colors
                pad_names = ["Albumin", "Creatinine", "pH"]
                colors = {}
                for i, name in enumerate(pad_names):
                    colors[name] = pad_colors[i]   # cleaned RGB
                
                # Now calculate concentrations (outside the loop)
                alb_conc = calculate_albumin_concentration(colors["Albumin"], refs)
                creat_conc = calculate_creatinine_concentration(colors["Creatinine"], refs)
                uacr = (alb_conc / creat_conc * 1000) if creat_conc > 0 else 0   # typical uACR formula
                
                # Store results
                st.session_state.results = {
                    "uacr":              uacr,
                    "albumin":           alb_conc,
                    "creatinine":        creat_conc,
                    "colors":            colors,
                    "refs":              refs,
                    "detection_results": detection_results,
                    "annotated":         annotated,
                    "pad_3_category":    interpret_pad_3(colors["pH"], refs),
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
  <li>Ensure all <strong>3 reference strips</strong> (Blue, Red, Green) are in frame</li>
  <li>Ensure all <strong>3 test pads</strong> (square regions with black outlines) are visible</li>
  <li>Avoid glare — diffuse natural light works best</li>
  <li>Hold camera <strong>directly above</strong> the strip, not at an angle</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Research prototype — not for clinical use. Consult a healthcare provider for medical decisions.")