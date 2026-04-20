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
    Detect all 6 outlined regions on a urine dipstick in top-to-bottom order.
    
    Order: Blue Reference, Red Reference, Green Reference, Pad 1, Pad 2, Pad 3
    
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
    
    # Convert RGB → BGR for OpenCV
    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img_area = img.shape[0] * img.shape[1]
    
    # ── 2. Area bounds for different region types ────────────────────────────
    min_strip_area = img_area * 0.0005      # 0.05% of image
    max_strip_area = img_area * 0.5         # 50% of image
    min_pad_area = img_area * 0.0003        # 0.03% of image (pads can be smaller)
    max_pad_area = img_area * 0.3           # 30% of image
    
    # ── 3. HSV color ranges for reference strips ─────────────────────────────
    color_ranges = {
        "Blue Reference": [
            ([95, 60, 50], [135, 255, 255]),   # pure blue range
        ],
        "Red Reference": [
            ([0,   80, 50], [10,  255, 255]),   # lower red hue
            ([165, 80, 50], [180, 255, 255]),   # upper red hue
        ],
        "Green Reference": [
            ([40, 60, 50], [90, 255, 255]),     # pure green range
        ],
    }
    
    # ── 4. Find all potential regions (both reference strips and pads) ───────
    all_regions = []
    
    # First, find reference strips by their specific colors
    for name, ranges in color_ranges.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological cleanup
        kernel_size = max(3, min(img.shape[0], img.shape[1]) // 150)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best match for this reference color
        best_region = None
        best_score = -1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_strip_area or area > max_strip_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Reference strips should be rectangular (width >= 2x height OR height >= 2x width)
            is_horizontal_rect = aspect_ratio >= 2.0
            is_vertical_rect = (1.0 / aspect_ratio) >= 2.0 if aspect_ratio > 0 else False
            
            if not (is_horizontal_rect or is_vertical_rect):
                continue
            
            # Check if roughly rectangular
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) < 4:
                continue
            
            # Score based on aspect ratio (prefer horizontal for dipsticks)
            target_ar = 2.5  # Ideal aspect ratio for reference strips
            if is_vertical_rect:
                aspect_score = max(0.0, 1.0 - abs((1.0/aspect_ratio) - target_ar) / target_ar)
            else:
                aspect_score = max(0.0, 1.0 - abs(aspect_ratio - target_ar) / target_ar)
            
            area_score = min(area / max_strip_area, 1.0)
            score = aspect_score * 0.6 + area_score * 0.4
            
            if score > best_score:
                best_score = score
                best_region = {
                    'name': name,
                    'type': 'reference',
                    'bounds': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': area,
                    'aspect_ratio': round(aspect_ratio, 3),
                    'score': round(best_score, 3),
                    'is_horizontal': is_horizontal_rect
                }
        
        if best_region:
            all_regions.append(best_region)
    
    # ── 5. Find test pads (square regions with black outlines) ───────────────
    # Create a mask for all dark/black outlines
    # Black outlines have low value (brightness) but can have any hue/saturation
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # Value <= 50 for dark areas
    
    outline_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Dilate to connect nearby outline pixels
    kernel = np.ones((5, 5), np.uint8)
    outline_mask = cv2.dilate(outline_mask, kernel, iterations=2)
    outline_mask = cv2.erode(outline_mask, kernel, iterations=1)
    
    # Find contours of the black outlines
    contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for square/rectangular regions that could be test pads
    potential_pads = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_pad_area or area > max_pad_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Test pads should be roughly square (aspect ratio between 0.8 and 1.2)
        is_square = 0.8 <= aspect_ratio <= 1.2
        
        if not is_square:
            continue
        
        # Check if contour is roughly rectangular
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) < 4:
            continue
        
        # Verify there's actually color inside (not just an empty outline)
        # Create mask for the interior
        interior_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(interior_mask, [contour], -1, 255, -1)  # Fill the contour
        
        # Check average saturation/value inside to ensure it's not just black
        interior_pixels = hsv[interior_mask == 255]
        if len(interior_pixels) > 0:
            avg_saturation = np.mean(interior_pixels[:, 1])
            avg_value = np.mean(interior_pixels[:, 2])
            
            # Should have some color (saturation > 30) and not be too dark
            if avg_saturation < 30 or avg_value < 40:
                continue  # Empty or too dark - likely not a test pad
        
        potential_pads.append({
            'type': 'test_pad',
            'bounds': (x, y, w, h),
            'center': (x + w//2, y + h//2),
            'area': area,
            'aspect_ratio': round(aspect_ratio, 3),
            'contour': contour
        })
    
    # Remove overlapping pads (keep the ones with best square aspect ratio)
    potential_pads.sort(key=lambda p: abs(p['aspect_ratio'] - 1.0))  # Prefer perfect squares
    
    unique_pads = []
    for pad in potential_pads:
        overlap = False
        px, py, pw, ph = pad['bounds']
        
        for existing in unique_pads:
            ex, ey, ew, eh = existing['bounds']
            # Check IoU (Intersection over Union)
            ix1, iy1 = max(px, ex), max(py, ey)
            ix2, iy2 = min(px + pw, ex + ew), min(py + ph, ey + eh)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = (pw * ph) + (ew * eh) - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.3:  # Significant overlap
                    overlap = True
                    break
        
        if not overlap and len(unique_pads) < 3:  # We only need 3 pads
            unique_pads.append(pad)
    
    # Sort pads by vertical position (top to bottom)
    unique_pads.sort(key=lambda p: p['center'][1])
    
    # Assign names to pads
    for i, pad in enumerate(unique_pads[:3]):
        pad['name'] = f"Pad {i+1}"
        all_regions.append(pad)
    
    # ── 6. Sort all regions top-to-bottom by center y-coordinate ─────────────
    all_regions.sort(key=lambda r: r['center'][1])
    
    # Verify we have all 6 regions
    expected_names = ["Blue Reference", "Red Reference", "Green Reference", "Pad 1", "Pad 2", "Pad 3"]
    found_names = [r['name'] for r in all_regions]
    
    success = len(all_regions) == 6
    
    if not success:
        print(f"Warning: Found {len(all_regions)}/6 regions. Found: {found_names}")
        print(f"Expected: {expected_names}")
    
    # ── 7. Extract color information for each region ─────────────────────────
    for region in all_regions:
        x, y, w, h = region['bounds']
        region_img = bgr[y:y+h, x:x+w]
        
        if region_img.size > 0:
            # Analyze the color inside the region
            hsv_region = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
            
            # Use median for robustness
            h_vals = hsv_region[:,:,0].flatten()
            s_vals = hsv_region[:,:,1].flatten()
            v_vals = hsv_region[:,:,2].flatten()
            
            region['color_hsv'] = (
                float(np.median(h_vals)),
                float(np.median(s_vals)),
                float(np.median(v_vals))
            )
            
            # Calculate average RGB
            rgb_region = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
            region['color_rgb'] = (
                float(np.median(rgb_region[:,:,0])),
                float(np.median(rgb_region[:,:,1])),
                float(np.median(rgb_region[:,:,2]))
            )
    
    # ── 8. Organize return data ──────────────────────────────────────────────
    return {
        'success': success,
        'regions': all_regions,
        'reference_strips': [r for r in all_regions if r['type'] == 'reference'],
        'test_pads': [r for r in all_regions if r['type'] == 'test_pad'],
        'order_verified': found_names == expected_names
    }

def visualize_detection_streamlit(image, detection_results: Dict):
    """
    Create visualization for Streamlit display (returns image array).
    """
    img = np.array(image)
    if img.shape[2] == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        display_img = img.copy()
    
    # Color coding
    colors = {
        'reference': (0, 255, 0),    # Green
        'test_pad': (255, 0, 0)      # Blue
    }
    
    for region in detection_results['regions']:
        x, y, w, h = region['bounds']
        color = colors[region['type']]
        
        # Draw bounding box
        cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 3)
        
        # Add label
        label = region['name']
        
        # Put text above bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Background for text
        cv2.rectangle(display_img, 
                     (x, y - text_size[1] - 8), 
                     (x + text_size[0] + 8, y - 3), 
                     color, -1)
        
        # Text
        cv2.putText(display_img, label, (x + 4, y - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    return cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
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
        st.caption("Detected regions (References: Green, Pads: Blue)")
        st.image(annotated)

    # Detection status badges for all 6 regions
    expected_regions = ["Blue Reference", "Red Reference", "Green Reference", "Pad 1", "Pad 2", "Pad 3"]
    found_names = [r['name'] for r in detection_results['regions']]
    
    badge_html = '<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0;">'
    for region in expected_regions:
        if region in found_names:
            badge_html += f'<span class="detect-badge badge-found">✓ {region}</span>'
        else:
            badge_html += f'<span class="detect-badge badge-missing">✗ {region}</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    if not detection_results['success']:
        st.warning(f"Only {len(detection_results['regions'])}/6 regions detected. "
                   "Try better lighting, reposition the dipstick, or ensure all regions have clear black outlines.")
        
        # Show which regions are missing
        missing = set(expected_regions) - set(found_names)
        if missing:
            st.info(f"Missing: {', '.join(missing)}")
    else:
        # Display detailed info for detected regions
        st.caption("### Detected Regions (top to bottom)")
        
        for region in detection_results['regions']:
            region_type = "Reference" if region['type'] == 'reference' else "Test Pad"
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{region['name']}** *({region_type})*")
            with col2:
                st.write(f"Aspect: {region['aspect_ratio']:.2f}")
            with col3:
                st.write(f"Area: {region['area']:.0f} px")
        
        # Option to show raw detection data in expander
        with st.expander("🔍 Show detection details"):
            st.json({
                'success': detection_results['success'],
                'num_regions': len(detection_results['regions']),
                'references': [r['name'] for r in detection_results['reference_strips']],
                'test_pads': [r['name'] for r in detection_results['test_pads']],
                'order_verified': detection_results['order_verified']
            })

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 3: Analyze ───────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">Step 03 — Analyze</div>', unsafe_allow_html=True)

    if st.button("🔬  Run Full Analysis", use_container_width=True, disabled=(not detection_results['success'])):
        with st.spinner("Measuring pad colors and calculating concentrations…"):
            # Extract test pads (should be Pad 1, Pad 2, Pad 3)
            test_pads = detection_results['test_pads']
            
            # Map pads to expected analytes (adjust based on your dipstick layout)
            # Assuming Pad 1 = Albumin, Pad 2 = Creatinine, Pad 3 = pH/others
            pad_colors = {}
            for pad in test_pads:
                pad_name = pad['name']
                pad_colors[pad_name] = pad['color_rgb']
            
            # For backward compatibility with existing functions
            # Map to expected keys if needed
            colors = {
                "Albumin": pad_colors.get("Pad 1", np.array([128,128,128])),
                "Creatinine": pad_colors.get("Pad 2", np.array([128,128,128])),
                "Pad 3": pad_colors.get("Pad 3", np.array([128,128,128]))
            }
            
            # Get reference strips for calibration
            refs = {}
            for ref in detection_results['reference_strips']:
                ref_name = ref['name'].replace(' Reference', '')
                refs[ref_name] = {
                    'center': ref['center'],
                    'bounds': ref['bounds'],
                    'area': ref['area'],
                    'aspect_ratio': ref['aspect_ratio'],
                    'score': 1.0  # Already scored in detection
                }
            
            
            # Store additional pad info
            pad_3_color = colors.get("Pad 3")
            pad_3_category = None
            if pad_3_color is not None:
                # You might want to classify pad 3 (e.g., pH, glucose, etc.)
                pad_3_category = "Analyzed"
            
            st.session_state.results = {
                "uacr": uacr,
                "albumin": alb_conc,
                "creatinine": creat_conc,
                "colors": colors,
                "refs": refs,
                "detection_results": detection_results,
                "annotated": annotated,
                "pad_3_category": pad_3_category,
            }
            st.rerun()

    if not detection_results['success']:
        st.caption("⚠️ Analysis disabled until all 6 regions are detected.")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results

    st.markdown('<div class="card"><div class="card-label">Results</div>', unsafe_allow_html=True)

    # Annotated image (re-show in results section)
    if "annotated" in res:
        st.image(res["annotated"], caption="Detected regions (Green: References, Blue: Test Pads)", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Albumin", f"{res['albumin']:.0f} mg/dL")
    c2.metric("Creatinine", f"{res['creatinine']:.0f} mg/dL")
    c3.metric("uACR", f"{res['uacr']:.1f} mg/g")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Color swatches for all detected pads
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
        
        # Show Pad 3 interpretation if available
        if res.get("pad_3_category"):
            st.info(f"Pad 3 analysis: {res['pad_3_category']}")

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
    
    # Add detection quality note
    if res['detection_results']['order_verified']:
        st.success("✓ All regions detected in correct order (top to bottom)")
    else:
        st.warning("⚠️ Regions detected but order may be incorrect. Check visualization.")

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
  <li>Ensure all <strong>3 reference strips</strong> (Blue, Red, Green) are in frame</li>
  <li>Ensure all <strong>3 test pads</strong> (square regions with black outlines) are visible</li>
  <li>Avoid glare — diffuse natural light works best</li>
  <li>Hold camera <strong>directly above</strong> the strip, not at an angle</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Research prototype — not for clinical use. Consult a healthcare provider for medical decisions.")