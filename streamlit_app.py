from colorsys import rgb_to_hsv  

import streamlit as st
import numpy as np
from PIL import Image 
import cv2

# ── Page config ────────────────────────────────────────────────────────────
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

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #F5F4F0 !important;
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 760px !important; }

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

.stButton > button {
    background: #1A1A1A !important;
    color: #FFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.55rem 1.2rem !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.82 !important; }
.stButton > button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

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
    font-size: 1.35rem !important;
    color: #1A1A1A !important;
}

.status-normal   { background:#EDFAF1; border-left:3px solid #27AE60; border-radius:6px; padding:0.8rem 1rem; margin-bottom: 1.5rem; }
.status-moderate { background:#FFF8E6; border-left:3px solid #F39C12; border-radius:6px; padding:0.8rem 1rem; margin-bottom: 1.5rem; }
.status-severe   { background:#FEECEC; border-left:3px solid #E74C3C; border-radius:6px; padding:0.8rem 1rem; margin-bottom: 1.5rem; }
.status-label    { font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:500; }
.status-sub      { font-size:0.83rem; color:#555; margin-top:0.3rem; }

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

.results-main-display {
    text-align: center;
    padding: 1.5rem 0 1.0rem 0;
    margin: 1rem 0;
}
.results-main-value {
    font-family: 'DM Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    color: #1A1A1A;
    line-height: 1;
}
.results-main-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

.processing-screen {
    background: #FFFFFF;
    border: 1px solid #E2E0D8;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}
.processing-title {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
    color: #1A1A1A;
}
.processing-sub {
    color: #666;
    font-size: 0.92rem;
}

p, span, div, label, li { color: #1A1A1A; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False

if "page" not in st.session_state:
    st.session_state.page = "capture"

if "use_normalization" not in st.session_state:
    st.session_state.use_normalization = False


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Reference color targets (ideal values)
REFERENCE_TARGETS = {
    "blue": {"hex": "#5170ff", "rgb": (81, 112, 255)},
    "red": {"hex": "#ff3131", "rgb": (255, 49, 49)},
    "green": {"hex": "#00bf63", "rgb": (0, 191, 99)}
}


def rgb2hsv_matlab_exact(rgb):
    """
    EXACT MATLAB rgb2hsv implementation
    """
    if isinstance(rgb, (list, tuple, np.ndarray)):
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        r, g, b = rgb, rgb, rgb
    
    # Normalize to 0-1
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    if max_val == 0:
        s = 0
    else:
        s = delta / max_val
    
    # Hue - MATLAB algorithm
    if delta == 0:
        h = 0
    elif max_val == r:
        h = 60 * (((g - b) / delta) % 6)
    elif max_val == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    
    # Normalize to 0-1
    h = h / 360.0
    
    return h, s, v


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

def crop_region(image_rgb: np.ndarray, bounds, inner_fraction: float = 0.80) -> np.ndarray:
    x, y, w, h = bounds

    inset_x = int(w * (1 - inner_fraction) / 2)
    inset_y = int(h * (1 - inner_fraction) / 2)

    x1 = max(0, x + inset_x)
    y1 = max(0, y + inset_y)
    x2 = min(image_rgb.shape[1], x + w - inset_x)
    y2 = min(image_rgb.shape[0], y + h - inset_y)

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = x, y, x + w, y + h

    return image_rgb[y1:y2, x1:x2].copy()


def extract_mean_rgb_from_crop(crop_rgb: np.ndarray, white_threshold: int = 240) -> dict:
    """Extract mean RGB from a cropped region"""
    if crop_rgb.size == 0:
        return {"mean_rgb": (np.nan, np.nan, np.nan), "num_pixels": 0}
    
    R = crop_rgb[:, :, 0].astype(np.float64)
    G = crop_rgb[:, :, 1].astype(np.float64)
    B = crop_rgb[:, :, 2].astype(np.float64)
    
    # Glare filtering
    glare = (R > white_threshold) & (G > white_threshold) & (B > white_threshold)
    mask = ~glare
    
    if np.sum(mask) == 0:
        mean_rgb = (np.mean(R), np.mean(G), np.mean(B))
        return {"mean_rgb": mean_rgb, "num_pixels": len(R.flatten())}
    
    mean_rgb = (np.mean(R[mask]), np.mean(G[mask]), np.mean(B[mask]))
    
    return {
        "mean_rgb": (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])),
        "num_pixels": int(np.sum(mask))
    }


def calculate_correction_factors(measured_blue_rgb, measured_red_rgb, measured_green_rgb):
    """Calculate RGB correction factors based on measured reference colors"""
    blue_target = np.array(REFERENCE_TARGETS["blue"]["rgb"], dtype=np.float64)
    red_target = np.array(REFERENCE_TARGETS["red"]["rgb"], dtype=np.float64)
    green_target = np.array(REFERENCE_TARGETS["green"]["rgb"], dtype=np.float64)
    
    measured_blue = np.array(measured_blue_rgb, dtype=np.float64)
    measured_red = np.array(measured_red_rgb, dtype=np.float64)
    measured_green = np.array(measured_green_rgb, dtype=np.float64)
    
    r_factors = []
    g_factors = []
    b_factors = []
    
    # Red reference
    if measured_red[0] > 0:
        r_factors.append(red_target[0] / measured_red[0])
    if measured_red[1] > 0:
        g_factors.append(red_target[1] / measured_red[1])
    if measured_red[2] > 0:
        b_factors.append(red_target[2] / measured_red[2])
    
    # Green reference
    if measured_green[0] > 0:
        r_factors.append(green_target[0] / measured_green[0])
    if measured_green[1] > 0:
        g_factors.append(green_target[1] / measured_green[1])
    if measured_green[2] > 0:
        b_factors.append(green_target[2] / measured_green[2])
    
    # Blue reference
    if measured_blue[0] > 0:
        r_factors.append(blue_target[0] / measured_blue[0])
    if measured_blue[1] > 0:
        g_factors.append(blue_target[1] / measured_blue[1])
    if measured_blue[2] > 0:
        b_factors.append(blue_target[2] / measured_blue[2])
    
    correction = {
        'R': np.clip(np.median(r_factors), 0.7, 1.3) if r_factors else 1.0,
        'G': np.clip(np.median(g_factors), 0.7, 1.3) if g_factors else 1.0,
        'B': np.clip(np.median(b_factors), 0.7, 1.3) if b_factors else 1.0
    }
    
    return correction


def normalize_color(rgb, correction_factors):
    """Apply correction factors to a measured color"""
    r, g, b = rgb
    
    r_norm = r * correction_factors['R']
    g_norm = g * correction_factors['G']
    b_norm = b * correction_factors['B']
    
    r_norm = np.clip(r_norm, 0, 255)
    g_norm = np.clip(g_norm, 0, 255)
    b_norm = np.clip(b_norm, 0, 255)
    
    return (int(r_norm), int(g_norm), int(b_norm))


def extract_mean_hue_from_crop_unified(crop_rgb: np.ndarray,
                                        white_threshold: int = 240,
                                        correction_factors: dict = None) -> dict:
    """Unified hue extraction with optional normalization"""
    if crop_rgb.size == 0:
        return {
            "mean_hue": np.nan,
            "mean_hue_deg": np.nan,
            "num_pixels": 0,
        }
    
    # Apply normalization if correction factors provided
    if correction_factors is not None:
        normalized_crop = np.zeros_like(crop_rgb)
        for i in range(crop_rgb.shape[0]):
            for j in range(crop_rgb.shape[1]):
                r, g, b = crop_rgb[i, j]
                normalized_crop[i, j] = normalize_color((r, g, b), correction_factors)
        analysis_crop = normalized_crop
    else:
        analysis_crop = crop_rgb
    
    R = analysis_crop[:, :, 0].astype(np.float64)
    G = analysis_crop[:, :, 1].astype(np.float64)
    B = analysis_crop[:, :, 2].astype(np.float64)
    
    # Glare filtering
    glare = (R > white_threshold) & (G > white_threshold) & (B > white_threshold)
    mask = ~glare
    
    # Calculate hue for each pixel
    hue_vals = []
    
    for i in range(analysis_crop.shape[0]):
        for j in range(analysis_crop.shape[1]):
            if mask[i, j]:
                rgb_pixel = np.array([R[i, j], G[i, j], B[i, j]])
                h, s, v = rgb2hsv_matlab_exact(rgb_pixel)
                hue_vals.append(h)
    
    # Fallback to all pixels if none passed filter
    if len(hue_vals) == 0:
        for i in range(analysis_crop.shape[0]):
            for j in range(analysis_crop.shape[1]):
                rgb_pixel = np.array([R[i, j], G[i, j], B[i, j]])
                h, s, v = rgb2hsv_matlab_exact(rgb_pixel)
                hue_vals.append(h)
    
    if len(hue_vals) == 0:
        return {
            "mean_hue": np.nan,
            "mean_hue_deg": np.nan,
            "num_pixels": 0,
        }
    
    mean_hue = float(np.mean(hue_vals))
    
    return {
        "mean_hue": mean_hue,
        "mean_hue_deg": mean_hue * 360.0,
        "num_pixels": len(hue_vals),
    }


def calculate_ph_from_hue_pchip(hue_norm: float) -> float:
    """Calculate pH from normalized hue using PCHIP interpolation"""
    CAL_PH = np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    CAL_HUE_DEG = np.array([0.6, 21.7, 33.4, 49.8, 73.8, 179.6, 223.2, 273.9], dtype=float)
    CAL_HUE_NORM = CAL_HUE_DEG / 360.0
    
    hue_norm = hue_norm % 1.0
    
    if hue_norm <= CAL_HUE_NORM[0]:
        return float(CAL_PH[0])
    elif hue_norm >= CAL_HUE_NORM[-1]:
        return float(CAL_PH[-1])
    
    return float(np.interp(hue_norm, CAL_HUE_NORM, CAL_PH))


def calculate_albumin(hue: float, pH: float) -> float:
    slope = -0.000165 * pH**3 + 0.002833 * pH**2 - 0.015671 * pH + 0.029497
    intercept = -0.000775 * pH**2 + 0.011425 * pH + 0.096394

    if abs(slope) < 1e-12 or not np.isfinite(slope):
        return np.nan

    albumin_mg_dL = (hue - intercept) / slope
    return float(max(albumin_mg_dL, 0.0))


def calculate_creatinine(hue: float, pH: float) -> float:
    a = 6.035287e-07 * pH**2 - 7.148671e-06 * pH + 2.274549e-05
    b = -6.621931e-05 * pH**2 + 7.645101e-04 * pH - 2.754278e-03
    c = -4.421482e-04 * pH**2 + 5.842984e-03 * pH + 1.457634e-01

    m_lin = -1.33453750e-05 * pH**2 + 1.50491050e-04 * pH - 7.33072700e-04
    b_lin = -5.44922489e-04 * pH**2 + 6.68320571e-03 * pH + 1.41714753e-01

    coeffs = [a, b, c - hue]
    roots_all = np.roots(coeffs)

    real_roots = []
    for r in roots_all:
        if abs(np.imag(r)) < 1e-10 and np.real(r) >= 0:
            real_roots.append(float(np.real(r)))

    if real_roots:
        c_quad = min(real_roots)
        if c_quad <= 100:
            return float(c_quad)

    if abs(m_lin) < 1e-12 or not np.isfinite(m_lin):
        return np.nan

    creatinine_mg_dL = (hue - b_lin) / m_lin
    return float(max(creatinine_mg_dL, 0.0))


def calculate_acr(albumin_mg_dL: float, creatinine_mg_dL: float) -> float:
    if not np.isfinite(albumin_mg_dL) or not np.isfinite(creatinine_mg_dL):
        return np.nan
    if creatinine_mg_dL <= 0:
        return np.nan

    creatinine_g_dL = creatinine_mg_dL / 1000.0
    return float(albumin_mg_dL / creatinine_g_dL)


def classify_acr(acr_mg_g: float):
    if not np.isfinite(acr_mg_g):
        return ("Invalid result", "status-severe")
    if acr_mg_g < 30:
        return ("Normal to mildly increased", "status-normal")
    if acr_mg_g < 300:
        return ("Moderately increased", "status-moderate")
    return ("Severely increased", "status-severe")


def run_full_analysis(image_rgb: np.ndarray, detection_results: dict, use_normalization: bool = False) -> dict:
    regions = detection_results["regions"][:6]
    
    # Find indices for each region type
    region_map = {}
    for i, region in enumerate(regions):
        region_map[region["name"]] = i
    
    # Extract regions by name
    blue_idx = region_map.get("Blue Ref", 0)
    red_idx = region_map.get("Red Ref", 1)
    green_idx = region_map.get("Green Ref", 2)
    albumin_idx = region_map.get("Albumin", 3)
    creatinine_idx = region_map.get("Creatinine", 4)
    ph_idx = region_map.get("pH", 5)
    
    # Extract cropped regions
    albumin_crop = crop_region(image_rgb, regions[albumin_idx]["bounds"], inner_fraction=0.80)
    creatinine_crop = crop_region(image_rgb, regions[creatinine_idx]["bounds"], inner_fraction=0.80)
    ph_crop = crop_region(image_rgb, regions[ph_idx]["bounds"], inner_fraction=0.80)
    
    # Calculate correction factors if normalization is enabled
    correction_factors = None
    normalization_info = None
    
    if use_normalization:
        blue_ref_crop = crop_region(image_rgb, regions[blue_idx]["bounds"], inner_fraction=0.80)
        red_ref_crop = crop_region(image_rgb, regions[red_idx]["bounds"], inner_fraction=0.80)
        green_ref_crop = crop_region(image_rgb, regions[green_idx]["bounds"], inner_fraction=0.80)
        
        blue_rgb = extract_mean_rgb_from_crop(blue_ref_crop)
        red_rgb = extract_mean_rgb_from_crop(red_ref_crop)
        green_rgb = extract_mean_rgb_from_crop(green_ref_crop)
        
        if (blue_rgb["num_pixels"] > 0 and red_rgb["num_pixels"] > 0 and green_rgb["num_pixels"] > 0):
            correction_factors = calculate_correction_factors(
                blue_rgb["mean_rgb"],
                red_rgb["mean_rgb"],
                green_rgb["mean_rgb"]
            )
            
            normalization_info = {
                "blue_measured": blue_rgb["mean_rgb"],
                "red_measured": red_rgb["mean_rgb"],
                "green_measured": green_rgb["mean_rgb"],
                "correction_factors": correction_factors
            }
    
    # Extract hues with optional normalization
    ph_stats = extract_mean_hue_from_crop_unified(ph_crop, white_threshold=240, correction_factors=correction_factors)
    albumin_stats = extract_mean_hue_from_crop_unified(albumin_crop, white_threshold=240, correction_factors=correction_factors)
    creatinine_stats = extract_mean_hue_from_crop_unified(creatinine_crop, white_threshold=240, correction_factors=correction_factors)
    
    # Calculate pH
    estimated_pH = calculate_ph_from_hue_pchip(ph_stats["mean_hue"])
    
    # Calculate concentrations
    albumin_mg_dL = calculate_albumin(albumin_stats["mean_hue"], estimated_pH)
    creatinine_mg_dL = calculate_creatinine(creatinine_stats["mean_hue"], estimated_pH)
    acr_mg_g = calculate_acr(albumin_mg_dL, creatinine_mg_dL)
    
    category, category_class = classify_acr(acr_mg_g)
    
    return {
        "estimated_pH": estimated_pH,
        "albumin_mg_dL": albumin_mg_dL,
        "creatinine_mg_dL": creatinine_mg_dL,
        "acr_mg_g": acr_mg_g,
        "category": category,
        "category_class": category_class,
        "normalization_used": use_normalization,
        "normalization_info": normalization_info,
        "debug": {
            "ph_hue_norm": ph_stats["mean_hue"],
            "ph_hue_deg": ph_stats["mean_hue_deg"],
            "ph_num_pixels": ph_stats["num_pixels"],
            "albumin_hue": albumin_stats["mean_hue"],
            "albumin_hue_deg": albumin_stats["mean_hue_deg"],
            "creatinine_hue": creatinine_stats["mean_hue"],
            "creatinine_hue_deg": creatinine_stats["mean_hue_deg"],
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="page-title">uACR Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Dipstick · 6-Point Detection (3 References + 3 Test Pads)</p>', unsafe_allow_html=True)

# ── Capture page ───────────────────────────────────────────────────────────
if st.session_state.page == "capture":
    st.markdown('<div class="card"><div class="card-label">Step 01 — Capture</div>', unsafe_allow_html=True)

    method = st.radio("", ["📷  Camera", "📁  Upload"], horizontal=True, label_visibility="collapsed")
    img_file = None

    if "Camera" in method:
        img_file = st.camera_input(
            "Position dipstick so all regions are visible",
            label_visibility="visible"
        )
    else:
        img_file = st.file_uploader(
            "Upload dipstick image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if img_file and not st.session_state.is_analyzing:
        image = Image.open(img_file).convert("RGB")

        st.markdown('<div class="card"><div class="card-label">Step 02 — Detect Regions</div>', unsafe_allow_html=True)

        col_orig, col_ann = st.columns(2)

        with col_orig:
            st.caption("Original")
            st.image(image)

        detection_results = detect_dipstick_regions(image)
        annotated = visualize_detection_streamlit(image, detection_results)

        with col_ann:
            st.caption("Detected regions")
            st.image(annotated)

        # Show detected regions count
        region_count = len(detection_results["regions"])
        if region_count >= 6:
            st.success(f"✅ Detected 6/6 regions")
        else:
            st.warning(f"⚠️ Only detected {region_count}/6 regions. Please ensure good lighting and contrast.")

        # Show region details
        with st.expander("Detected Regions Detail"):
            for i, region in enumerate(detection_results["regions"]):
                st.write(f"{i+1}. {region['name']}: position {region['center']}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Step 03 — Normalization (Optional)</div>', unsafe_allow_html=True)
        
        # Normalization toggle
        st.session_state.use_normalization = st.checkbox(
            "Enable color normalization using reference patches", 
            value=False,
            help="Uses the blue, red, and green reference patches to correct for lighting variations"
        )
        
        if st.session_state.use_normalization:
            st.info("📊 Color normalization will be applied using the detected reference patches")
            st.markdown("**Reference targets:**")
            st.markdown(f"- Blue: `{REFERENCE_TARGETS['blue']['hex']}` → RGB{REFERENCE_TARGETS['blue']['rgb']}")
            st.markdown(f"- Red: `{REFERENCE_TARGETS['red']['hex']}` → RGB{REFERENCE_TARGETS['red']['rgb']}")
            st.markdown(f"- Green: `{REFERENCE_TARGETS['green']['hex']}` → RGB{REFERENCE_TARGETS['green']['rgb']}")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Step 04 — Analyze</div>', unsafe_allow_html=True)

        analyze_disabled = not detection_results["success"] and len(detection_results["regions"]) < 6
        
        if st.button("🔬 Run Full Analysis", use_container_width=True, disabled=analyze_disabled):
            st.session_state.is_analyzing = True
            st.session_state.results = None
            st.session_state.page = "processing"
            st.session_state.uploaded_image = image
            st.rerun()

        if analyze_disabled:
            st.caption(f"⚠️ Analysis requires 6 regions. Only {len(detection_results['regions'])} detected.")

        st.markdown("</div>", unsafe_allow_html=True)

# ── Processing page ────────────────────────────────────────────────────────
if st.session_state.page == "processing":
    st.markdown(
        """
        <div class="processing-screen">
            <div class="processing-title">Calculating ACR…</div>
            <div class="processing-sub">
                Estimating pH, albumin, creatinine, and final ratio.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Analyzing..."):
        if "uploaded_image" in st.session_state and st.session_state.uploaded_image is not None:
            image = st.session_state.uploaded_image
            image_rgb = np.array(image)
            detection_results = detect_dipstick_regions(image)
            st.session_state.results = run_full_analysis(
                image_rgb, 
                detection_results, 
                use_normalization=st.session_state.use_normalization
            )
            st.session_state.is_analyzing = False
            st.session_state.page = "results"
            st.rerun()
        else:
            st.session_state.is_analyzing = False
            st.session_state.page = "capture"
            st.rerun()

# ── Results page ───────────────────────────────────────────────────────────
if st.session_state.page == "results" and st.session_state.results is not None:
    r = st.session_state.results

    st.markdown('<div class="card"><div class="card-label">Results</div>', unsafe_allow_html=True)

    # Show normalization status
    if r.get("normalization_used", False):
        st.success("✅ Color normalization applied using reference patches")
    
    st.markdown(
        f"""
        <div class="{r['category_class']}">
            <div class="status-label">{r['category']}</div>
            <div class="status-sub">Calculated albumin-to-creatinine ratio.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="results-main-display">
            <div class="results-main-value">{r['acr_mg_g']:.1f}</div>
            <div class="results-main-unit">mg/g ACR</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Estimated pH", f"{r['estimated_pH']:.2f}")
        st.metric("Albumin", f"{r['albumin_mg_dL']:.2f} mg/dL")
    with c2:
        st.metric("Creatinine", f"{r['creatinine_mg_dL']:.2f} mg/dL")
        st.metric("ACR", f"{r['acr_mg_g']:.1f} mg/g")

    # Show normalization details if used
    if r.get("normalization_used", False) and r.get("normalization_info"):
        norm_info = r["normalization_info"]
        with st.expander("Color Normalization Details"):
            st.write("**Measured Reference Colors:**")
            st.write(f"Blue reference:  RGB{norm_info['blue_measured']}")
            st.write(f"Red reference:   RGB{norm_info['red_measured']}")
            st.write(f"Green reference: RGB{norm_info['green_measured']}")
            
            st.write("**Correction Factors Applied:**")
            st.write(f"Red channel (R):   {norm_info['correction_factors']['R']:.4f}")
            st.write(f"Green channel (G): {norm_info['correction_factors']['G']:.4f}")
            st.write(f"Blue channel (B):  {norm_info['correction_factors']['B']:.4f}")
            
            st.write("**Target Reference Colors:**")
            st.write(f"Blue target:  {REFERENCE_TARGETS['blue']['hex']} → RGB{REFERENCE_TARGETS['blue']['rgb']}")
            st.write(f"Red target:   {REFERENCE_TARGETS['red']['hex']} → RGB{REFERENCE_TARGETS['red']['rgb']}")
            st.write(f"Green target: {REFERENCE_TARGETS['green']['hex']} → RGB{REFERENCE_TARGETS['green']['rgb']}")

    # Show debug info
    with st.expander("Debug Info (Hue Values)"):
        if "debug" in r:
            st.write("**pH Pad:**")
            st.write(f"Mean Hue: {r['debug']['ph_hue_deg']:.2f}°")
            st.write(f"Pixels analyzed: {r['debug']['ph_num_pixels']}")
            st.write(f"Calculated pH: {r['estimated_pH']:.2f}")
            
            st.write("**Albumin Pad:**")
            st.write(f"Mean Hue: {r['debug']['albumin_hue_deg']:.2f}° (0-1: {r['debug']['albumin_hue']:.4f})")
            
            st.write("**Creatinine Pad:**")
            st.write(f"Mean Hue: {r['debug']['creatinine_hue_deg']:.2f}° (0-1: {r['debug']['creatinine_hue']:.4f})")
        
        st.write("**pH Calibration Table:**")
        cal_data = {
            "pH": [3, 4, 5, 6, 7, 8, 9, 10],
            "Hue (deg)": [0.6, 21.7, 33.4, 49.8, 73.8, 179.6, 223.2, 273.9],
        }
        st.dataframe(cal_data)

    if st.button("Analyze Another Image", use_container_width=True):
        st.session_state.results = None
        st.session_state.is_analyzing = False
        st.session_state.page = "capture"
        st.session_state.uploaded_image = None
        st.rerun()

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
  <li><strong>Enable normalization</strong> for consistent results across different lighting conditions</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Research prototype — not for clinical use.")