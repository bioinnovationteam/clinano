import streamlit as st
import numpy as np
from PIL import Image
import colorsys

# FORCE LIGHT MODE - MUST BE FIRST
st.set_page_config(
    page_title="uACR Analyzer", 
    page_icon="🧪", 
    layout="centered",
    initial_sidebar_state="auto"
)

# Inject CSS to force light mode
st.markdown("""
    <style>
        /* Force white background everywhere */
        .stApp {
            background: white !important;
        }
        
        /* Override Streamlit's default dark mode */
        .stApp > header {
            background-color: white !important;
        }
        
        /* Make sure all text is dark */
        * {
            color: #000000 !important;
        }
        
        /* Keep your blue buttons */
        .stButton > button {
            background-color: #0066CC !important;
            color: white !important;
        }
        
        /* Keep info boxes light blue */
        .info-box, .result-card {
            background-color: #E6F3FF !important;
        }
    </style>
""", unsafe_allow_html=True)

# CSS styling
st.markdown("""
    <style>
    .stButton > button { background-color: #0066CC !important; color: white !important; border-radius: 25px !important; }
    h1, h2, h3 { color: #0066CC !important; }
    .info-box { background-color: #E6F3FF; border-left: 5px solid #0066CC; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .result-card { background-color: #E6F3FF; border: 2px solid #0066CC; padding: 20px; border-radius: 15px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Back end

def load_image_for_processing(image_input):
    """
    Universal image loader that handles HEIC, file paths, URLs, and numpy arrays
    
    Args:
        image_input: Can be:
            - Path to HEIC file (string)
            - Path to JPG/PNG file (string)
            - URL (string starting with http)
            - numpy array (already loaded image)
            - PIL Image object
    
    Returns:
        RGB numpy array ready for find_reference_strips()
    """
    import requests
    from urllib.parse import urlparse
    
    # Case 1: Already a numpy array
    if isinstance(image_input, np.ndarray):
        # Assume it's RGB, if BGR convert
        if image_input.shape[-1] == 3:
            return image_input
        else:
            raise ValueError(f"Unexpected array shape: {image_input.shape}")
    
    # Case 2: PIL Image
    if hasattr(image_input, 'mode') and hasattr(image_input, 'convert'):
        return np.array(image_input.convert('RGB'))
    
    # Case 3: String (path or URL)
    if isinstance(image_input, str):
        # Check if URL
        parsed = urlparse(image_input)
        if parsed.scheme in ('http', 'https'):
            # Download from URL
            response = requests.get(image_input, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'heic' in content_type or image_input.lower().endswith('.heic'):
                # Save to temp file and convert
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                
                try:
                    result = convert_heic_to_readable(tmp_path)
                    return result
                finally:
                    import os
                    os.unlink(tmp_path)
            else:
                # Regular image from URL
                from PIL import Image
                img = Image.open(response.raw)
                return np.array(img.convert('RGB'))
        
        else:
            # Local file path
            if image_input.lower().endswith(('.heic', '.heif')):
                return convert_heic_to_readable(image_input)
            else:
                # Regular image format
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Could not read image: {image_input}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def batch_process_heic_files(folder_path, output_folder=None):
    """
    Batch convert all HEIC files in a folder to JPG for testing
    
    Args:
        folder_path: Folder containing HEIC files
        output_folder: Where to save JPGs (default: same folder)
    """
    import glob
    import os
    from pathlib import Path
    
    if output_folder is None:
        output_folder = folder_path
    
    os.makedirs(output_folder, exist_ok=True)
    
    heic_files = glob.glob(os.path.join(folder_path, "*.heic")) + \
                 glob.glob(os.path.join(folder_path, "*.HEIC"))
    
    results = []
    
    for heic_path in heic_files:
        try:
            # Convert to numpy array
            img_rgb = convert_heic_to_readable(heic_path)
            
            # Save as JPG for easy viewing
            stem = Path(heic_path).stem
            jpg_path = os.path.join(output_folder, f"{stem}.jpg")
            
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(jpg_path, img_bgr)
            
            results.append({'file': heic_path, 'status': 'success', 'output': jpg_path})
            print(f"✓ Converted: {Path(heic_path).name} -> {Path(jpg_path).name}")
            
        except Exception as e:
            results.append({'file': heic_path, 'status': 'failed', 'error': str(e)})
            print(f"✗ Failed: {Path(heic_path).name} - {e}")
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nConverted {success_count}/{len(heic_files)} files")
    
    return results

def find_reference_strips(image):
    """Find RGB reference strips with adaptive thresholds and robust detection"""
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Adaptive thresholds based on image statistics
    img_area = img.shape[0] * img.shape[1]
    min_strip_area = img_area * 0.001  # 0.1% of image area
    max_strip_area = img_area * 0.2    # 20% of image area
    
    # More permissive ranges (can be made adaptive)
    color_ranges = {
        'Red': [
            ([0, 30, 30], [10, 255, 255]),      # Lower saturation threshold
            ([160, 30, 30], [180, 255, 255])    # Extended range
        ],
        'Green': [([35, 30, 30], [85, 255, 255])],
        'Blue': [([100, 30, 30], [130, 255, 255])]
    }
    
    refs = {}
    
    for name, ranges in color_ranges.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Dynamic kernel size based on image dimensions
        kernel_size = max(3, min(img.shape[0], img.shape[1]) // 200)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and collect all valid contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_strip_area or area > max_strip_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # More permissive aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 < aspect_ratio < 3.0:  # Much wider range
                # Check if contour is roughly rectangular
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                if len(approx) >= 4:  # At least quadrilateral
                    valid_contours.append((area, x, y, w, h, contour))
        
        if valid_contours:
            # Sort by area and take best candidate
            valid_contours.sort(key=lambda c: c[0], reverse=True)
            
            # Try top 3 contours, pick one with best aspect ratio
            best = None
            best_score = -1
            for area, x, y, w, h, contour in valid_contours[:3]:
                # Score based on area and aspect ratio
                aspect_score = 1.0 - min(abs(1.0 - w/h), 1.0)
                area_score = area / max_strip_area
                score = aspect_score * 0.6 + area_score * 0.4
                
                if score > best_score:
                    best_score = score
                    best = (area, x, y, w, h, contour)
            
            if best:
                area, x, y, w, h, contour = best
                refs[name] = {
                    'center': (x + w//2, y + h//2),
                    'bounds': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': w/h,
                    'score': best_score
                }
    
    # More robust validation
    if len(refs) < 3:
        missing = set(['Blue', 'Red', 'Green']) - set(refs.keys())
        print(f"Error: Missing strips for colors: {missing}")
        return refs  # Return what we found, let caller handle
    
    # Validate ordering with tolerance
    ordered = sorted(refs.items(), key=lambda item: item[1]['center'][1])
    detected_order = [name for name, _ in ordered]
    
    expected_order = ['Blue', 'Red', 'Green']
    if detected_order != expected_order:
        print(f"Warning: Order mismatch. Found: {detected_order}, Expected: {expected_order}")
        # Could attempt to re-map based on expected order
    
    # Use relative tolerance for alignment
    x_centers = [refs[name]['center'][0] for name in refs]
    x_range = max(x_centers) - min(x_centers)
    img_width = img.shape[1]
    relative_x_range = x_range / img_width
    
    if relative_x_range > 0.1:  # Centers vary more than 10% of image width
        print(f"Warning: Strips not well-aligned (X range: {relative_x_range:.1%} of width)")
    
    return refs

def estimate_pads(refs):
    """Estimate pad positions from references"""
    white = refs['white']
    gray = refs['gray']
    
    dx = gray[0] - white[0]
    dy = gray[1] - white[1]
    
    pads = {
        'Albumin': (white[0] + dx, white[1] + dy//2),
        'Creatinine': (white[0] + dx*2, white[1] + dy//2),
        'pH': (white[0] + dx*3, white[1] + dy//2)
    }
    return pads

def measure_colors(image, positions):
    """Measure RGB at positions"""
    img = np.array(image)
    h, w = img.shape[:2]
    colors = {}
    
    for name, (x, y) in positions.items():
        x, y = int(x), int(y)
        x1, x2 = max(0, x-10), min(w, x+10)
        y1, y2 = max(0, y-10), min(h, y+10)
        rgb = np.mean(img[y1:y2, x1:x2], axis=(0,1))
        colors[name] = rgb
    
    return colors

def rgb_to_conc(rgb, biomarker):
    """RGB to concentration mapping"""
    intensity = np.mean(rgb)
    
    if biomarker == 'albumin':
        conc = (255 - intensity) * (300/155)  # Maps 255→0, 100→300 mg/dL
    else:
        conc = (255 - intensity) * (200/105)  # Maps 255→0, 150→200 mg/dL
    
    return max(0, min(conc, 500))

# App UI
st.title("🧪 uACR Dipstick Analyzer")
st.markdown("*Demo: Automatic detection with 3-point reference calibration*")

# Image input
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("📸 Step 1: Capture or Upload")

method = st.radio("", ["Take Photo", "Upload"], horizontal=True, label_visibility="collapsed")
image = None

if method == "Take Photo":
    img_file = st.camera_input("Position dipstick with all 3 reference strips visible")
else:
    img_file = st.file_uploader("Upload dipstick image", type=['jpg', 'png'])

if img_file:
    image = Image.open(img_file)
    st.image(image, use_container_width=True)
    
    if st.button("🔬 Analyze Dipstick", use_container_width=True):
        with st.spinner("Detecting reference strips and analyzing..."):
            # Find references
            refs = find_reference_strips(image)
            
            # Estimate pad positions
            pads = estimate_pads(refs)
            
            # Measure colors
            colors = measure_colors(image, pads)
            
            # Calculate concentrations
            alb_conc = rgb_to_conc(colors['Albumin'], 'albumin')
            creat_conc = rgb_to_conc(colors['Creatinine'], 'creatinine')
            uacr = alb_conc / (creat_conc / 100) if creat_conc > 0 else 0
            
            # Store results
            st.session_state.results = {
                'uacr': uacr,
                'albumin': alb_conc,
                'creatinine': creat_conc,
                'colors': colors
            }
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Display results
if st.session_state.results is not None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Results")
    
    res = st.session_state.results
    uacr = res['uacr']
    
    # Show measured colors
    st.markdown("**Detected Colors:**")
    cols = st.columns(3)
    for idx, (name, rgb) in enumerate(res['colors'].items()):
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        with cols[idx]:
            st.markdown(f"**{name}**")
            st.markdown(f"RGB: {int(rgb[0])},{int(rgb[1])},{int(rgb[2])}")
            st.markdown(f"<div style='background:{hex_color}; width:50px; height:50px; border-radius:5px; border:2px solid #0066CC; margin:auto;'></div>", 
                       unsafe_allow_html=True)
    
    st.divider()
    
    # Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Albumin", f"{res['albumin']:.0f} mg/dL")
    col2.metric("Creatinine", f"{res['creatinine']:.0f} mg/dL")
    col3.metric("uACR", f"{uacr:.1f} mg/g")
    
    st.divider()
    
    # Interpretation
    if uacr < 30:
        st.success("### ✅ Normal")
        st.caption("uACR < 30 mg/g - Within normal range")
    elif uacr < 300:
        st.warning("### ⚠️ Moderately Increased")
        st.caption("uACR 30-300 mg/g - Microalbuminuria")
    else:
        st.error("### 🔴 Severely Increased")
        st.caption("uACR > 300 mg/g - Macroalbuminuria")
    
    # New analysis button
    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state.results = None
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Demo instructions footer
st.markdown("---")
st.markdown("""
<div class="info-box">
<b>📋 Demo Instructions:</b><br>
1. Your dipstick must have <b>white, gray, and black</b> reference strips<br>
2. Position dipstick on flat surface with good lighting<br>
3. Take photo showing all 3 reference strips clearly<br>
4. Click analyze to see uACR result
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Demo prototype - Not for clinical use. Consult healthcare provider for medical advice.")