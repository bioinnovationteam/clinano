import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import colorsys

st.set_page_config(page_title="uACR Analyzer", page_icon="🧪", layout="centered")

# CSS styling
st.markdown("""
    <style>
    .stButton > button { background-color: #0066CC !important; color: white !important; border-radius: 25px !important; }
    h1, h2, h3 { color: #0066CC !important; }
    .info-box { background-color: #E6F3FF; border-left: 5px solid #0066CC; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .result-card { background-color: #E6F3FF; border: 2px solid #0066CC; padding: 20px; border-radius: 15px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state properly
if 'results' not in st.session_state:
    st.session_state.results = None

def rgb_to_hsv(r,g,b):
    r,g,b = r/255.0, g/255.0, b/255.0
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return h*360, s*100, v*100

def find_reference_strips(image):
    """Find RGB reference strips"""
    img = np.array(image)
    refs = {}
    
    # Target colors (white, gray, black)
    targets = {'white': (255,255,255), 'gray': (128,128,128), 'black': (32,32,32)}
    
    for name, target in targets.items():
        # Find closest matching pixel region
        distances = np.sqrt(sum((img - target[i])**2 for i in range(3)))
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        refs[name] = (min_idx[1], min_idx[0])  # x, y
    
    return refs

def estimate_pads(refs):
    """Estimate pad positions from references"""
    white = refs['white']
    gray = refs['gray']
    
    # Calculate distance and direction
    dx = gray[0] - white[0]
    dy = gray[1] - white[1]
    
    # Pads are offset from white reference
    pads = {
        'Albumin': (white[0] + dx, white[1] + dy//2),
        'Creatinine': (white[0] + dx*2, white[1] + dy//2),
        'pH': (white[0] + dx*3, white[1] + dy//2)
    }
    return pads

def measure_colors(image, positions):
    """Measure RGB at positions"""
    img = np.array(image)
    h,w = img.shape[:2]
    colors = {}
    
    for name, (x,y) in positions.items():
        x, y = int(x), int(y)
        # Sample 20x20 region
        x1, x2 = max(0,x-10), min(w,x+10)
        y1, y2 = max(0,y-10), min(h,y+10)
        rgb = np.mean(img[y1:y2, x1:x2], axis=(0,1))
        colors[name] = rgb
    
    return colors

def rgb_to_conc(rgb, biomarker):
    """Simple RGB to concentration mapping"""
    intensity = np.mean(rgb)
    
    if biomarker == 'albumin':
        # Intensity 255->0, 100->300 mg/dL
        conc = (255 - intensity) * (300/155)
    else:  # creatinine
        # Intensity 255->0, 150->200 mg/dL
        conc = (255 - intensity) * (200/105)
    
    return max(0, min(conc, 500))

st.title("uACR Dipstick Analyzer")

# Image input
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("📸 Step 1: Take Photo")

# Fixed: Added label to radio button
method = st.radio("Image input method", ["Take Photo", "Upload"], horizontal=True, label_visibility="collapsed")
image = None

if method == "Take Photo":
    img_file = st.camera_input("Take a photo of the dipstick")
else:
    img_file = st.file_uploader("Upload an image", type=['jpg','png'])

if img_file:
    image = Image.open(img_file)
    st.image(image, use_container_width=True)
    
    if st.button("🔬 Analyze", use_container_width=True):
        with st.spinner("Analyzing..."):
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
    st.subheader("📊 Results")
    
    res = st.session_state.results
    uacr = res['uacr']
    
    # Color display
    for name, rgb in res['colors'].items():
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        st.markdown(f"**{name}** RGB: {int(rgb[0])},{int(rgb[1])},{int(rgb[2])} "
                   f"<span style='background:{hex_color}; display:inline-block; width:20px; height:20px; border-radius:3px;'></span>",
                   unsafe_allow_html=True)
    
    st.divider()
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Albumin", f"{res['albumin']:.1f} mg/dL")
    col2.metric("Creatinine", f"{res['creatinine']:.1f} mg/dL")
    st.metric("uACR", f"{uacr:.1f} mg/g")
    
    # Interpretation
    if uacr < 30:
        st.success("✅ Normal")
        st.caption("uACR <30 mg/g - Normal range")
    elif uacr < 300:
        st.warning("⚠️ Moderate Increase")
        st.caption("30-300 mg/g - Microalbuminuria")
    else:
        st.error("🔴 Severe Increase")
        st.caption(">300 mg/g - Macroalbuminuria")
    
    # New analysis button
    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state.results = None
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("⚠️ Prototype only. Consult healthcare provider for medical advice.")