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

def find_reference_strips(image):
    """Find white, gray, black reference strips"""
    img = np.array(image)
    refs = {}
    
    targets = {'white': (255,255,255), 'gray': (128,128,128), 'black': (32,32,32)}
    
    for name, target in targets.items():
        distances = np.sqrt(sum((img - target[i])**2 for i in range(3)))
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        refs[name] = (min_idx[1], min_idx[0])
    
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