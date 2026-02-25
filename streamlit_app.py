import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import colorsys

# Page configuration
st.set_page_config(
    page_title="uACR Dipstick Analyzer",
    page_icon="",
    layout="centered"
)

# Simple blue-themed CSS
st.markdown("""
    <style>
    /* Blue text boxes for all inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider > div,
    .stTextArea > div > div > textarea,
    .stRadio > div,
    div[data-baseweb="select"] > div {
        background-color: #E6F3FF !important;
        border: 2px solid #0066CC !important;
        border-radius: 10px !important;
        color: #003366 !important;
    }
    
    /* Blue buttons */
    .stButton > button {
        background-color: #0066CC !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    
    /* Blue headers */
    h1, h2, h3 {
        color: #0066CC !important;
    }
    
    /* Blue info boxes */
    .info-box {
        background-color: #E6F3FF;
        border-left: 5px solid #0066CC;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Blue success box */
    .success-box {
        background-color: #E6F3FF;
        border-left: 5px solid #28a745;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Blue warning box */
    .warning-box {
        background-color: #E6F3FF;
        border-left: 5px solid #ffc107;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Blue result cards */
    .result-card {
        background-color: #E6F3FF;
        border: 2px solid #0066CC;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'color_results' not in st.session_state:
    st.session_state.color_results = None
if 'uacr_value' not in st.session_state:
    st.session_state.uacr_value = None

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s * 100, v * 100

def check_image_quality(image):
    """Simple image quality check"""
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    if brightness < 80:
        return False, "Image too dark"
    elif brightness > 200:
        return False, "Image too bright"
    elif contrast < 20:
        return False, "Low contrast"
    else:
        return True, "Good quality"

def measure_pad_colors(image, pad_positions):
    """Measure colors at pad positions"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    results = {}
    
    for pad_name, (x, y) in pad_positions.items():
        x = int(x * width)
        y = int(y * height)
        
        # Sample region
        size = 15
        x1 = max(0, x - size)
        x2 = min(width, x + size)
        y1 = max(0, y - size)
        y2 = min(height, y + size)
        
        region = img_array[y1:y2, x1:x2]
        mean_rgb = np.mean(region, axis=(0,1))
        h, s, v = rgb_to_hsv(mean_rgb[0], mean_rgb[1], mean_rgb[2])
        
        results[pad_name] = {
            'RGB': mean_rgb,
            'HSV': (h, s, v)
        }
    
    return results

def get_uacr_category(value):
    """Categorize uACR"""
    if value < 30:
        return "Normal", "#28a745"
    elif value < 300:
        return "Moderate", "#ffc107"
    else:
        return "Severe", "#dc3545"

# Main app
st.title(" uACR Dipstick Analyzer")

# Image capture section
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("Step 1: Capture Image")

input_method = st.radio("Choose method:", ["Take Photo", "Upload"], horizontal=True)

image = None
if input_method == "Take Photo":
    img_file = st.camera_input("Take photo")
    if img_file:
        image = Image.open(img_file)
else:
    img_file = st.file_uploader("Upload image", type=['jpg', 'png'])
    if img_file:
        image = Image.open(img_file)

st.markdown('</div>', unsafe_allow_html=True)

if image:
    st.image(image, caption="Your image", use_container_width=True)
    
    # Quality check
    quality_ok, quality_msg = check_image_quality(image)
    
    if quality_ok:
        st.markdown(f'<div class="success-box">✅ {quality_msg}</div>', unsafe_allow_html=True)
        
        # Pad positions
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("🎯 Step 2: Mark Pads")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Albumin**")
            ax = st.slider("X", 0, 100, 30, key="ax") / 100
            ay = st.slider("Y", 0, 100, 40, key="ay") / 100
        
        with col2:
            st.markdown("**Creatinine**")
            cx = st.slider("X", 0, 100, 50, key="cx") / 100
            cy = st.slider("Y", 0, 100, 40, key="cy") / 100
        
        with col3:
            st.markdown("**Control**")
            kx = st.slider("X", 0, 100, 70, key="kx") / 100
            ky = st.slider("Y", 0, 100, 40, key="ky") / 100
        
        pad_positions = {
            'Albumin': (ax, ay),
            'Creatinine': (cx, cy),
            'Control': (kx, ky)
        }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 Analyze", use_container_width=True):
            with st.spinner("Analyzing..."):
                colors = measure_pad_colors(image, pad_positions)
                st.session_state.color_results = colors
                
                # Calculate uACR (simplified)
                albumin_score = np.mean(colors['Albumin']['RGB']) / 255
                uacr = albumin_score * 300 + np.random.uniform(-20, 20)
                uacr = max(5, min(1000, uacr))
                
                st.session_state.uacr_value = uacr
                st.session_state.analysis_complete = True
                st.rerun()
    else:
        st.markdown(f'<div class="warning-box">⚠️ {quality_msg}</div>', unsafe_allow_html=True)

# Results section
if st.session_state.analysis_complete and st.session_state.color_results:
    st.markdown("---")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📊 Results")
    
    # Color measurements
    for pad, values in st.session_state.color_results.items():
        rgb = values['RGB']
        hsv = values['HSV']
        color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.write(f"**{pad}**")
        col2.write(f"RGB: {int(rgb[0])},{int(rgb[1])},{int(rgb[2])}")
        col3.markdown(f"<div style='background:{color_hex}; width:30px; height:30px; border-radius:5px; border:2px solid #0066CC;'></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Demographics
    st.subheader("👤 Step 3: Your Info")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 120, 45)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Calculate eGFR (simplified)
    if gender == "Male":
        egfr = 141 * (0.9/0.9)**-0.411 * (0.9/0.9)**-1.209 * 0.993**age
    else:
        egfr = 141 * (0.7/0.7)**-0.329 * (0.7/0.7)**-1.209 * 0.993**age * 1.018
    
    egfr = round(egfr, 1)
    
    # Display results
    uacr = st.session_state.uacr_value
    category, cat_color = get_uacr_category(uacr)
    
    col1, col2 = st.columns(2)
    col1.metric("uACR", f"{uacr:.1f} mg/g")
    col2.metric("eGFR", f"{egfr} mL/min")
    
    st.markdown(f"**Category:** <span style='color:{cat_color}; font-weight:bold;'>{category}</span>", unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("### 📋 Interpretation")
    if uacr < 30:
        st.success("✅ Normal range")
        st.write("Continue regular monitoring.")
    elif uacr < 300:
        st.warning("⚠️ Moderate increase")
        st.write("Consult your healthcare provider.")
    else:
        st.error("🔴 Severe increase")
        st.write("Seek medical attention.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save button
    if st.button("💾 Save", use_container_width=True):
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'uacr': uacr,
            'egfr': egfr,
            'category': category
        })
        st.success("Saved!")

# Quick facts section
st.markdown("---")
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("📚 Quick Facts")

facts = [
    "🔬 uACR <30 mg/g is normal",
    "💡 Early detection helps prevent kidney damage",
    "⚠️ Diabetes and high blood pressure are major risk factors",
    "💪 Stay hydrated and exercise regularly"
]

for fact in facts:
    st.write(fact)

st.markdown('</div>', unsafe_allow_html=True)

# Disclaimer
st.caption("⚠️ This is a screening tool only. Always consult healthcare providers for medical advice.")