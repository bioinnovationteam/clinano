import streamlit as st
import numpy as np
from PIL import Image, ImageStat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import colorsys
import math

# Page configuration - optimized for mobile
st.set_page_config(
    page_title="uACR Dipstick Analyzer",
    page_icon="🧪",
    layout="centered",  # Changed to centered for better mobile view
    initial_sidebar_state="collapsed"  # Collapsed sidebar on mobile
)

# Custom CSS for white/blue theme and mobile optimization with improved contrast
st.markdown("""
    <style>
    /* Main container padding for mobile */
    .main > div {
        padding: 0 0.5rem;
    }
    
    /* Base text styles for better contrast */
    .stMarkdown, p, li, .stText {
        color: #1A1A1A !important;  /* Dark gray for better contrast */
        font-size: 1rem;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #003366 !important;  /* Darker blue for headers */
        font-weight: 600;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2rem;
        color: #003366;  /* Darker blue for better contrast */
        text-align: center;
        margin: 1rem 0 1.5rem 0;
        padding: 0.5rem;
        background: linear-gradient(135deg, #E6F0FF 0%, #FFFFFF 100%);
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,51,102,0.1);
        font-weight: 700;
    }
    
    /* Info box - white with blue border and dark text */
    .info-box {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #0066CC;
        box-shadow: 0 2px 8px rgba(0,102,204,0.08);
    }
    
    .info-box p, .info-box li, .info-box div {
        color: #1A1A1A !important;
    }
    
    /* Warning box - white with yellow border */
    .warning-box {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 8px rgba(255,193,7,0.1);
    }
    
    .warning-box p, .warning-box li {
        color: #1A1A1A !important;
    }
    
    /* Success box - white with green border */
    .success-box {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 8px rgba(40,167,69,0.1);
    }
    
    .success-box p, .success-box li {
        color: #1A1A1A !important;
    }
    
    /* Fact cards - white with blue accents and dark text */
    .fact-card {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,51,102,0.1);
        margin: 0.8rem 0;
        border: 1px solid #CCE0FF;
        transition: transform 0.2s;
    }
    
    .fact-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,51,102,0.15);
    }
    
    .fact-card h3 {
        color: #003366 !important;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    .fact-card p {
        color: #1A1A1A !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066CC 0%, #004C99 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(0,102,204,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #004C99 0%, #003366 100%);
        box-shadow: 0 6px 16px rgba(0,51,102,0.3);
        transform: translateY(-2px);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #F5F9FF;
        padding: 0.8rem;
        border-radius: 25px;
        border: 1px solid #CCE0FF;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 1rem;
        justify-content: center;
    }
    
    .stRadio label {
        color: #1A1A1A !important;
        font-weight: 500;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F5F9FF;
        border-radius: 10px;
        color: #003366 !important;
        font-weight: 600;
        border: 1px solid #CCE0FF;
    }
    
    .streamlit-expanderContent {
        background-color: #FFFFFF;
        border-radius: 0 0 10px 10px;
        border: 1px solid #CCE0FF;
        border-top: none;
        padding: 1rem;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,51,102,0.05);
        border: 1px solid #CCE0FF;
    }
    
    .stMetric label {
        color: #003366 !important;
        font-weight: 600;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #1A1A1A !important;
        font-weight: 700;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #F5F9FF;
        padding: 0.5rem;
        border-radius: 30px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px;
        padding: 0.5rem 1rem;
        color: #1A1A1A !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066CC !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* Slider styling */
    .stSlider label {
        color: #003366 !important;
        font-weight: 600;
    }
    
    .stSlider div[data-baseweb="slider"] {
        background-color: #F5F9FF;
    }
    
    /* Number input styling */
    .stNumberInput label {
        color: #003366 !important;
        font-weight: 600;
    }
    
    .stNumberInput input {
        background-color: #FFFFFF;
        border: 1px solid #CCE0FF;
        border-radius: 8px;
        color: #1A1A1A !important;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #003366 !important;
        font-weight: 600;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF;
        border: 1px solid #CCE0FF;
        border-radius: 8px;
    }
    
    /* Divider styling */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #0066CC, transparent);
    }
    
    /* Image caption */
    .stImage caption {
        color: #003366;
        font-weight: 500;
        text-align: center;
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        .fact-card {
            padding: 1rem;
        }
        
        .stButton > button {
            padding: 0.6rem 1rem;
            font-size: 0.95rem;
        }
        
        h3 {
            font-size: 1.2rem;
        }
        
        p, li {
            font-size: 0.95rem;
        }
    }
    
    /* Custom class for blue text */
    .blue-text {
        color: #003366;
        font-weight: 600;
    }
    
    /* Card container for results */
    .result-card {
        background: linear-gradient(135deg, #F8FCFF 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #CCE0FF;
        margin: 1rem 0;
    }
    
    .result-card p, .result-card div {
        color: #1A1A1A !important;
    }
    
    /* Color swatch styling */
    .color-swatch {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        border: 2px solid #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0 auto;
    }
    
    /* Ensure all text has good contrast */
    .stAlert p {
        color: #1A1A1A !important;
    }
    
    .stException {
        color: #1A1A1A !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        color: #1A1A1A !important;
    }
    
    .dataframe th {
        background-color: #E6F0FF !important;
        color: #003366 !important;
        font-weight: 600;
    }
    
    /* Camera input styling */
    .stCameraInput {
        border: 2px dashed #CCE0FF;
        border-radius: 15px;
        background-color: #F8FCFF;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #CCE0FF;
        border-radius: 15px;
        padding: 1rem;
        background-color: #F8FCFF;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = "analyzer"  # Default view

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s * 100, v * 100

def check_image_quality(image):
    """
    Check if image meets quality requirements using PIL
    Returns: (bool, str, dict) - (passed, message, metrics)
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        # Simple grayscale conversion
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Calculate brightness (mean of grayscale)
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation of grayscale)
    contrast = np.std(gray)
    
    # Calculate sharpness using simple gradient method
    # Horizontal gradient
    grad_x = np.diff(gray, axis=1)
    # Vertical gradient
    grad_y = np.diff(gray, axis=0)
    # Ensure arrays have same shape by trimming
    min_shape = min(grad_x.shape[0], grad_y.shape[0])
    grad_x = grad_x[:min_shape, :]
    grad_y = grad_y[:, :min_shape]
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    sharpness = np.mean(gradient_magnitude)
    
    metrics = {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'sharpness': float(sharpness)
    }
    
    # Quality thresholds
    brightness_ok = 80 <= brightness <= 200
    contrast_ok = contrast > 20
    sharpness_ok = sharpness > 10
    
    if not brightness_ok:
        return False, "Image too dark or too bright. Please ensure proper lighting.", metrics
    elif not contrast_ok:
        return False, "Image contrast too low. Please ensure good lighting conditions.", metrics
    elif not sharpness_ok:
        return False, "Image is blurry. Please hold the camera steady.", metrics
    
    return True, "Image quality is good!", metrics

def measure_pad_colors(image, pad_positions):
    """
    Measure RGB and HSV values at specified pad positions
    pad_positions: dict with pad names and (x,y) coordinates relative to image
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    results = {}
    
    for pad_name, (x, y) in pad_positions.items():
        # Convert relative coordinates to absolute if needed
        if isinstance(x, float) and x <= 1.0:
            x = int(x * width)
        if isinstance(y, float) and y <= 1.0:
            y = int(y * height)
        
        # Sample a small region around the point
        region_size = 20
        x1 = max(0, x - region_size)
        x2 = min(width, x + region_size)
        y1 = max(0, y - region_size)
        y2 = min(height, y + region_size)
        
        region = img_array[y1:y2, x1:x2]
        
        # Calculate mean RGB
        mean_rgb = np.mean(region, axis=(0,1))
        
        # Convert to HSV
        r, g, b = mean_rgb
        h, s, v = rgb_to_hsv(r, g, b)
        
        results[pad_name] = {
            'RGB': mean_rgb,
            'HSV': (h, s, v)
        }
    
    return results

def calculate_egfr(age, creatinine_value, gender='male'):
    """
    Calculate eGFR using CKD-EPI equation (simplified)
    Note: This is for demonstration - in production use actual serum creatinine
    """
    # Placeholder creatinine value (in reality, this would come from the dipstick)
    # This is just for demonstration
    if gender == 'male':
        scr = 0.9  # placeholder
        kappa = 0.9
        alpha = -0.411
    else:
        scr = 0.7  # placeholder
        kappa = 0.7
        alpha = -0.329
    
    # CKD-EPI equation
    egfr = 141 * (min(scr/kappa, 1))**alpha * (max(scr/kappa, 1))**-1.209 * 0.993**age
    
    if gender == 'female':
        egfr *= 1.018
    
    return round(egfr, 1)

def get_uacr_category(uacr_value):
    """Categorize uACR value"""
    if uacr_value < 30:
        return "Normal to mildly increased", "#28a745"
    elif uacr_value < 300:
        return "Moderately increased", "#ffc107"
    else:
        return "Severely increased", "#dc3545"

def display_kidney_facts():
    """Display educational content about kidney disease"""
    
    facts = [
        {
            "title": "What is uACR?",
            "content": "Urine Albumin-to-Creatinine Ratio measures the amount of albumin (a protein) in your urine. It's a key indicator of kidney damage.",
            "icon": "🔬"
        },
        {
            "title": "Why It Matters",
            "content": "Early detection of kidney disease can slow progression and prevent complications. Regular screening is crucial for at-risk individuals.",
            "icon": "💡"
        },
        {
            "title": "Risk Factors",
            "content": "Diabetes, hypertension, family history, age >60, cardiovascular disease, and obesity are major risk factors for kidney disease.",
            "icon": "⚠️"
        },
        {
            "title": "Lifestyle Tips",
            "content": "Maintain healthy blood pressure, control blood sugar, exercise regularly, stay hydrated, and avoid excessive NSAIDs.",
            "icon": "💪"
        }
    ]
    
    for fact in facts:
        st.markdown(f"""
        <div class="fact-card">
            <h3>{fact['icon']} {fact['title']}</h3>
            <p>{fact['content']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header with blue gradient
    st.markdown('<h1 class="main-header">🧪 uACR Dipstick Analyzer</h1>', unsafe_allow_html=True)
    
    # Simple inline navigation with pills (without radio buttons)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📸 Analyzer", use_container_width=True, 
                    type="primary" if st.session_state.current_view == "analyzer" else "secondary"):
            st.session_state.current_view = "analyzer"
            st.rerun()
    
    with col2:
        if st.button("📚 Education", use_container_width=True,
                    type="primary" if st.session_state.current_view == "education" else "secondary"):
            st.session_state.current_view = "education"
            st.rerun()
    
    with col3:
        if st.button("📋 Guidelines", use_container_width=True,
                    type="primary" if st.session_state.current_view == "guidelines" else "secondary"):
            st.session_state.current_view = "guidelines"
            st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Show the selected view
    if st.session_state.current_view == "analyzer":
        analyzer_page()
    elif st.session_state.current_view == "education":
        education_page()
    elif st.session_state.current_view == "guidelines":
        guidelines_page()

def analyzer_page():
    # Mobile-optimized single column layout
    st.subheader("📸 Image Capture")
    
    # Image input method
    input_method = st.radio(
        "Choose input method:",
        ["📷 Take Photo", "📁 Upload Image"],
        horizontal=True
    )
    
    image = None
    
    if input_method == "📷 Take Photo":
        img_file = st.camera_input("Take a photo of your dipstick")
        if img_file is not None:
            image = Image.open(img_file)
            st.session_state.image_captured = image
    else:
        img_file = st.file_uploader("Upload dipstick image", type=['jpg', 'jpeg', 'png'])
        if img_file is not None:
            image = Image.open(img_file)
            st.session_state.image_captured = image
    
    if image is not None:
        # Display image
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Check image quality
        quality_passed, quality_msg, metrics = check_image_quality(image)
        
        if quality_passed:
            st.markdown(f'<div class="success-box">✅ {quality_msg}</div>', unsafe_allow_html=True)
            
            # Show image metrics
            with st.expander("📊 Image Quality Metrics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Brightness", f"{metrics['brightness']:.1f}")
                col2.metric("Contrast", f"{metrics['contrast']:.1f}")
                col3.metric("Sharpness", f"{metrics['sharpness']:.1f}")
            
            # Let user define pad positions
            st.markdown("### 🎯 Define Pad Positions")
            st.markdown("Adjust the sliders to match your dipstick")
            
            # Use relative coordinates for flexibility
            pad_positions = {}
            
            # Pad 1: Albumin
            st.markdown("**Albumin Pad**")
            col1, col2 = st.columns(2)
            with col1:
                albumin_x = st.slider("X position (%)", 0, 100, 30, key="alb_x") / 100
            with col2:
                albumin_y = st.slider("Y position (%)", 0, 100, 40, key="alb_y") / 100
            pad_positions['Albumin'] = (albumin_x, albumin_y)
            
            # Pad 2: Creatinine
            st.markdown("**Creatinine Pad**")
            col1, col2 = st.columns(2)
            with col1:
                creat_x = st.slider("X position (%)", 0, 100, 50, key="creat_x") / 100
            with col2:
                creat_y = st.slider("Y position (%)", 0, 100, 40, key="creat_y") / 100
            pad_positions['Creatinine'] = (creat_x, creat_y)
            
            # Pad 3: Control
            st.markdown("**Control Pad**")
            col1, col2 = st.columns(2)
            with col1:
                control_x = st.slider("X position (%)", 0, 100, 70, key="ctrl_x") / 100
            with col2:
                control_y = st.slider("Y position (%)", 0, 100, 40, key="ctrl_y") / 100
            pad_positions['Control'] = (control_x, control_y)
            
            if st.button("🔍 Analyze Dipstick", type="primary"):
                with st.spinner("Analyzing colors..."):
                    # Measure colors
                    color_results = measure_pad_colors(image, pad_positions)
                    st.session_state.color_results = color_results
                    
                    # Placeholder uACR calculation
                    albumin_rgb = color_results['Albumin']['RGB']
                    
                    # Simple algorithm for demonstration
                    albumin_score = np.mean(albumin_rgb) / 255
                    
                    # Generate uACR value based on color (placeholder logic)
                    uacr_value = albumin_score * 300 + np.random.uniform(-20, 20)
                    uacr_value = max(5, min(1000, uacr_value))
                    
                    st.session_state.uacr_value = uacr_value
                    st.session_state.analysis_complete = True
                    
                    st.success("Analysis complete!")
                    st.rerun()
        else:
            st.markdown(f'<div class="warning-box">⚠️ {quality_msg}</div>', unsafe_allow_html=True)
            
            # Tips for better image
            with st.expander("💡 Tips for better images"):
                st.markdown("""
                - Ensure even lighting
                - Avoid shadows on the dipstick
                - Hold camera steady
                - Keep dipstick in focus
                - Place on a neutral background
                """)
    
    # Results section
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Results card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Color measurements
        with st.expander("🎨 Color Measurements", expanded=True):
            for pad, values in st.session_state.color_results.items():
                rgb = values['RGB']
                hsv = values['HSV']
                
                st.markdown(f"**{pad}**")
                col1, col2, col3 = st.columns([2, 2, 1])
                
                # RGB display
                color_hex = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0]), int(rgb[1]), int(rgb[2])
                )
                
                col1.markdown(f"RGB: {int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}")
                col2.markdown(f"HSV: {hsv[0]:.1f}°, {hsv[1]:.1f}%, {hsv[2]:.1f}%")
                col3.markdown(f"<div style='background-color:{color_hex}; width:40px; height:40px; border-radius:10px; border:2px solid #0066CC;'></div>", 
                            unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Demographics input
        st.subheader("👤 Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=45)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Calculate eGFR
        egfr = calculate_egfr(age, None, gender.lower())
        
        # Display uACR and eGFR
        uacr = st.session_state.uacr_value
        category, cat_color = get_uacr_category(uacr)
        
        # Metrics display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("uACR", f"{uacr:.1f} mg/g")
        with col2:
            st.metric("eGFR", f"{egfr} mL/min/1.73m²")
        
        st.markdown(f"**Category:** <span style='color:{cat_color}; font-weight:600;'>{category}</span>", 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interpretation
        st.markdown("### 📋 Interpretation")
        if uacr < 30:
            st.success("✅ **Normal to mildly increased albuminuria**")
            st.markdown('<p style="color: #1A1A1A;">Your uACR is within normal range. Continue regular monitoring as recommended.</p>', unsafe_allow_html=True)
        elif uacr < 300:
            st.warning("⚠️ **Moderately increased albuminuria**")
            st.markdown('<p style="color: #1A1A1A;">This may indicate early kidney damage. Please consult your healthcare provider for further evaluation.</p>', unsafe_allow_html=True)
        else:
            st.error("🔴 **Severely increased albuminuria**")
            st.markdown('<p style="color: #1A1A1A;">This suggests significant kidney damage. Please consult your healthcare provider immediately.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Results"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                result_entry = {
                    'timestamp': timestamp,
                    'uacr': uacr,
                    'egfr': egfr,
                    'category': category,
                    'age': age,
                    'gender': gender
                }
                
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append(result_entry)
                st.success("Results saved!")
        
        with col2:
            if st.button("📄 Download Report"):
                report = f"""
uACR Dipstick Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RESULTS:
uACR: {uacr:.1f} mg/g
eGFR: {egfr} mL/min/1.73m²
Category: {category}

DEMOGRAPHICS:
Age: {age}
Gender: {gender}

COLOR MEASUREMENTS:
"""
                for pad, values in st.session_state.color_results.items():
                    rgb = values['RGB']
                    report += f"{pad}: RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})\n"
                
                report += """
DISCLAIMER:
This is a screening tool only. Please consult with a healthcare professional for proper medical advice.
"""
                
                st.download_button(
                    label="Download",
                    data=report,
                    file_name=f"uACR_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        # Show instructions
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 📱 How to use:
        1. **Take or upload** a clear photo of your dipstick
        2. **Adjust the sliders** to mark each test pad
        3. **Click Analyze** to measure colors
        4. **Enter your age and gender**
        5. **View your results** and save for tracking
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display quick facts
        st.subheader("📚 Quick Facts")
        display_kidney_facts()

def education_page():
    st.subheader("📚 Understanding Kidney Health")
    
    tabs = st.tabs(["uACR", "eGFR", "Risk Factors", "Prevention"])
    
    with tabs[0]:
        st.markdown("""
        ### What is uACR?
        
        **Urine Albumin-to-Creatinine Ratio** measures protein (albumin) in your urine.
        
        #### Your Results:
        - 🟢 **<30 mg/g**: Normal
        - 🟡 **30-300 mg/g**: Moderate increase
        - 🔴 **>300 mg/g**: Severe increase
        
        Early detection can slow kidney disease progression.
        """)
        
        # Visual representation
        fig = go.Figure(data=[
            go.Bar(name='Normal', x=['<30 mg/g'], y=[30], marker_color='#28a745'),
            go.Bar(name='Moderate', x=['30-300 mg/g'], y=[270], marker_color='#ffc107'),
            go.Bar(name='Severe', x=['>300 mg/g'], y=[100], marker_color='#dc3545')
        ])
        fig.update_layout(
            title="uACR Categories",
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#003366')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("""
        ### What is eGFR?
        
        **Estimated Glomerular Filtration Rate** shows kidney filtering ability.
        
        | Stage | eGFR | Status |
        |-------|------|--------|
        | 1 | ≥90 | Normal |
        | 2 | 60-89 | Mild decrease |
        | 3 | 30-59 | Moderate decrease |
        | 4 | 15-29 | Severe decrease |
        | 5 | <15 | Kidney failure |
        """)
    
    with tabs[2]:
        st.markdown("""
        ### Risk Factors
        
        #### Major Risks:
        - 🩺 **Diabetes** - Leading cause
        - ❤️ **High blood pressure**
        - 👴 **Age >60**
        - 🧬 **Family history**
        - ⚖️ **Obesity**
        - 🚬 **Smoking**
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Prevention Tips
        
        #### Healthy Habits:
        - 🥗 **Low sodium diet**
        - 💧 **Stay hydrated**
        - 🏃 **Regular exercise**
        - 📊 **Monitor blood pressure**
        - 🩺 **Annual check-ups**
        - 💊 **Limit NSAIDs**
        """)

def guidelines_page():
    st.subheader("📋 Clinical Guidelines")
    
    st.markdown("""
    ### KDIGO 2024 Guidelines
    
    #### Screening Frequency:
    
    | Risk Level | Testing |
    |------------|---------|
    | Low Risk | Every 3-5 years |
    | Moderate Risk | Annually |
    | High Risk | Every 6-12 months |
    
    #### When to See a Doctor:
    - 🔴 **eGFR <30** - Severe decrease in kidney function
    - 🔴 **uACR >300** - Severely increased albuminuria
    - 📈 **Rapid decline** - eGFR drops quickly over time
    - 💊 **Uncontrolled BP** - Blood pressure remains high
    
    ---
    
    > ⚠️ **Disclaimer**: This is a screening tool only. Always consult healthcare providers for medical decisions.
    """)

if __name__ == "__main__":
    main()