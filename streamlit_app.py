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

# Page configuration
st.set_page_config(
    page_title="uACR Dipstick Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #013220;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #013220;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #013220;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #013220;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fact-card {
        background-color: black;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = None
if 'history' not in st.session_state:
    st.session_state.history = []

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
    
    cols = st.columns(2)
    
    for i, fact in enumerate(facts):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="fact-card">
                <h3>{fact['icon']} {fact['title']}</h3>
                <p>{fact['content']}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 uACR Dipstick Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Analyzer", "Education", "Guidelines", "History"])
    
    if page == "Analyzer":
        analyzer_page()
    elif page == "Education":
        education_page()
    elif page == "Guidelines":
        guidelines_page()
    elif page == "History":
        history_page()

def analyzer_page():
    # Create two columns for layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📸 Image Capture")
        
        # Image input method
        input_method = st.radio(
            "Choose input method:",
            ["Take Photo", "Upload Image"],
            horizontal=True
        )
        
        image = None
        
        if input_method == "Take Photo":
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
                with st.expander("Image Quality Metrics"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Brightness", f"{metrics['brightness']:.1f}")
                    col2.metric("Contrast", f"{metrics['contrast']:.1f}")
                    col3.metric("Sharpness", f"{metrics['sharpness']:.1f}")
                
                # Let user define pad positions
                st.markdown("### Define Pad Positions")
                st.markdown("Click on the image to mark the center of each pad")
                
                # Use relative coordinates for flexibility
                pad_positions = {}
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Albumin Pad**")
                    albumin_x = st.slider("Albumin X position (%)", 0, 100, 30) / 100
                    albumin_y = st.slider("Albumin Y position (%)", 0, 100, 40) / 100
                    pad_positions['Albumin'] = (albumin_x, albumin_y)
                
                with col2:
                    st.markdown("**Creatinine Pad**")
                    creat_x = st.slider("Creatinine X position (%)", 0, 100, 50) / 100
                    creat_y = st.slider("Creatinine Y position (%)", 0, 100, 40) / 100
                    pad_positions['Creatinine'] = (creat_x, creat_y)
                
                with col3:
                    st.markdown("**Control Pad**")
                    control_x = st.slider("Control X position (%)", 0, 100, 70) / 100
                    control_y = st.slider("Control Y position (%)", 0, 100, 40) / 100
                    pad_positions['Control'] = (control_x, control_y)
                
                if st.button("🔍 Analyze Dipstick", type="primary"):
                    with st.spinner("Analyzing colors..."):
                        # Measure colors
                        color_results = measure_pad_colors(image, pad_positions)
                        st.session_state.color_results = color_results
                        
                        # Placeholder uACR calculation (in reality, would map colors to concentration)
                        # This is a simplified example based on color values
                        albumin_rgb = color_results['Albumin']['RGB']
                        creat_rgb = color_results['Creatinine']['RGB']
                        
                        # Simple algorithm for demonstration
                        albumin_score = np.mean(albumin_rgb) / 255
                        creat_score = np.mean(creat_rgb) / 255
                        
                        # Generate uACR value based on color (placeholder logic)
                        uacr_value = albumin_score * 300 + np.random.uniform(-20, 20)
                        uacr_value = max(5, min(1000, uacr_value))
                        
                        st.session_state.uacr_value = uacr_value
                        st.session_state.analysis_complete = True
                        
                        st.success("Analysis complete!")
            else:
                st.markdown(f'<div class="warning-box">⚠️ {quality_msg}</div>', unsafe_allow_html=True)
                
                # Tips for better image
                with st.expander("Tips for better images"):
                    st.markdown("""
                    - Ensure even lighting
                    - Avoid shadows on the dipstick
                    - Hold camera steady
                    - Include the color reference chart
                    - Keep dipstick in focus
                    - Place on a neutral background
                    """)
    
    with right_col:
        if st.session_state.analysis_complete:
            st.subheader("📊 Analysis Results")
            
            # Display color measurements
            with st.expander("Color Measurements", expanded=True):
                for pad, values in st.session_state.color_results.items():
                    rgb = values['RGB']
                    hsv = values['HSV']
                    
                    st.markdown(f"**{pad}**")
                    col1, col2, col3 = st.columns(3)
                    
                    # RGB display
                    color_hex = '#{:02x}{:02x}{:02x}'.format(
                        int(rgb[0]), int(rgb[1]), int(rgb[2])
                    )
                    
                    col1.markdown(f"🎨 RGB: {int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}")
                    col2.markdown(f"🌈 HSV: {hsv[0]:.1f}°, {hsv[1]:.1f}, {hsv[2]:.1f}")
                    col3.markdown(f"<div style='background-color:{color_hex}; width:50px; height:20px; border-radius:5px;'></div>", 
                                unsafe_allow_html=True)
            
            st.divider()
            
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
            category, color = get_uacr_category(uacr)
            
            # Metrics display
            col1, col2, col3 = st.columns(3)
            col1.metric("uACR", f"{uacr:.1f} mg/g")
            col2.metric("eGFR", f"{egfr} mL/min/1.73m²")
            col3.markdown(f"**Category:** <span style='color:{color}'>{category}</span>", 
                         unsafe_allow_html=True)
            
            # Interpretation
            st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
            st.markdown("### 📋 Interpretation")
            
            if uacr < 30:
                st.markdown("✅ **Normal to mildly increased albuminuria**")
                st.markdown("Your uACR is within normal range. Continue regular monitoring as recommended.")
            elif uacr < 300:
                st.markdown("⚠️ **Moderately increased albuminuria**")
                st.markdown("This may indicate early kidney damage. Please consult your healthcare provider for further evaluation.")
            else:
                st.markdown("🔴 **Severely increased albuminuria**")
                st.markdown("This suggests significant kidney damage. Please consult your healthcare provider immediately.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save results button
            if st.button("💾 Save Results"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Create results dictionary
                result_entry = {
                    'timestamp': timestamp,
                    'uacr': uacr,
                    'egfr': egfr,
                    'category': category,
                    'age': age,
                    'gender': gender
                }
                
                st.session_state.history.append(result_entry)
                st.success("Results saved to history!")
            
            # Download report
            if st.button("📄 Download Report"):
                # Create report content
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
                    label="Download as Text",
                    data=report,
                    file_name=f"uACR_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            # Show placeholder and instructions
            st.subheader("ℹ️ Instructions")
            st.markdown("""
            <div class="info-box">
            1. Take or upload a clear photo of your dipstick
            2. Ensure good lighting and focus
            3. Adjust the pad position sliders to match your dipstick
            4. Click 'Analyze' when ready
            5. Enter your demographic information
            </div>
            """, unsafe_allow_html=True)
            
            # Display quick facts
            st.subheader("📚 Quick Facts")
            display_kidney_facts()

def education_page():
    st.subheader("📚 Understanding Kidney Health")
    
    tabs = st.tabs(["uACR Explained", "eGFR Explained", "Risk Factors", "Prevention"])
    
    with tabs[0]:
        st.markdown("""
        ### What is uACR?
        
        **Urine Albumin-to-Creatinine Ratio (uACR)** is a test that measures the amount of albumin (a protein) in your urine compared to creatinine.
        
        #### Why is it important?
        - Detects early kidney damage
        - Monitors kidney disease progression
        - Assesses cardiovascular risk
        
        #### Understanding Your Results:
        - **<30 mg/g**: Normal to mildly increased
        - **30-300 mg/g**: Moderately increased (early kidney disease)
        - **>300 mg/g**: Severely increased (advanced kidney disease)
        """)
        
        # Add a visual representation
        fig = go.Figure(data=[
            go.Bar(name='Normal', x=['<30 mg/g'], y=[30], marker_color='green'),
            go.Bar(name='Moderate', x=['30-300 mg/g'], y=[270], marker_color='yellow'),
            go.Bar(name='Severe', x=['>300 mg/g'], y=[100], marker_color='red')
        ])
        fig.update_layout(title="uACR Categories", barmode='stack')
        st.plotly_chart(fig)
    
    with tabs[1]:
        st.markdown("""
        ### What is eGFR?
        
        **Estimated Glomerular Filtration Rate (eGFR)** estimates how well your kidneys are filtering waste from your blood.
        
        #### CKD Stages based on eGFR:
        | Stage | eGFR (mL/min) | Description |
        |-------|---------------|-------------|
        | 1 | ≥90 | Normal kidney function |
        | 2 | 60-89 | Mildly decreased |
        | 3a | 45-59 | Mild to moderate decrease |
        | 3b | 30-44 | Moderate to severe decrease |
        | 4 | 15-29 | Severe decrease |
        | 5 | <15 | Kidney failure |
        """)
    
    with tabs[2]:
        st.markdown("""
        ### Risk Factors for Kidney Disease
        
        #### Major Risk Factors:
        - 🩺 **Diabetes** - Leading cause of kidney disease
        - ❤️ **Hypertension** - High blood pressure damages kidney vessels
        - 👴 **Age >60** - Increased risk with age
        - 🧬 **Family History** - Genetic predisposition
        - 🏥 **Cardiovascular Disease** - Related to kidney health
        - ⚖️ **Obesity** - Increases diabetes and hypertension risk
        
        #### Other Risk Factors:
        - Smoking
        - Excessive alcohol use
        - Autoimmune diseases
        - Recurrent kidney infections
        - Prolonged use of certain medications (NSAIDs)
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Prevention and Lifestyle Tips
        
        #### 🥗 Diet:
        - Reduce sodium intake
        - Limit processed foods
        - Control protein consumption if advised
        - Stay hydrated
        
        #### 🏃‍♂️ Exercise:
        - 150 minutes moderate activity weekly
        - Maintain healthy weight
        - Control blood pressure
        
        #### 🩺 Regular Monitoring:
        - Annual check-ups if at risk
        - Monitor blood pressure at home
        - Regular blood sugar testing if diabetic
        - Know your family history
        """)

def guidelines_page():
    st.subheader("📋 Clinical Guidelines")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### KDIGO 2024 Guidelines
        
        #### Screening Recommendations:
        
        **For General Population:**
        - No routine screening recommended
        - Assess risk factors during periodic health exams
        
        **For High-Risk Individuals:**
        - Screen annually if:
            - Diabetes (Type 1 or 2)
            - Hypertension
            - Cardiovascular disease
            - Family history of kidney disease
            - Age >60 years
            - Previous acute kidney injury
        
        #### Testing Frequency:
        | Risk Level | uACR Testing | eGFR Testing |
        |------------|--------------|--------------|
        | Low Risk | Every 3-5 years | Every 3-5 years |
        | Moderate Risk | Annually | Annually |
        | High Risk | Every 6-12 months | Every 6-12 months |
        
        #### Treatment Targets:
        - **Blood Pressure**: <130/80 mmHg for most patients
        - **uACR Reduction**: Aim for >30% reduction in high-risk patients
        - **eGFR Decline**: Slow progression to <2-3 mL/min/year
        """)
    
    with col2:
        st.markdown("""
        ### Quick Reference
        
        #### When to Refer:
        - eGFR <30 mL/min
        - Rapid eGFR decline (>5 mL/min/year)
        - uACR >300 mg/g
        - Difficult to control BP
        - Suspected glomerulonephritis
        
        #### Emergency Signs:
        - Severe hypertension
        - Pulmonary edema
        - Pericarditis
        - Uremic symptoms
        - Severe hyperkalemia
        """)
        
        st.markdown("""
        <div class="warning-box">
        ⚠️ This is a screening tool only. Always consult healthcare providers for medical decisions.
        </div>
        """, unsafe_allow_html=True)

def history_page():
    st.subheader("📊 History & Trends")
    
    if st.session_state.history:
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Display metrics over time
        st.line_chart(df.set_index('timestamp')[['uacr', 'egfr']])
        
        # Show history table
        st.dataframe(
            df[['timestamp', 'uacr', 'egfr', 'category']].round(2),
            use_container_width=True
        )
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name=f"uACR_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No history available. Complete an analysis to see your history.")
        
        # Sample data for demonstration
        st.markdown("### Sample Trend Visualization")
        dates = pd.date_range(start='2024-01-01', periods=6, freq='W')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'uacr': np.random.uniform(15, 50, 6),
            'egfr': np.random.uniform(70, 90, 6)
        })
        st.line_chart(sample_data.set_index('timestamp'))

if __name__ == "__main__":
    main()