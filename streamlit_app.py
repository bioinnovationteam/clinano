import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import io
import base64
from datetime import datetime
import json

st.set_page_config(page_title="Dipstick uACR Analyzer", layout="wide")

# Custom CSS for better camera UI
st.markdown("""
<style>
    .camera-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .stButton button {
        width: 100%;
    }
    .capture-btn {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time camera feed"""
    def __init__(self):
        self.frame = None
        self.captured_frame = None
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img
    
    def capture(self):
        if self.frame is not None:
            self.captured_frame = self.frame.copy()
            return True
        return False

class DipstickAnalyzer:
    def __init__(self):
        # Reference strips expected RGB values (printed on dipstick)
        self.REF_STRIPS = {
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'black': (32, 32, 32)
        }
        
        # Pad locations relative to white reference strip (in mm)
        self.PAD_OFFSETS = {
            'albumin': (10, 0),    # 10mm right, 0mm down
            'creatinine': (10, 12), # 10mm right, 12mm down
            'ph': (10, 24)         # 10mm right, 24mm down
        }
        
        # Pad sizes in mm
        self.PAD_SIZE_MM = 5
        
        # RGB -> Concentration models (example - replace with your actual data)
        self.load_models()
    
    def load_models(self):
        """Load your pre-trained models or lookup tables"""
        # Example: Albumin model (RGB -> mg/dL)
        self.alb_model = {
            'rgb_values': [
                (255, 255, 255),  # 0 mg/dL
                (240, 230, 220),  # 5 mg/dL
                (200, 180, 150),  # 30 mg/dL
                (150, 120, 100),  # 100 mg/dL
                (100, 70, 50)     # 300 mg/dL
            ],
            'conc': [0, 5, 30, 100, 300]
        }
        
        # Creatinine model (RGB -> mg/dL)
        self.creat_model = {
            'rgb_values': [
                (255, 255, 255),  # 0 mg/dL
                (230, 235, 240),  # 50 mg/dL
                (200, 210, 220),  # 100 mg/dL
                (150, 170, 190),  # 200 mg/dL
            ],
            'conc': [0, 50, 100, 200]
        }
        
        # Build interpolators
        self.alb_interp = self.build_interpolator(self.alb_model)
        self.creat_interp = self.build_interpolator(self.creat_model)
    
    def build_interpolator(self, model):
        """Create interpolation function from RGB to concentration"""
        # For better accuracy, use color distance or 3D interpolation
        # Here using grayscale intensity for simplicity
        gray_values = [np.mean(rgb) for rgb in model['rgb_values']]
        return interp1d(gray_values, model['conc'], 
                       kind='linear', fill_value='extrapolate')
    
    def find_reference_strips(self, image):
        """Locate the 3 reference strips on the dipstick"""
        detected = {}
        
        for name, target_rgb in self.REF_STRIPS.items():
            # Color thresholding with dynamic tolerance
            tolerance = 40
            lower = np.array([max(0, c-tolerance) for c in target_rgb])
            upper = np.array([min(255, c+tolerance) for c in target_rgb])
            mask = cv2.inRange(image, lower, upper)
            
            # Morphological operations to clean up mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Filter by area (remove noise)
                min_area = (image.shape[0] * image.shape[1]) / 5000
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                
                if valid_contours:
                    largest = max(valid_contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detected[name] = (cx, cy)
        
        # Validate we found all 3 and they're reasonably spaced
        if len(detected) == 3:
            # Check distances between reference strips
            white_gray_dist = np.sqrt((detected['gray'][0] - detected['white'][0])**2 +
                                     (detected['gray'][1] - detected['white'][1])**2)
            gray_black_dist = np.sqrt((detected['black'][0] - detected['gray'][0])**2 +
                                     (detected['black'][1] - detected['gray'][1])**2)
            
            # Distances should be similar within 30%
            if abs(white_gray_dist - gray_black_dist) / max(white_gray_dist, gray_black_dist) < 0.3:
                return detected
        
        return None
    
    def estimate_mm_per_pixel(self, ref_centers):
        """Estimate pixels per mm based on reference strip spacing"""
        if 'gray' in ref_centers and 'white' in ref_centers:
            dist_px = np.sqrt((ref_centers['gray'][0] - ref_centers['white'][0])**2 +
                            (ref_centers['gray'][1] - ref_centers['white'][1])**2)
            # Assuming gray is 12mm from white (adjust to your design)
            mm_per_px = 12.0 / dist_px if dist_px > 0 else 1
            return mm_per_px
        return 1
    
    def extract_pad_rgb(self, image, ref_centers, pad_name):
        """Extract RGB from a specific detection pad"""
        white_center = ref_centers['white']
        offset = self.PAD_OFFSETS[pad_name]
        
        # Estimate pixels per mm
        mm_per_px = self.estimate_mm_per_pixel(ref_centers)
        
        # Calculate pad center in pixels
        pad_x = white_center[0] + int(offset[0] * mm_per_px)
        pad_y = white_center[1] + int(offset[1] * mm_per_px)
        
        # Calculate ROI size in pixels
        roi_size_px = int(self.PAD_SIZE_MM * mm_per_px)
        roi_size_px = max(10, min(roi_size_px, 50))  # Clamp between 10-50 pixels
        
        # Extract ROI
        x1 = max(0, pad_x - roi_size_px//2)
        y1 = max(0, pad_y - roi_size_px//2)
        x2 = min(image.shape[1], pad_x + roi_size_px//2)
        y2 = min(image.shape[0], pad_y + roi_size_px//2)
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Return median RGB (more robust to outliers)
        median_rgb = np.median(roi, axis=(0,1))
        return tuple(median_rgb.astype(int))
    
    def color_correct(self, pad_rgb, measured_refs):
        """Apply color correction using reference strips"""
        white_rgb = measured_refs.get('white')
        black_rgb = measured_refs.get('black')
        
        if white_rgb and black_rgb:
            corrected = []
            for c in range(3):
                # Linear mapping: target = a * measured + b
                target_white = self.REF_STRIPS['white'][c]
                target_black = self.REF_STRIPS['black'][c]
                
                # Avoid division by zero
                if white_rgb[c] - black_rgb[c] == 0:
                    a = 1
                else:
                    a = (target_white - target_black) / (white_rgb[c] - black_rgb[c])
                
                b = target_white - a * white_rgb[c]
                
                corrected_val = a * pad_rgb[c] + b
                corrected.append(int(np.clip(corrected_val, 0, 255)))
            return tuple(corrected)
        return pad_rgb
    
    def rgb_to_concentration(self, rgb, model_type='albumin'):
        """Convert RGB to biomarker concentration"""
        # Use Euclidean distance to reference colors for better accuracy
        if model_type == 'albumin':
            model = self.alb_model
            interp = self.alb_interp
        else:
            model = self.creat_model
            interp = self.creat_interp
        
        # Find closest reference RGB and interpolate
        rgb_array = np.array(rgb)
        distances = []
        for ref_rgb in model['rgb_values']:
            dist = np.linalg.norm(rgb_array - np.array(ref_rgb))
            distances.append(dist)
        
        # Use intensity for interpolation (simplified)
        intensity = np.mean(rgb)
        conc = float(interp(intensity))
        return max(0, conc)
    
    def calculate_uacr(self, albumin_mgdl, creatinine_mgdl):
        """Calculate uACR in mg/g"""
        # Convert creatinine from mg/dL to g/L
        creatinine_gl = creatinine_mgdl / 100.0
        if creatinine_gl == 0:
            return None
        uacr = albumin_mgdl / creatinine_gl
        return uacr
    
    def draw_annotations(self, image, ref_centers, results):
        """Draw annotations on the image for visualization"""
        img_copy = image.copy()
        
        # Draw reference strip markers
        for name, center in ref_centers.items():
            cv2.circle(img_copy, center, 10, (0, 255, 0), 2)
            cv2.putText(img_copy, name, (center[0]+10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw pad regions
        mm_per_px = self.estimate_mm_per_pixel(ref_centers)
        for pad_name, offset in self.PAD_OFFSETS.items():
            white_center = ref_centers['white']
            pad_x = white_center[0] + int(offset[0] * mm_per_px)
            pad_y = white_center[1] + int(offset[1] * mm_per_px)
            roi_size = int(self.PAD_SIZE_MM * mm_per_px)
            
            cv2.rectangle(img_copy, 
                         (pad_x - roi_size//2, pad_y - roi_size//2),
                         (pad_x + roi_size//2, pad_y + roi_size//2),
                         (255, 0, 0), 2)
            cv2.putText(img_copy, pad_name, (pad_x - 20, pad_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add concentration text if available
        if 'albumin' in results:
            y_offset = 30
            for key, value in results.items():
                if 'conc' in str(key):
                    text = f"{key}: {value:.1f}"
                    cv2.putText(img_copy, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
        
        return img_copy
    
    def analyze(self, image):
        """Main analysis pipeline"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image[0,0,0] > 1.0:  # Assume RGB format (0-255)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
        else:
            image_bgr = image
        
        # Find reference strips
        ref_centers = self.find_reference_strips(image_bgr)
        if ref_centers is None:
            return {'error': 'Could not detect reference strips. Please ensure good lighting and all 3 reference strips are visible.'}
        
        # Extract reference strip RGB values for color correction
        ref_rgbs = {}
        for name, center in ref_centers.items():
            x, y = center
            roi_size = 10
            x1 = max(0, x - roi_size//2)
            y1 = max(0, y - roi_size//2)
            x2 = min(image_bgr.shape[1], x + roi_size//2)
            y2 = min(image_bgr.shape[0], y + roi_size//2)
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                ref_rgbs[name] = tuple(np.median(roi, axis=(0,1)).astype(int))
        
        # Extract and analyze each pad
        results = {}
        for pad in ['albumin', 'creatinine', 'ph']:
            pad_rgb = self.extract_pad_rgb(image_bgr, ref_centers, pad)
            if pad_rgb:
                corrected_rgb = self.color_correct(pad_rgb, ref_rgbs)
                results[pad] = {
                    'raw_rgb': pad_rgb,
                    'corrected_rgb': corrected_rgb
                }
        
        # Calculate concentrations
        if 'albumin' in results and 'creatinine' in results:
            alb_conc = self.rgb_to_concentration(results['albumin']['corrected_rgb'], 'albumin')
            creat_conc = self.rgb_to_concentration(results['creatinine']['corrected_rgb'], 'creatinine')
            uacr = self.calculate_uacr(alb_conc, creat_conc)
            
            results['albumin']['conc_mgdl'] = alb_conc
            results['creatinine']['conc_mgdl'] = creat_conc
            results['uacr_mg_per_g'] = uacr
            
            # Clinical interpretation
            if uacr < 30:
                results['interpretation'] = "Normal (<30 mg/g)"
                results['risk_level'] = "Low"
            elif uacr < 300:
                results['interpretation'] = "Moderately increased (30-300 mg/g) - Microalbuminuria"
                results['risk_level'] = "Moderate"
            else:
                results['interpretation'] = "Severely increased (>300 mg/g) - Macroalbuminuria"
                results['risk_level'] = "High"
            
            # Quality assessment
            results['quality_score'] = self.assess_quality(ref_rgbs)
        
        # Draw annotations
        annotated_image = self.draw_annotations(image_bgr, ref_centers, results)
        
        return results, annotated_image
    
    def assess_quality(self, ref_rgbs):
        """Assess image quality based on reference strips"""
        score = 100
        
        # Check white reference
        if 'white' in ref_rgbs:
            white_brightness = np.mean(ref_rgbs['white'])
            if white_brightness < 200:
                score -= 30
                st.warning("White reference too dark - improve lighting")
        
        # Check black reference
        if 'black' in ref_rgbs:
            black_brightness = np.mean(ref_rgbs['black'])
            if black_brightness > 80:
                score -= 20
                st.warning("Black reference too light - check for glare")
        
        return max(0, min(100, score))

# Camera capture component
def camera_capture_section():
    """Handle camera capture functionality"""
    st.subheader("📸 Capture with Camera")
    
    # Choose camera source
    camera_source = st.radio("Select camera:", ["Front", "Rear"], horizontal=True)
    
    # RTC configuration for better mobile support
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Webrtc streamer
    ctx = webrtc_streamer(
        key="dipstick-camera",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_transformer:
        # Capture button
        if st.button("📷 Capture Image", key="capture_btn", use_container_width=True):
            if ctx.video_transformer.capture():
                st.success("Image captured successfully!")
                return ctx.video_transformer.captured_frame
            else:
                st.error("Failed to capture image. Please ensure camera is working.")
    
    return None

# File upload section
def file_upload_section():
    """Handle file upload"""
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", 
                                     type=['jpg', 'jpeg', 'png', 'heic'])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    return None

# Main app
def main():
    st.title("📱 Smartphone Dipstick uACR Analyzer")
    st.markdown("*Instant analysis of albumin-to-creatinine ratio from dipstick photos*")
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        ### For best results:
        1. **Good lighting** - Avoid shadows and glare
        2. **Flat surface** - Place dipstick on a flat surface
        3. **Steady hand** - Hold camera steady when capturing
        4. **Fill frame** - Dipstick should fill most of the frame
        5. **Reference strips** - Ensure all 3 color references are visible
        
        ### Reference Strip Colors:
        - ⬜ **White** (255,255,255)
        - ⬛ **Gray** (128,128,128)  
        - ◼️ **Black** (32,32,32)
        """)
        
        st.divider()
        
        # Analysis history
        if st.session_state.analysis_history:
            st.header("📊 Recent Analyses")
            for i, result in enumerate(reversed(st.session_state.analysis_history[-5:])):
                st.metric(
                    result['timestamp'],
                    f"{result['uacr']:.1f} mg/g",
                    result['interpretation']
                )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Camera", "📁 Upload", "📖 About"])
    
    with tab1:
        captured_image = camera_capture_section()
        if captured_image is not None:
            # Convert BGR to RGB for display
            captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            st.image(captured_image_rgb, caption="Captured Image", use_container_width=True)
            
            if st.button("🔬 Analyze Captured Image", type="primary", use_container_width=True):
                process_image(Image.fromarray(captured_image_rgb))
    
    with tab2:
        uploaded_image = file_upload_section()
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔬 Analyze Uploaded Image", type="primary", use_container_width=True):
                process_image(uploaded_image)
    
    with tab3:
        st.header("About This Test")
        st.markdown("""
        ### uACR (Urine Albumin-to-Creatinine Ratio)
        
        The uACR test measures kidney function by comparing the amount of albumin (a protein) 
        to creatinine (a waste product) in urine.
        
        **Clinical Significance:**
        - **<30 mg/g**: Normal
        - **30-300 mg/g**: Moderately increased (Microalbuminuria)
        - **>300 mg/g**: Severely increased (Macroalbuminuria)
        
        **Important Notes:**
        - This is a demo application for educational purposes
        - Not for clinical diagnosis without validation
        - Results should be confirmed by laboratory testing
        - Consult healthcare provider for medical decisions
        
        **Technical Details:**
        - Uses computer vision for dipstick detection
        - Color correction using reference strips
        - RGB to concentration mapping via calibration models
        """)

def process_image(image):
    """Process the image and display results"""
    analyzer = DipstickAnalyzer()
    
    with st.spinner("Analyzing dipstick..."):
        results, annotated_image = analyzer.analyze(image)
        
        if 'error' in results:
            st.error(results['error'])
            return
        
        # Display results in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔍 Detection Results")
            # Convert annotated image back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Regions", use_container_width=True)
            
            # Quality score
            if 'quality_score' in results:
                quality_color = "🟢" if results['quality_score'] > 80 else "🟡" if results['quality_score'] > 60 else "🔴"
                st.metric("Image Quality", f"{quality_color} {results['quality_score']}%")
        
        with col2:
            st.subheader("📊 Analysis Results")
            
            # Metrics grid
            col_alb, col_creat = st.columns(2)
            with col_alb:
                st.metric(
                    "Albumin", 
                    f"{results['albumin']['conc_mgdl']:.1f} mg/dL",
                    delta=None
                )
                with st.expander("RGB Values"):
                    st.write(f"Raw: {results['albumin']['raw_rgb']}")
                    st.write(f"Corrected: {results['albumin']['corrected_rgb']}")
            
            with col_creat:
                st.metric(
                    "Creatinine", 
                    f"{results['creatinine']['conc_mgdl']:.1f} mg/dL"
                )
                with st.expander("RGB Values"):
                    st.write(f"Raw: {results['creatinine']['raw_rgb']}")
                    st.write(f"Corrected: {results['creatinine']['corrected_rgb']}")
            
            st.divider()
            
            # uACR result with styling
            uacr = results['uacr_mg_per_g']
            if uacr < 30:
                st.success(f"### ✅ uACR: {uacr:.1f} mg/g")
                st.info(f"**{results['interpretation']}**")
            elif uacr < 300:
                st.warning(f"### ⚠️ uACR: {uacr:.1f} mg/g")
                st.warning(f"**{results['interpretation']}**")
            else:
                st.error(f"### 🔴 uACR: {uacr:.1f} mg/g")
                st.error(f"**{results['interpretation']}**")
            
            # Additional info
            if 'ph' in results:
                st.caption(f"pH Pad: {results['ph']['corrected_rgb']}")
        
        # Save to history
        st.session_state.analysis_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'uacr': uacr,
            'interpretation': results['interpretation'],
            'albumin': results['albumin']['conc_mgdl'],
            'creatinine': results['creatinine']['conc_mgdl']
        })
        
        # Export options
        st.divider()
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📄 Export as PDF", use_container_width=True):
                # Simplified PDF export (would need reportlab or similar)
                st.info("PDF export would be implemented here")
        
        with col_export2:
            if st.button("💾 Save Results", use_container_width=True):
                # Save results to JSON
                result_json = json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'uacr': uacr,
                    'albumin_mgdl': results['albumin']['conc_mgdl'],
                    'creatinine_mgdl': results['creatinine']['conc_mgdl'],
                    'interpretation': results['interpretation'],
                    'quality_score': results.get('quality_score', 0)
                }, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=result_json,
                    file_name=f"uacr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col_export3:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.rerun()

if __name__ == "__main__":
    # Check for required packages
    try:
        import streamlit_webrtc
    except ImportError:
        st.error("Please install streamlit-webrtc: pip install streamlit-webrtc")
        st.stop()
    
    main()
