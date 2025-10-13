import os
from pathlib import Path
import base64

# CRITICAL FIX: Set environment variables BEFORE any OpenCV imports
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# Get the absolute path to the root of the deployed app
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Safty.pt"

st.set_page_config(layout="wide", page_title="PPE Compliance Detection", page_icon="⚠️")

# Define PPE compliance rules
REQUIRED_PPE = {'Hardhat', 'Mask', 'Safety Vest'}
VIOLATION_ITEMS = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}

@st.cache_resource
def load_yolo_model(path):
    """Simplified YOLO model loader compatible with latest Ultralytics."""
    try:
        import torch
        from ultralytics import YOLO

        # Always use CPU for Streamlit
        device = torch.device("cpu")

        # Directly load YOLO model
        model = YOLO(str(path))
        model.to(device)

        st.success(f"✅ Model loaded successfully from: {path}")
        return model

    except Exception as e:
        import sys
        from ultralytics import __version__ as ultralytics_version
        st.error(f"❌ Error loading model: {type(e).__name__}: {str(e)}")
        st.write("System Information:")
        st.write(f"- Python version: {sys.version}")
        st.write(f"- PyTorch version: {torch.__version__}")
        st.write(f"- Ultralytics version: {ultralytics_version}")
        st.write(f"- Model path: {path}")
        st.write(f"- Working directory: {os.getcwd()}")
        return None


# Clear any existing cache
if hasattr(load_yolo_model, 'clear'):
    load_yolo_model.clear()

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = []

# Load the model
with st.spinner("Loading AI model..."):
    model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error("Failed to load model. Please check the model file.")
    st.stop()

st.title("👷 PPE Compliance Detection System")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    # Run detection
    with st.spinner("🔍 Analyzing PPE compliance..."):
        results = model(img_bgr, conf=0.25)
    
    # Process results
    detections = []
    compliant = 0
    partial_compliant = 0
    non_compliant = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            cls_name = model.names[cls]
            detections.append({
                'class': cls_name,
                'confidence': conf
            })
    
    # Count people and their compliance
    person_count = len([d for d in detections if d['class'] == 'Person'])
    
    # Analyze compliance for each person
    for i in range(person_count):
        person_items = set(d['class'] for d in detections)
        missing_ppe = REQUIRED_PPE - person_items
        has_violations = bool(VIOLATION_ITEMS & person_items)
        
        if not missing_ppe and not has_violations:
            compliant += 1
        elif len(missing_ppe) == len(REQUIRED_PPE) or has_violations:
            non_compliant += 1
        else:
            partial_compliant += 1
    
    # Display detection results
    with col2:
        st.image(results[0].plot(), caption="Detection Results", use_column_width=True)
    
    # Add to history
    st.session_state['detection_history'].append({
        'total': person_count,
        'compliant': compliant,
        'partial': partial_compliant,
        'non_compliant': non_compliant,
        'timestamp': pd.Timestamp.now()
    })
    
    # Display metrics
    st.subheader("📊 Compliance Summary")
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Workers", person_count)
    with m2:
        st.metric("Fully Compliant", compliant)
    with m3:
        st.metric("Partially Compliant", partial_compliant)
    with m4:
        st.metric("Non-Compliant", non_compliant)
    
    # Compliance Distribution Chart
    compliance_data = pd.DataFrame({
        'Status': ['Compliant', 'Partially Compliant', 'Non-Compliant'],
        'Count': [compliant, partial_compliant, non_compliant]
    })
    
    fig = px.pie(compliance_data, values='Count', names='Status',
                 title='PPE Compliance Distribution',
                 color_discrete_sequence=['green', 'yellow', 'red'])
    st.plotly_chart(fig)
    
    # Violation Alerts
    if non_compliant > 0:
        st.error(f"⚠️ ALERT: {non_compliant} workers detected without proper PPE!")
        
    if partial_compliant > 0:
        st.warning(f"⚠️ WARNING: {partial_compliant} workers with partial PPE compliance")
    
    # Historical Trends
    if len(st.session_state['detection_history']) > 1:
        st.subheader("📈 Compliance Trends")
        hist_df = pd.DataFrame(st.session_state['detection_history'])
        
        fig_trend = px.line(hist_df, x='timestamp', 
                           y=['compliant', 'partial', 'non_compliant'],
                           title='Compliance Trends Over Time')
        st.plotly_chart(fig_trend)

else:
    st.info("👆 Please upload an image to begin PPE compliance detection")

# Sidebar information
st.sidebar.title("ℹ️ System Information")
st.sidebar.markdown("""
### Required PPE
- Hard Hat
- Safety Vest
- Face Mask

### Detection Categories
- ✅ Compliant: All PPE present
- ⚠️ Partial: Some PPE missing
- ❌ Non-compliant: No PPE or unsafe conditions
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This system uses AI to monitor workplace safety compliance and PPE usage.")