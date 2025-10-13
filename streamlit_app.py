import os
from pathlib import Path

# CRITICAL FIX: Set environment variables BEFORE any OpenCV imports
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import cv2

# CRITICAL FIX: Patch cv2.imshow for headless environment
cv2.imshow = lambda *args: None
cv2.waitKey = lambda *args: None
cv2.destroyAllWindows = lambda *args: None

import numpy as np
from ultralytics import YOLO
import plotly.express as px
import pandas as pd
from PIL import Image

# Get the absolute path to the root of the deployed app
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Safty.pt"

st.set_page_config(layout="wide", page_title="PPE Compliance Detection", page_icon="⚠️")

# Load the model using proven loading pattern
@st.cache_resource
def load_yolo_model(path):
    """Load the YOLO model using the corrected absolute path."""
    try:
        import torch
        
        # Add required safe globals before model loading
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.models.yolo.detect.DetectionModel',
            'ultralytics.engine.model',
            'ultralytics.nn.tasks',
            'ultralytics.models.yolo.detect'
        ])
        
        # Disable CUDNN and set device
        torch.backends.cudnn.enabled = False
        device = torch.device('cpu')
        
        # Configure torch loading settings
        torch.set_default_dtype(torch.float32)
        
        # Load model with explicit settings
        model = YOLO(str(path))
        model.to(device)
        
        st.success(f"Model loaded successfully from: {path}")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Python working directory:", os.getcwd())
        st.write("Model path tried:", path)
        st.write("Available files:", os.listdir(BASE_DIR))
        return None

# Load the model using the fixed path
with st.spinner("Loading AI model..."):
    model = load_yolo_model(MODEL_PATH)

# Check if model loaded successfully
if model is None:
    st.error("Application setup failed. Check logs for model loading errors.")
    st.stop()

# Initialize session state for statistics
if 'detection_count' not in st.session_state:
    st.session_state['detection_count'] = 0

# Sidebar
st.sidebar.title("PPE Compliance Detection")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Main content
st.title("Worker Safety Compliance Detection")

if uploaded_file is not None:
    # Read and process image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Run inference with spinner
    with st.spinner("🔍 Analyzing image..."):
        results = model(img_bgr, verbose=False, conf=0.75, device='cpu')
    
    # Process results
    compliant = 0
    partial_compliant = 0
    non_compliant = 0
    total_workers = len(results[0].boxes)
    
    # Count compliance categories
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        if class_id == 0:  # Compliant
            compliant += 1
        elif class_id == 1:  # Partial
            partial_compliant += 1
        else:  # Non-compliant
            non_compliant += 1
    
    # Increment detection count
    st.session_state['detection_count'] += 1
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(results[0].plot(), caption="Detection Results", use_column_width=True)
    
    with col2:
        # Metrics
        st.subheader("Detection Summary")
        st.metric("Total Workers Detected", total_workers)
        st.metric("Compliant Workers", compliant)
        st.metric("Partially Compliant", partial_compliant)
        st.metric("Non-Compliant", non_compliant)
        
        # Compliance Chart
        compliance_data = pd.DataFrame({
            'Status': ['Compliant', 'Partially Compliant', 'Non-Compliant'],
            'Count': [compliant, partial_compliant, non_compliant]
        })
        
        fig = px.pie(compliance_data, values='Count', names='Status',
                     title='Compliance Distribution')
        st.plotly_chart(fig)
        
        # Alert System with improved visibility
        if non_compliant > 0:
            st.error(f"⚠️ Alert: {non_compliant} workers detected without proper PPE!")
        
        if partial_compliant > 0:
            st.warning(f"⚠️ Note: {partial_compliant} workers with partial PPE compliance")
        
        if compliant == total_workers:
            st.success("✅ All workers are compliant with PPE regulations")

else:
    st.info("👆 Please upload an image to begin detection")

# Show statistics in sidebar
if st.session_state['detection_count'] > 0:
    st.sidebar.metric("Total Detections", st.session_state['detection_count'])

# Footer
st.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This app uses YOLO AI to detect and monitor PPE compliance in workplace safety scenarios.")