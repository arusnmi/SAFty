import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import plotly.express as px
import pandas as pd
import os

# Page config
st.set_page_config(page_title="PPE Compliance Detection", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    try:
        import torch
        torch.backends.cudnn.enabled = False  # Disable CUDNN
        
        # Add all potentially needed safe globals
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.models.yolo.detect.DetectionModel',
            'ultralytics.engine.model'
        ])
        
        model_path = os.path.abspath('best.pt')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        # Load model with explicit settings
        model = YOLO(model_path, task='detect')
        st.success(f"Model loaded successfully from: {model_path}")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Python working directory:", os.getcwd())
        st.write("Model path tried:", os.path.abspath('best.pt'))
        return None

# Clear any existing cache
try:
    load_model.clear()
except:
    pass

model = load_model()

# Modify the main content section to check if model loaded successfully
if model is None:
    st.error("Failed to load the model. Please ensure the model file exists and is valid.")
    st.stop()

# Sidebar
st.sidebar.title("PPE Compliance Detection")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Main content
st.title("Worker Safety Compliance Detection")

if uploaded_file is not None:
    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Run inference
    results = model(image)
    
    # Process results
    compliant = 0
    partial_compliant = 0
    non_compliant = 0
    total_workers = len(results[0].boxes)
    
    # Count compliance categories (adjust based on your model's classes)
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        if class_id == 0:  # Assuming 0 is compliant
            compliant += 1
        elif class_id == 1:  # Assuming 1 is partial
            partial_compliant += 1
        else:  # Assuming 2 is non-compliant
            non_compliant += 1
    
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
        
        # Alert System
        if non_compliant > 0:
            st.error(f"⚠️ Alert: {non_compliant} workers detected without proper PPE!")
        
        if partial_compliant > 0:
            st.warning(f"⚠️ Note: {partial_compliant} workers with partial PPE compliance")
        
        if compliant == total_workers:
            st.success("✅ All workers are compliant with PPE regulations")

else:
    st.info("Please upload an image to begin detection")

# Footer
st.markdown("---")
st.caption("PPE Compliance Detection System - Powered by YOLO")