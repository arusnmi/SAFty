import os
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# -------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Safty.pt"  

# -------------------------------------------------------------------
# Streamlit setup
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="PPE Compliance Detection", page_icon="⚠️")
st.title("👷 PPE Compliance Detection System")

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(path: Path):
    """Load YOLOv11 model safely for CPU inference."""
    try:
        model = YOLO(str(path))
        model.to("cpu")
        st.success(f"✅ YOLOv11 model loaded: {path.name}")
        return model
    except Exception as e:
        import sys
        from ultralytics import __version__ as ultralytics_version
        st.error(f"❌ Error loading model: {type(e).__name__}: {e}")
        st.write(f"- Python: {sys.version}")
        st.write(f"- Ultralytics: {ultralytics_version}")
        st.write(f"- Model path: {path}")
        return None

with st.spinner("Loading AI model..."):
    model = load_yolo_model(MODEL_PATH)

if model is None:
    st.stop()

# -------------------------------------------------------------------
# Automatically detect model labels
# -------------------------------------------------------------------
all_labels = set(model.names.values())

REQUIRED_PPE = {lbl for lbl in all_labels if any(k in lbl.lower() for k in ["helmet", "hardhat", "mask", "vest"])}
VIOLATION_ITEMS = {lbl for lbl in all_labels if "no" in lbl.lower() or "without" in lbl.lower()}

if not REQUIRED_PPE:
    REQUIRED_PPE = {"Hardhat", "Mask", "Safety Vest"}
if not VIOLATION_ITEMS:
    VIOLATION_ITEMS = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# -------------------------------------------------------------------
# Color mapping
# -------------------------------------------------------------------
def get_color(label):
    label_l = label.lower()
    if "no" in label_l or "without" in label_l:
        return (0, 0, 255)  # Red
    if "mask" in label_l:
        return (255, 255, 0)
    if "vest" in label_l:
        return (0, 165, 255)
    if "helmet" in label_l or "hardhat" in label_l:
        return (0, 255, 0)
    return (255, 255, 255)

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
if "detection_history" not in st.session_state:
    st.session_state["detection_history"] = []

# -------------------------------------------------------------------
# File uploader
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # ----------------------------------------------------------------
    # YOLOv11 Inference
    # ----------------------------------------------------------------
    with st.spinner("🔍 Analyzing PPE compliance..."):
        results = model(img_bgr, conf=0.25)

    detections = []
    output_image = img_bgr.copy()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names[cls]
            detections.append({"class": cls_name, "confidence": conf})

            # Draw bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = get_color(cls_name)
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_image, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    with col2:
        st.image(output_rgb, caption="Detection Results", use_column_width=True)

    # ----------------------------------------------------------------
    # Compliance analysis
    # ----------------------------------------------------------------
    person_count = len([d for d in detections if d["class"].lower() == "person"])
    compliant = partial_compliant = non_compliant = 0

    for i in range(person_count if person_count > 0 else 1):
        detected_items = set(d["class"] for d in detections)
        missing_ppe = REQUIRED_PPE - detected_items
        has_violations = bool(VIOLATION_ITEMS & detected_items)

        if not missing_ppe and not has_violations:
            compliant += 1
        elif len(missing_ppe) == len(REQUIRED_PPE) or has_violations:
            non_compliant += 1
        else:
            partial_compliant += 1

    st.session_state["detection_history"].append({
        "total": person_count,
        "compliant": compliant,
        "partial": partial_compliant,
        "non_compliant": non_compliant,
        "timestamp": pd.Timestamp.now()
    })

    # ----------------------------------------------------------------
    # Metrics & Charts
    # ----------------------------------------------------------------
    st.subheader("📊 Compliance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Workers", person_count)
    m2.metric("Fully Compliant", compliant)
    m3.metric("Partially Compliant", partial_compliant)
    m4.metric("Non-Compliant", non_compliant)

    compliance_data = pd.DataFrame({
        "Status": ["Compliant", "Partially Compliant", "Non-Compliant"],
        "Count": [compliant, partial_compliant, non_compliant]
    })
    fig = px.pie(compliance_data, values="Count", names="Status",
                 title="PPE Compliance Distribution",
                 color_discrete_sequence=["green", "yellow", "red"])
    st.plotly_chart(fig)

    if non_compliant > 0:
        st.error(f"⚠️ ALERT: {non_compliant} workers detected without proper PPE!")
    if partial_compliant > 0:
        st.warning(f"⚠️ WARNING: {partial_compliant} workers with partial PPE compliance")

    # ----------------------------------------------------------------
    # Historical Trends
    # ----------------------------------------------------------------
    if len(st.session_state["detection_history"]) > 1:
        st.subheader("📈 Compliance Trends")
        hist_df = pd.DataFrame(st.session_state["detection_history"])
        fig_trend = px.line(hist_df, x="timestamp",
                            y=["compliant", "partial", "non_compliant"],
                            title="Compliance Trends Over Time")
        st.plotly_chart(fig_trend)

else:
    st.info("👆 Please upload an image to begin PPE compliance detection")

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("ℹ️ System Information")
st.sidebar.markdown(f"""
### Detected Model Classes
{', '.join(sorted(all_labels))}
---
### Required PPE (auto-detected)
{', '.join(sorted(REQUIRED_PPE))}
---
### Violation Labels (auto-detected)
{', '.join(sorted(VIOLATION_ITEMS))}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This AI system monitors workplace safety compliance and PPE usage using YOLOv11 object detection.")
