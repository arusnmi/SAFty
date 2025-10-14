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
# Helper function to check IoU (Intersection over Union)
# -------------------------------------------------------------------
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# -------------------------------------------------------------------
# Helper function to check if PPE is nearby person
# -------------------------------------------------------------------
def is_nearby(person_bbox, ppe_bbox, threshold=100):
    """Check if PPE bounding box is near the person."""
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_bbox
    
    # Calculate center points
    p_center_x = (p_x1 + p_x2) / 2
    p_center_y = (p_y1 + p_y2) / 2
    ppe_center_x = (ppe_x1 + ppe_x2) / 2
    ppe_center_y = (ppe_y1 + ppe_y2) / 2
    
    # Calculate distance
    distance = np.sqrt((p_center_x - ppe_center_x)**2 + (p_center_y - ppe_center_y)**2)
    
    return distance < threshold

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

    # Collect all detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

    # ----------------------------------------------------------------
    # Separate persons and PPE items
    # ----------------------------------------------------------------
    persons = [d for d in detections if d["class"].lower() == "person"]
    ppe_items = [d for d in detections if d["class"] in REQUIRED_PPE or d["class"] in VIOLATION_ITEMS]

    # ----------------------------------------------------------------
    # Compliance analysis per person
    # ----------------------------------------------------------------
    compliant = partial_compliant = non_compliant = 0
    person_compliance = []

    for person in persons:
        person_bbox = person["bbox"]
        
        # Find PPE items associated with this person (using IoU)
        associated_ppe = []
        for ppe in ppe_items:
            ppe_bbox = ppe["bbox"]
            iou = calculate_iou(person_bbox, ppe_bbox)
            
            # If PPE overlaps or is nearby the person
            if iou > 0.1 or is_nearby(person_bbox, ppe_bbox):
                associated_ppe.append(ppe["class"])
        
        # Count how many required PPE items are present
        detected_ppe_set = set(associated_ppe) & REQUIRED_PPE
        has_violations = bool(set(associated_ppe) & VIOLATION_ITEMS)
        ppe_count = len(detected_ppe_set)
        
        # Determine compliance status
        if ppe_count == 3 and not has_violations:
            status = "compliant"
            box_color = (0, 255, 0)  # Green
            compliant += 1
        elif ppe_count == 0 or has_violations:
            status = "non_compliant"
            box_color = (0, 0, 255)  # Red
            non_compliant += 1
        else:
            status = "partial"
            box_color = (0, 255, 255)  # Yellow
            partial_compliant += 1
        
        person_compliance.append({
            "bbox": person_bbox,
            "status": status,
            "color": box_color,
            "ppe_count": ppe_count
        })
        
        # Draw compliance box around person
        x1, y1, x2, y2 = person_bbox
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 4)
        
        # Add status label
        label = f"PPE: {ppe_count}/3"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(output_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw PPE item detections (smaller boxes)
    for ppe in ppe_items:
        x1, y1, x2, y2 = ppe["bbox"]
        cls_name = ppe["class"]
        conf = ppe["confidence"]
        
        # Determine color based on item type
        if "no" in cls_name.lower() or "without" in cls_name.lower():
            color = (0, 0, 255)  # Red for violations
        elif "mask" in cls_name.lower():
            color = (255, 255, 0)  # Cyan for mask
        elif "vest" in cls_name.lower():
            color = (0, 165, 255)  # Orange for vest
        elif "helmet" in cls_name.lower() or "hardhat" in cls_name.lower():
            color = (0, 255, 0)  # Green for helmet
        else:
            color = (255, 255, 255)  # White for others
        
        label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, label, (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    with col2:
        st.image(output_rgb, caption="Detection Results", use_column_width=True)

    # ----------------------------------------------------------------
    # Update session state
    # ----------------------------------------------------------------
    person_count = len(persons)
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
    m2.metric("Fully Compliant", compliant, delta="🟢" if compliant > 0 else None)
    m3.metric("Partially Compliant", partial_compliant, delta="🟡" if partial_compliant > 0 else None)
    m4.metric("Non-Compliant", non_compliant, delta="🔴" if non_compliant > 0 else None)

    compliance_data = pd.DataFrame({
        "Status": ["Compliant", "Partially Compliant", "Non-Compliant"],
        "Count": [compliant, partial_compliant, non_compliant]
    })
    
    # --- FIX 1: Explicitly map colors for pie chart ---
    color_map = {
        "Compliant": "green",
        "Partially Compliant": "yellow",
        "Non-Compliant": "red"
    }
    
    fig = px.pie(compliance_data, values="Count", names="Status",
                 title="PPE Compliance Distribution",
                 color="Status", 
                 color_discrete_map=color_map)
    st.plotly_chart(fig)

    # Compliance legend
    st.markdown("""
    **Compliance Legend:**
    - 🟢 **Green Box**: All 3 PPE items detected (Hardhat, Mask, Safety Vest)
    - 🟡 **Yellow Box**: 1 or 2 PPE items detected
    - 🔴 **Red Box**: No PPE detected or violations present
    """)

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
---
### Compliance Color Coding
- 🟢 **Green**: All 3 PPE items present
- 🟡 **Yellow**: 1 or 2 PPE items present
- 🔴 **Red**: No PPE or violations detected
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This AI system monitors workplace safety compliance and PPE usage using YOLOv11 object detection.")