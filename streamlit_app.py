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
# Automatically detect model labels and define types
# -------------------------------------------------------------------
all_labels = set(model.names.values())

# Define PPE and Violation labels based on model output
REQUIRED_PPE = {lbl for lbl in all_labels if any(k in lbl.lower() for k in ["helmet", "hardhat", "mask", "vest"])}
VIOLATION_ITEMS = {lbl for lbl in all_labels if "no" in lbl.lower() or "without" in lbl.lower()}

if not REQUIRED_PPE:
    REQUIRED_PPE = {"Hardhat", "Mask", "Safety Vest"}
if not VIOLATION_ITEMS:
    VIOLATION_ITEMS = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# Define the set of required PPE types for accurate counting (3 total types)
PPE_TYPE_MAP = {
    "Hardhat/Helmet": ["hardhat", "helmet"],
    "Mask": ["mask", "face mask"],
    "Safety Vest": ["vest", "safety vest"]
}
TOTAL_REQUIRED_PPE = len(PPE_TYPE_MAP)

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
def is_nearby(person_bbox, ppe_bbox):
    """Check if PPE bounding box is near the person using normalized threshold."""
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_bbox
    
    # Calculate center points
    p_center_x = (p_x1 + p_x2) / 2
    p_center_y = (p_y1 + p_y2) / 2
    ppe_center_x = (ppe_x1 + ppe_x2) / 2
    ppe_center_y = (ppe_y1 + ppe_y2) / 2
    
    # Calculate distance
    distance = np.sqrt((p_center_x - ppe_center_x)**2 + (p_center_y - ppe_center_y)**2)
    
    # Calculate diagonal length of person box to normalize threshold
    person_diag = np.sqrt((p_x2 - p_x1)**2 + (p_y2 - p_y1)**2)
    
    # Use a normalized threshold (e.g., 30% of person diagonal)
    # FIX: Increased threshold from 0.2 to 0.3 to better associate distant PPE like vests/helmets.
    normalized_threshold = person_diag * 0.3 
    
    return distance < normalized_threshold

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
        # Handle grayscale or single channel image conversion to BGR if necessary
        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_array

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # ----------------------------------------------------------------
    # YOLOv11 Inference
    # ----------------------------------------------------------------
    with st.spinner("🔍 Analyzing PPE compliance..."):
        # Ensure the model receives an array in the expected BGR format
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
        
        # Find all associated PPE items (positive and violation labels)
        associated_ppe_classes = []
        for ppe in ppe_items:
            ppe_bbox = ppe["bbox"]
            
            # Check if PPE is sufficiently close to the person (proximity check is more reliable for body-worn items)
            if is_nearby(person_bbox, ppe_bbox):
                associated_ppe_classes.append(ppe["class"])
        
        # --- CRITICAL FIX: Calculate unique PPE *types* present ---
        # This prevents overcounting if, e.g., both 'Hardhat' and 'Helmet' are detected.
        detected_ppe_types = set()
        
        # Check for required PPE labels only
        for ppe_class in set(associated_ppe_classes) & REQUIRED_PPE:
            ppe_class_lower = ppe_class.lower()
            for ppe_type, keywords in PPE_TYPE_MAP.items():
                if any(k in ppe_class_lower for k in keywords):
                    # Add the high-level type (e.g., "Hardhat/Helmet") to the set
                    detected_ppe_types.add(ppe_type)
                    break
        
        ppe_count = len(detected_ppe_types)
        
        # Check for explicit violation items (e.g., NO-Hardhat) associated with the person
        # This is a critical safety check: any "NO" detection should fail the compliance.
        has_violations = bool(set(associated_ppe_classes) & VIOLATION_ITEMS)
        
        # Determine compliance status
        # Rule 1: Fully Compliant (all 3 types present AND no explicit violation)
        if ppe_count == TOTAL_REQUIRED_PPE and not has_violations:
            status = "FULLY COMPLIANT"
            box_color = (0, 255, 0)  # Green (BGR format)
            compliant += 1
        # Rule 2: Non-Compliant (0 types present OR an explicit violation detected)
        elif ppe_count == 0 or has_violations:
            status = "NON-COMPLIANT"
            box_color = (0, 0, 255)  # Red (BGR format)
            non_compliant += 1
        # Rule 3: Partially Compliant (1 or 2 types present, and no explicit violation)
        else: # ppe_count > 0 and ppe_count < TOTAL_REQUIRED_PPE and not has_violations
            status = "PARTIALLY COMPLIANT"
            box_color = (0, 255, 255)  # Yellow (BGR format)
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
        
        # Add status label (PPE count and compliance status)
        label = f"PPE: {ppe_count}/{TOTAL_REQUIRED_PPE} ({status.split()[0].upper()})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for the label
        cv2.rectangle(output_image, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), box_color, -1)
        
        # Put the text on top
        cv2.putText(output_image, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw PPE item detections (smaller boxes)
    for ppe in ppe_items:
        x1, y1, x2, y2 = ppe["bbox"]
        cls_name = ppe["class"]
        conf = ppe["confidence"]
        
        # Determine color based on item type (using BGR format)
        if "no" in cls_name.lower() or "without" in cls_name.lower():
            color = (0, 0, 255)  # Red for violations
        elif "mask" in cls_name.lower():
            color = (255, 255, 0)  # Cyan for mask
        elif "vest" in cls_name.lower():
            color = (0, 165, 255)  # Orange/Brown for vest (B:0, G:165, R:255)
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
    
    # Explicitly map colors for pie chart for consistency
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
    st.markdown(f"""
    **Compliance Legend:**
    - 🟢 **Green Box (Fully Compliant)**: {TOTAL_REQUIRED_PPE}/{TOTAL_REQUIRED_PPE} PPE items detected (Hardhat, Mask, Safety Vest) **AND** no violations.
    - 🟡 **Yellow Box (Partially Compliant)**: 1/{TOTAL_REQUIRED_PPE} or 2/{TOTAL_REQUIRED_PPE} PPE items detected, and no violations.
    - 🔴 **Red Box (Non-Compliant)**: 0/{TOTAL_REQUIRED_PPE} PPE items detected **OR** any explicit violation detected (e.g., 'NO-Hardhat').
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
### Compliance Color Coding (Based on PPE Count out of {TOTAL_REQUIRED_PPE})
- 🟢 **Green**: {TOTAL_REQUIRED_PPE}/{TOTAL_REQUIRED_PPE} PPE items present (Fully Compliant)
- 🟡 **Yellow**: 1/{TOTAL_REQUIRED_PPE} or 2/{TOTAL_REQUIRED_PPE} PPE items present (Partially Compliant)
- 🔴 **Red**: 0/{TOTAL_REQUIRED_PPE} PPE items present (Non-Compliant)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This AI system monitors workplace safety compliance and PPE usage using YOLOv11 object detection.")
