import os
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- Environment setup ---
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- Paths and config ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Safty.pt"

# --- Streamlit page setup ---
st.set_page_config(layout="wide", page_title="PPE Compliance Detection", page_icon="⚠️")
st.title("👷 PPE Compliance Detection System")

# --- Load YOLO model ---
@st.cache_resource
def load_yolo_model(path: Path):
    try:
        model = YOLO(str(path))
        model.to("cpu")
        st.success(f"✅ YOLO model loaded: {path.name}")
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

# --- Detect PPE and violations from model labels ---
all_labels = set(model.names.values())

REQUIRED_PPE = {
    lbl for lbl in all_labels
    if any(k.lower() in lbl.lower() for k in ["hardhat", "mask", "safety vest", "helmet", "vest"])
}
VIOLATION_ITEMS = {
    lbl for lbl in all_labels
    if lbl.lower().startswith("no") or "without" in lbl.lower()
}

# Fallback defaults
if not REQUIRED_PPE:
    REQUIRED_PPE = {"Hardhat", "Mask", "Safety Vest"}
if not VIOLATION_ITEMS:
    VIOLATION_ITEMS = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# --- Helper functions ---
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_nearby(person_bbox, ppe_bbox, scale=0.5):
    """Check if PPE bbox is near a person bbox based on relative size."""
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_bbox
    p_center_x, p_center_y = (p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2
    ppe_center_x, ppe_center_y = (ppe_x1 + ppe_x2) / 2, (ppe_y1 + ppe_y2) / 2
    distance = np.sqrt((p_center_x - ppe_center_x) ** 2 + (p_center_y - ppe_center_y) ** 2)
    person_height = p_y2 - p_y1
    threshold = person_height * scale
    return distance < threshold


# --- Session state ---
if "detection_history" not in st.session_state:
    st.session_state["detection_history"] = []

# --- File upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if len(img_array.shape) == 3 else img_array

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # --- Run inference ---
    with st.spinner("🔍 Analyzing PPE compliance..."):
        results = model(img_bgr, conf=0.25)

    detections, output_image = [], img_bgr.copy()
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names[cls]
            x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))  # ✅ Fixed unpack
            detections.append({"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)})

    persons = [d for d in detections if d["class"].lower() == "person"]
    ppe_items = [d for d in detections if d["class"] in REQUIRED_PPE or d["class"] in VIOLATION_ITEMS]

    compliant = partial_compliant = non_compliant = 0
    person_compliance = []

    # --- Compliance logic ---
    for person in persons:
        person_bbox = person["bbox"]
        associated_ppe = [
            ppe["class"]
            for ppe in ppe_items
            if calculate_iou(person_bbox, ppe["bbox"]) > 0.1 or is_nearby(person_bbox, ppe["bbox"])
        ]
        detected_ppe_set = set(associated_ppe) & REQUIRED_PPE
        has_violations = bool(set(associated_ppe) & VIOLATION_ITEMS)
        ppe_count = len(detected_ppe_set)

        if ppe_count == 3 and not has_violations:
            status, box_color = "compliant", (0, 255, 0)
            compliant += 1
        elif ppe_count == 0 or has_violations:
            status, box_color = "non_compliant", (0, 0, 255)
            non_compliant += 1
        else:
            status, box_color = "partial", (0, 255, 255)
            partial_compliant += 1

        person_compliance.append({
            "bbox": person_bbox,
            "status": status,
            "color": box_color,
            "ppe_count": ppe_count,
            "detected_types": list(detected_ppe_set),
            "all_associated": list(set(associated_ppe))
        })

        # Draw person box
        x1, y1, x2, y2 = person_bbox
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 4)
        label = f"PPE: {ppe_count}/3"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(output_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Draw PPE boxes ---
    for ppe in ppe_items:
        x1, y1, x2, y2 = ppe["bbox"]
        cls_name, conf = ppe["class"], ppe["confidence"]

        if "no" in cls_name.lower() or "without" in cls_name.lower():
            color = (0, 0, 255)
        elif "mask" in cls_name.lower():
            color = (255, 255, 0)
        elif "vest" in cls_name.lower():
            color = (0, 165, 255)
        elif "helmet" in cls_name.lower() or "hardhat" in cls_name.lower():
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # --- Display results ---
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    with col2:
        st.image(output_rgb, caption="Detection Results", use_column_width=True)

    # --- Metrics ---
    person_count = len(persons)
    st.session_state["detection_history"].append({
        "total": person_count,
        "compliant": compliant,
        "partial": partial_compliant,
        "non_compliant": non_compliant,
        "timestamp": pd.Timestamp.now()
    })

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
    fig = px.pie(compliance_data, values="Count", names="Status",
                 title="PPE Compliance Distribution",
                 color_discrete_sequence=["green", "yellow", "red"])
    st.plotly_chart(fig)

    st.markdown("""
    **Compliance Legend:**
    - 🟢 Green: All 3 PPE items detected (Hardhat, Mask, Safety Vest)
    - 🟡 Yellow: 1 or 2 PPE items detected
    - 🔴 Red: No PPE detected or violations present
    """)

    # --- Alerts and Trends ---
    if non_compliant > 0:
        st.error(f"⚠️ ALERT: {non_compliant} workers detected without proper PPE!")
    if partial_compliant > 0:
        st.warning(f"⚠️ WARNING: {partial_compliant} workers with partial PPE compliance")

    if len(st.session_state["detection_history"]) > 1:
        st.subheader("📈 Compliance Trends")
        hist_df = pd.DataFrame(st.session_state["detection_history"])
        fig_trend = px.line(hist_df, x="timestamp",
                            y=["compliant", "partial", "non_compliant"],
                            title="Compliance Trends Over Time")
        st.plotly_chart(fig_trend)

else:
    st.info("👆 Please upload an image to begin PPE compliance detection")
