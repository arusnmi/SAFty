import os

# --- Auto-disable file watcher on Streamlit Cloud ---
if "STREAMLIT_RUNTIME" in os.environ or os.environ.get("STREAMLIT_SHARING_MODE") == "true":
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    print("🔒 Streamlit Cloud detected — file watcher disabled to prevent inotify limit errors.")
else:
    print("💻 Running locally — file watcher remains enabled for auto-reload.")

from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import random

# --- Environment setup ---
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- Paths and config ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Safty.pt"

# --- Streamlit page setup ---
st.set_page_config(layout="wide", page_title="PPE Compliance Detection", page_icon="⚠️")
st.title("👷 PPE Compliance Detection System")

# --- Sidebar options ---
st.sidebar.header("🧩 Debug Options")
show_debug_overlay = st.sidebar.toggle("Show PPE-Person Association Debug Overlay", value=False)

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

# --- Define valid and invalid PPE classes explicitly ---
REQUIRED_PPE = {"Hardhat", "Mask", "Safety Vest"}
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
            x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))
            detections.append({"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)})

    persons = [d for d in detections if d["class"].lower() == "person"]
    ppe_items = [d for d in detections if d["class"] in REQUIRED_PPE or d["class"] in VIOLATION_ITEMS]

    compliant = partial_compliant = non_compliant = 0

    # --- Step 1: Exclusive PPE assignment ---
    person_ppe_map = {i: [] for i in range(len(persons))}
    person_colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(persons))}

    for ppe in ppe_items:
        if ppe["class"] in VIOLATION_ITEMS:
            continue  # Skip "NO-" items for compliance assignment

        best_iou, best_person_idx = 0, None
        for i, person in enumerate(persons):
            person_bbox = person["bbox"]
            iou = calculate_iou(person_bbox, ppe["bbox"])
            if iou > best_iou:
                best_iou, best_person_idx = iou, i

        # If no strong IoU, assign based on nearest person
        if best_person_idx is None or best_iou < 0.1:
            best_distance, best_person_idx = float("inf"), None
            for i, person in enumerate(persons):
                if is_nearby(person["bbox"], ppe["bbox"], scale=0.5):
                    p_x1, p_y1, p_x2, p_y2 = person["bbox"]
                    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe["bbox"]
                    p_center = ((p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2)
                    ppe_center = ((ppe_x1 + ppe_x2) / 2, (ppe_y1 + ppe_y2) / 2)
                    dist = np.sqrt((p_center[0] - ppe_center[0]) ** 2 + (p_center[1] - ppe_center[1]) ** 2)
                    if dist < best_distance:
                        best_distance, best_person_idx = dist, i

        if best_person_idx is not None:
            person_ppe_map[best_person_idx].append(ppe)

            # --- Debug overlay line from PPE to assigned person ---
            if show_debug_overlay:
                ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe["bbox"]
                ppe_center = ((ppe_x1 + ppe_x2) // 2, (ppe_y1 + ppe_y2) // 2)
                px1, py1, px2, py2 = persons[best_person_idx]["bbox"]
                person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                cv2.line(output_image, ppe_center, person_center, person_colors[best_person_idx], 2)

    # --- Step 2: Compliance determination per person ---
    for i, person in enumerate(persons):
        person_bbox = person["bbox"]
        detected_required = [p for p in person_ppe_map[i] if p["class"] in REQUIRED_PPE]
        ppe_count = min(len(detected_required), 3)  # cap at 3 max

        if ppe_count == 0:
            status, box_color, compliant_label = "non_compliant", (0, 0, 255), "0/3"
            non_compliant += 1
        elif ppe_count in (1, 2):
            status, box_color, compliant_label = "partial", (0, 255, 255), f"{ppe_count}/3"
            partial_compliant += 1
        else:
            status, box_color, compliant_label = "compliant", (0, 255, 0), "3/3"
            compliant += 1

        # Draw bounding box + label
        x1, y1, x2, y2 = person_bbox
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 4)
        label = f"PPE: {compliant_label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(output_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Step 3: Draw PPE boxes (including NO- items) ---
    for ppe in ppe_items:
        x1, y1, x2, y2 = ppe["bbox"]
        cls_name, conf = ppe["class"], ppe["confidence"]

        if cls_name in VIOLATION_ITEMS:
            color = (0, 0, 255)
        elif cls_name in REQUIRED_PPE:
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

    # --- Summary metrics ---
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

    # --- Detected valid PPE classes in the frame ---
    valid_classes_in_frame = sorted({ppe["class"] for ppe in ppe_items if ppe["class"] in REQUIRED_PPE})
    if valid_classes_in_frame:
        st.subheader("🧾 Detected Valid PPE Classes in Frame:")
        st.write(", ".join(valid_classes_in_frame))
    else:
        st.subheader("🧾 Detected Valid PPE Classes in Frame:")
        st.write("No valid PPE items detected.")

    st.markdown("""
    **Compliance Legend:**
    - 🟢 Green: 3 valid PPE items detected near the person  
    - 🟡 Yellow: 1 or 2 valid PPE items detected  
    - 🔴 Red: No PPE detected  
    *(Only valid PPE detections are counted; 'NO-' violation boxes are ignored.)*
    """)

    # --- Alerts and trends ---
    if non_compliant > 0:
        st.error(f"⚠️ ALERT: {non_compliant} workers detected without PPE!")
    if partial_compliant > 0:
        st.warning(f"⚠️ WARNING: {partial_compliant} workers partially compliant")

    if len(st.session_state["detection_history"]) > 1:
        st.subheader("📈 Compliance Trends")
        hist_df = pd.DataFrame(st.session_state["detection_history"])
        fig_trend = px.line(hist_df, x="timestamp",
                            y=["compliant", "partial", "non_compliant"],
                            title="Compliance Trends Over Time")
        st.plotly_chart(fig_trend)

else:
    st.info("👆 Please upload an image to begin PPE compliance detection")
