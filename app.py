import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os
from io import BytesIO

# -----------------------------
# Load trained YOLOv8 model
# -----------------------------
@st.cache_resource
def load_model():
    # Change path if needed
    model_path = "E:/github/blood_cell_detection_results/model/best.pt"
    model = YOLO(model_path)
    return model

model = load_model()
st.set_page_config(page_title="Blood Cell Detection", layout="centered")
st.title("🩸 Blood Cell Detection App (YOLOv8)")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload a blood smear image", type=["jpg","jpeg","png"])

def detect_and_annotate(image, model, conf_thresh=0.25):
    # Convert PIL to numpy (RGB -> BGR for OpenCV)
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Predict using YOLOv8
    results = model.predict(img_np, conf=conf_thresh, imgsz=640)
    annotated = results[0].plot(line_width=3)  # numpy array BGR

    # Convert annotated to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Count cells by class
    class_counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        class_counts[label] = class_counts.get(label, 0) + 1

    return annotated_rgb, class_counts

# -----------------------------
# Main logic
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting blood cells...")

    annotated_image, cell_counts = detect_and_annotate(image, model)

    st.image(annotated_image, caption="Prediction", use_column_width=True)

    # Display cell counts
    st.subheader("Detected Cells Count")
    for cls, count in cell_counts.items():
        st.write(f"{cls}: {count}")

    # Allow user to download annotated image
    _, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    st.download_button(
        label="Download Annotated Image",
        data=BytesIO(buffer.tobytes()),
        file_name="annotated_blood_cells.png",
        mime="image/png"
    )
else:
    st.info("Upload an image to start detection.")
