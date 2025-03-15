import streamlit as st
import os
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import tempfile

# Configuration
st.set_page_config(page_title="YOLO Object Detection", layout="centered")

# Create necessary directories
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}

# YOLO Prediction Class
class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)
        
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
        
        detections = preds[0]
        boxes, confidences, classes = [], [], []
        image_w, image_h = input_image.shape[:2]
        x_factor, y_factor = image_w / INPUT_WH_YOLO, image_h / INPUT_WH_YOLO

        for row in detections:
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(confidence)
                    classes.append(class_id)
        
        index = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()
        for ind in index:
            x, y, w, h = boxes[ind]
            class_name = self.labels[classes[ind]]
            colors = self.generate_colors(classes[ind])
            
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            cv2.putText(image, f'{class_name}: {int(confidences[ind] * 100)}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors, 2)
        return image

    def generate_colors(self, ID):
        np.random.seed(ID)
        return tuple(np.random.randint(100, 255, 3).tolist())

def process_video(video_path, yolo):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Unable to open video.")
        return

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_frame = yolo.predictions(frame)
        
        if out is None:
            h, w, _ = pred_frame.shape
            out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (w, h))
        
        out.write(pred_frame)
    
    cap.release()
    out.release()
    return temp_output.name

# Load YOLO model
yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

# Streamlit UI
st.title("YOLO Object Detection")
st.write("Upload an image or video to detect objects using YOLO.")

uploaded_file = st.file_uploader("Choose an image or video...", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_file.type.startswith("image"):
        image = cv2.imread(file_path)
        processed_image = yolo.predictions(image.copy())
        original_pil = Image.open(uploaded_file)
        processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_pil, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_pil, caption="Processed Image", use_column_width=True)
        
    elif uploaded_file.type.startswith("video"):
        processed_video_path = process_video(file_path, yolo)
        st.video(processed_video_path)
