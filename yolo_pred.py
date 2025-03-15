import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        # Load YAML configuration
        with open(data_yaml, 'r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Generate colors for classes
        np.random.seed(10)
        self.colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()

    def predictions(self, image):
        row, col, _ = image.shape
        max_rc = max(row, col)

        # Create a square image
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Preprocess image for YOLO
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Process detections
        detections = preds[0]
        boxes, confidences, classes = [], [], []

        x_factor = input_image.shape[1] / INPUT_WH_YOLO
        y_factor = input_image.shape[0] / INPUT_WH_YOLO

        for detection in detections:
            confidence = detection[4]
            if confidence > 0.4:
                class_scores = detection[5:]
                class_id = class_scores.argmax()
                class_score = class_scores[class_id]

                if class_score > 0.25:
                    cx, cy, w, h = detection[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    classes.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        if len(indices) == 0:
            return image

        for i in indices.flatten():
            x, y, w, h = boxes[i]
            conf = int(confidences[i] * 100)
            class_id = classes[i]
            label = f"{self.labels[class_id]}: {conf}%"
            color = tuple(self.colors[class_id])

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(image, (x, y - 20), (x + w, y), color, -1)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image


def process_video(video_path, yolo):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_frame = yolo.predictions(frame)
        cv2.imshow('YOLO Predictions', pred_frame)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

