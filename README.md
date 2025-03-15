# 🚀 YOLOv5 Object Detection with Streamlit

## 📌 Overview
This repository provides an end-to-end pipeline for object detection using YOLOv5. It includes data preprocessing, model training, and a user-friendly Streamlit application for real-time image and video inference. 

## 📂 Repository Structure
```
📦 YOLOv5-Streamlit-Detection
├── 📁 model/            # Contains training summary images (F1 curve, confusion matrix, etc.)
├── 📁 test_images/      # Test images and videos for inference
├── 📄 app_pred.py       # Streamlit application for object detection
├── 📄 data.yaml         # Dataset configuration file
├── 📄 yolo_pred.ipynb   # Preprocessing and inference code
├── 📄 yolo_train.ipynb  # Model training and weight saving
├── 📁 weights/          # Directory for trained model weights
```

## ⚙️ Installation
Ensure you have Python installed, then clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Training the Model
To train the YOLOv5 model:
1. Open `yolo_train.ipynb` in Jupyter Notebook or Google Colab.
2. Run the notebook cells to preprocess data, train the model, and save the trained weights.
3. The model weights will be saved in the `weights/` directory.

## 🖼️ Running Inference
To test the model on new images and videos:
1. Open `yolo_pred.ipynb` and run the cells to preprocess test data and perform inference.
2. Test images and videos should be placed in the `test_images/` folder.

## 🌐 Streamlit Application
To run the Streamlit-based object detection app:
```bash
streamlit run app_pred.py
```
This will launch a web interface where you can upload images and videos for real-time object detection.

## 🖥️ Screenshots
### 🔹 Streamlit App Interface:
![Streamlit App](screenshots/.png)

### 🔹 Object Detection on Image:
![Object Detection Image](screenshots/detection_image.png)

### 🔹 Object Detection on Video:
![Object Detection Video](screenshots/detection_video.png)

## 📊 Dataset Configuration
The dataset details are specified in `data.yaml`. This file includes:
- 🔢 Number of classes (`nc`)
- 🏷️ Class names
- 📍 Path to training and validation data

## 📈 Results & Model Performance
Training performance metrics such as:
✅ F1-score
✅ Precision
✅ Recall
✅ Confusion Matrix

Can be found in the `model/` directory. 📊

## 🤝 Contributing
Feel free to fork this repository, make improvements, and submit a pull request! 🚀

## 📜 License
This project is open-source and available under the MIT License.

