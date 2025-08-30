🌱 Crop and Weed Detection using YOLOv8
📌 Project Overview

This project implements an AI-powered system to detect and classify crops and weeds using the YOLOv8 (You Only Look Once) object detection framework.
Weeds compete with crops for nutrients, water, and land, which reduces crop yield. Farmers often rely on pesticides for weed control, but excessive use of chemicals can harm crops, soil, and human health.

This solution provides a real-time, computer-vision-based alternative that reduces dependence on pesticides and promotes precision agriculture.

🎯 Objectives

Detect and classify crops vs. weeds using YOLOv8.

Prepare and annotate a custom dataset in YOLO format.

Train YOLOv8 on the dataset and evaluate model performance.

Implement inference on images, videos, and live webcam streams.

Develop a Tkinter GUI for user-friendly weed detection.

Export detection results (bounding boxes, labels, confidence scores) into CSV files.

🛠️ Tools and Technologies

YOLOv8 (Ultralytics) – Object detection framework.

Python – Core programming language.

PyTorch – Deep learning backend for YOLOv8.

OpenCV (cv2) – Image and video processing.

Tkinter – GUI development.

Pandas/CSV – Data handling and result storage.

📂 Project Structure
Crop-Weed-Detection/
│── agri_data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
│── train.py              # Model training script
│── inference.py          # Inference on images
│── inferencevideo.py     # Inference on video files
│── inferencewebcam.py    # Inference on webcam
│── gui_inference.py      # Tkinter GUI application
│── yolov8n.pt            # Pretrained weights
│
└── README.md

🚀 Getting Started
1️⃣ Clone Repository
git clone https://github.com/yourusername/Crop-Weed-Detection-YOLOv8.git
cd Crop-Weed-Detection-YOLOv8

2️⃣ Install Dependencies
pip install ultralytics opencv-python pillow pandas tk

3️⃣ Train Model
python train.py

4️⃣ Run Inference

On Image

python inference.py


On Video

python inferencevideo.py


On Webcam

python inferencewebcam.py

5️⃣ GUI Application
python gui_inference.py

📊 Results

Classes: Crop (0), Weed (1)

Model trained for 50 epochs on a custom dataset.

Achieved reliable detection for crops and weeds, producing bounding boxes and CSV logs.

Outputs are saved under:

runs/detect/train/weights/best.pt   # Best model weights
runs/detect/predict/exp/            # Annotated images/videos
detection_output.csv                # CSV of detections

⚡ Constraints & Considerations

Memory & Speed: Optimized using YOLOv8n for CPU-friendly training.

Accuracy: Achieved good results, but can be improved with larger datasets.

Scalability: Future integration possible with drones/IoT devices for industrial use.

🔮 Future Scope

Drone-based weed detection for large-scale farms.

Automated weed removal with robotic sprayers.

Deployment on edge AI devices (NVIDIA Jetson, Raspberry Pi).

Multi-class detection (different weed species, crop health monitoring).

🙏 Acknowledgments

UCT (Unicoverage Technologies) for providing the internship opportunity and structured guidance.

Mentors and faculty for continuous support.

Open-source contributors of YOLOv8 (Ultralytics) for the framework and documentation.

📜 License

This project is released under the MIT License. You are free to use, modify, and distribute it with attribution.

Sample Outputs

![val_batch2_labels](https://github.com/user-attachments/assets/ca91a377-f983-4bc3-9a5a-563f5b313f14)
