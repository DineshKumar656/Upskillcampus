from ultralytics import YOLO

# Load the trained model
model = YOLO(r"agri_data/agri_yolov8_cpu_train6/weights/best.pt")

# Predict from webcam (0 = default webcam)
results = model.predict(
    source=0,                # 👈 Webcam index (0 = default)
    conf=0.5,                # Confidence threshold
    imgsz=640,               # Image size
    show=True,               # 👁️ Show real-time window
    save=True,               # Save the output video
    project='runs/detect',
    name='webcam_output',
    exist_ok=True
)
