from ultralytics import YOLO

# Load the trained model
model = YOLO(r"agri_data/agri_yolov8_cpu_train6/weights/best.pt")

# Path to your input video
video_path = r"C:\Users\hp\Documents\PROJET's\Project5_Ag_Crop and weed detection\test_videos\896754f0a2dd8dee0976a3497c3c1acc.mp4" # ✅ change this to your video file path

# Predict from video
results = model.predict(
    source=video_path,
    conf=0.5,
    imgsz=640,
    save=True,
    show=True,
    project='runs/detect',
    name='video_output',
    exist_ok=True
)
