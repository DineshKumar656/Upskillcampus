from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="agri_data/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device="cpu",
    name="agri_yolov8_cpu_train"
)
