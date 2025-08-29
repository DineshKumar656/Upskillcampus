from ultralytics import YOLO

# Load your trained model
model = YOLO(r"agri_data/agri_yolov8_cpu_train6/weights/best.pt")  # update if needed

# Run inference on test image folder or a single image
results = model.predict(
    source="test_images",  # or path to a specific image
    save=True,             # save prediction results
    imgsz=640              # input image size
)
