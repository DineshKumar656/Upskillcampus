from ultralytics import YOLO
import pandas as pd
import os

# ✅ 1. Load the trained model
model = YOLO(r"agri_data/agri_yolov8_cpu_train6/weights/best.pt")

# ✅ 2. Predict on test images
results = model.predict(
    source="test_images",  # folder containing test images
    save=True,             # save the prediction images
    imgsz=640,             # image size
    conf=0.5               # confidence threshold
)

# ✅ 3. Prepare to store detections
output_data = []

# ✅ 4. Extract results from each image
for r in results:
    image_path = r.path
    for box in r.boxes:
        cls = int(box.cls[0])                       # class index
        label = r.names[cls]                        # class name ('crop' or 'weed')
        conf = float(box.conf[0])                   # confidence score
        xyxy = box.xyxy[0].tolist()                 # bounding box [x1, y1, x2, y2]

        output_data.append({
            "image": os.path.basename(image_path),  # just filename
            "label": label,
            "confidence": round(conf, 3),
            "x1": round(xyxy[0], 2),
            "y1": round(xyxy[1], 2),
            "x2": round(xyxy[2], 2),
            "y2": round(xyxy[3], 2)
        })

# ✅ 5. Save results to CSV
os.makedirs("detection_logs", exist_ok=True)
df = pd.DataFrame(output_data)
df.to_csv("detection_logs/inference_results.csv", index=False)

print("✅ Detection results saved to: detection_logs/inference_results.csv")
