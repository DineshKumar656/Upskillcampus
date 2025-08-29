import os
import re
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("agri_data/agri_yolov8_cpu_train6/weights/best.pt")

# Create main window
root = tk.Tk()
root.title("Crop and Weed Detection GUI")
root.geometry("720x600")

# GUI image display
panel = tk.Label(root)
panel.pack(pady=10)

def run_inference(image_path):
    # Run inference
    results = model.predict(image_path, save=False)
    result = results[0]

    # Convert result to image
    result_img_array = result.plot()
    result_img = Image.fromarray(result_img_array)

    # Clean filename
    original_name = os.path.basename(image_path)
    safe_name = re.sub(r'[^\w\-_.]', '_', original_name)

    # Output folder
    save_dir = os.path.join("runs", "detect", "predict_gui")
    os.makedirs(save_dir, exist_ok=True)

    # Save result image
    result_image_path = os.path.join(save_dir, safe_name)
    result_img.save(result_image_path)

    # ✅ Save detections as CSV using `to_df()`
    df = result.to_df()
    csv_path = os.path.join(save_dir, "detection_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ Image saved to: {result_image_path}")
    print(f"📄 CSV saved to: {csv_path}")

    # Show result in GUI
    img_tk = ImageTk.PhotoImage(result_img.resize((640, 400)))
    panel.config(image=img_tk)
    panel.image = img_tk

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        run_inference(file_path)

# Browse button
btn = tk.Button(root, text="Browse Image", command=browse_file,
                font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10)
btn.pack(pady=20)

root.mainloop()
