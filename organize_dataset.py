import os
import random
import shutil
from pathlib import Path

# Set base data path
base_path = Path("agri_data/data")
image_ext = ".jpeg"

# Collect all image files
image_files = [f for f in os.listdir(base_path) if f.endswith(image_ext)]
random.shuffle(image_files)

# Create train/val split (80/20)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Create target folders
for subset in ['train', 'val']:
    (base_path / f"images/{subset}").mkdir(parents=True, exist_ok=True)
    (base_path / f"labels/{subset}").mkdir(parents=True, exist_ok=True)

# Move files
def move_files(files, subset):
    for file in files:
        img_src = base_path / file
        lbl_src = base_path / file.replace(".jpeg", ".txt")

        img_dst = base_path / f"images/{subset}" / file
        lbl_dst = base_path / f"labels/{subset}" / file.replace(".jpeg", ".txt")

        shutil.move(img_src, img_dst)
        shutil.move(lbl_src, lbl_dst)

move_files(train_files, "train")
move_files(val_files, "val")

print(f"✅ Dataset organized: {len(train_files)} training and {len(val_files)} validation images moved.")
