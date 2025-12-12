import os
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

'''
BEFORE YOU RUN!! This is designed to train a YOLO model in conjunction with the YOLO converter (see folder).
'''

project_dir = #r"C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project" CHANGE THIS TO YOUR OWN DIRECTORY
output_dir = os.path.join(project_dir, "yolo_data")

# Create YOLO folders
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

# Collect all (image, label, video_name)
image_label_pairs = []

for video_folder in os.listdir(project_dir):
    video_path = os.path.join(project_dir, video_folder)
    if not os.path.isdir(video_path):
        continue
    img_dir = os.path.join(video_path, "images")
    lbl_dir = os.path.join(video_path, "labels")
    if not (os.path.exists(img_dir) and os.path.exists(lbl_dir)):
        continue
    
    for file in os.listdir(img_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(img_dir, file)
            lbl_path = os.path.join(lbl_dir, file.replace(".jpg", ".txt"))
            if os.path.exists(lbl_path):
                image_label_pairs.append((img_path, lbl_path, video_folder))

print(f"Found {len(image_label_pairs)} image-label pairs total.")

# Split into train/val
train_pairs, val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=42)

def copy_pairs(pairs, split):
    for img_path, lbl_path, vid_name in pairs:
        base_name = f"{vid_name}_{os.path.basename(img_path)}"
        new_img_path = os.path.join(output_dir, "images", split, base_name)
        new_lbl_path = os.path.join(output_dir, "labels", split, base_name.replace(".jpg", ".txt"))
        
        shutil.copy(img_path, new_img_path)
        shutil.copy(lbl_path, new_lbl_path)

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print(f"Copied {len(train_pairs)} train and {len(val_pairs)} val pairs.")


output_dir = os.path.join(#r"C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project", "yolo_data") CHANGE THIS TO YOUR OWN DIRECTORY INFO

data = {
    "train": os.path.join(output_dir, "images/train").replace("\\", "/"),
    "val": os.path.join(output_dir, "images/val").replace("\\", "/"),
    "nc": 2,  # number of classes (moving_baseball or baseball)
    "names": ["baseball", "moving_baseball"]  
}

yaml_path = os.path.join(output_dir, "data.yaml")

with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"data.yaml created at: {yaml_path}")

model = YOLO("yolov8n.pt")

model.train(data="C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project\yolo_data\data.yaml",
            epochs = 30,
            imgsz = 640,
            batch = 8)
