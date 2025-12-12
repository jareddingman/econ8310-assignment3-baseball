import os
import xml.etree.ElementTree as ET
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

#CVAT to YOLO
"""
It takes the xml files in CVAT form and make it readable to YOLO.
"""

def convert_cvat_video_to_yolo(xml_path, images_dir, labels_dir, class_map):

    os.makedirs(labels_dir, exist_ok=True) #sanity check

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for track in root.findall("track"):
        base_label = track.get("label").strip().lower() #this beats regex for name change

        for box in track.findall("box"):
            frame_num = int(box.get("frame"))
            image_name = f"frame_{frame_num:06d}.jpg"
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                continue  # skip if image not found, very important

            img = cv2.imread(image_path) #detect image
            if img is None:
                continue  # skip if unreadable
            img_height, img_width = img.shape[:2]

            # Check for 'moving'
            moving_attr = None
            for attr in box.findall("attribute"):
                attr_name = attr.get("name").strip().lower()
                attr_value = attr.text.strip().lower()
                if "moving" in attr_name and attr_value == "true":
                    moving_attr = True
                    break

            # Choose label name
            label_name = base_label
            if moving_attr:
                label_name = f"moving_{base_label}"

            label_name = label_name.lower()
            if label_name not in class_map:
                continue
            class_id = class_map[label_name]

            # Bounding box (major assistance from GitHub user Koldim2001)
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            width = xbr - xtl
            height = ybr - ytl
            x_center = xtl + width / 2
            y_center = ytl + height / 2

            # Normalize coordinates
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # Write YOLO label file (append in case multiple boxes per frame)
            label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
            with open(label_path, "a") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Converted {os.path.basename(xml_path)} to YOLO format.")


#Batch conversion for all videos------------------------------

#Make this YOUR OWN DIRECTORY
project_dir = #r"C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project"

# Define class map dictionary — all lowercase!!!
class_map = {
    "baseball": 0,
    "moving_baseball": 1
}

for video_folder in os.listdir(project_dir):
    video_path = os.path.join(project_dir, video_folder)
    xml_path = os.path.join(video_path, "annotations.xml")
    images_dir = os.path.join(video_path, "images")
    labels_dir = os.path.join(video_path, "labels")

    if os.path.exists(xml_path) and os.path.exists(images_dir):
        print(f"\nProcessing {video_folder}")
        convert_cvat_video_to_yolo(xml_path, images_dir, labels_dir, class_map)
    else:
        print(f"Skipping {video_folder}: missing XML or images folder")

