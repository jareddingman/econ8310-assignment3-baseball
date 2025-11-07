import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomBaseballDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all (image_path, boxes, labels) tuples
        self.samples, self.label_map = self._parse_all_annotations()

        print(f"Loaded {len(self.samples)} samples with labels: {self.label_map}")

    def _parse_all_annotations(self):
        all_samples = []
        label_map = {} 
        next_label_id = 0

        for video_folder in os.listdir(self.root_dir):
            video_path = os.path.join(self.root_dir, video_folder)
            if not os.path.isdir(video_path):
                continue

            xml_path = os.path.join(video_path, "annotations.xml")
            images_dir = os.path.join(video_path, "images")

            if not os.path.exists(xml_path):
                continue

            tree = ET.parse(xml_path) #ET is very very useful for xml files
            root = tree.getroot()

            for track in root.findall("track"):
                label = track.attrib["label"]

                if label not in label_map:
                    label_map[label] = next_label_id
                    next_label_id += 1

                for box in track.findall("box"): #used AI for this part
                    frame_id = int(box.attrib["frame"])
                    xtl = float(box.attrib["xtl"])
                    ytl = float(box.attrib["ytl"])
                    xbr = float(box.attrib["xbr"])
                    ybr = float(box.attrib["ybr"])

                    frame_name = f"frame_{frame_id:06d}.jpg"
                    frame_path = os.path.join(images_dir, frame_name)

                    if os.path.exists(frame_path):
                        all_samples.append({
                            "image_path": frame_path,
                            "boxes": torch.tensor([[xtl, ytl, xbr, ybr]], dtype=torch.float32),
                            "labels": torch.tensor([label_map[label]], dtype=torch.int64),
                        })


        return all_samples, label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        labels = sample["labels"] 

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, labels.squeeze(0)  #google told me to do this
dataset = CustomBaseballDataset(root_dir = "C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project")
print("Number of samples found:", len(dataset))
