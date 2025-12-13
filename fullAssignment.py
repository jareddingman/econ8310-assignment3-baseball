import os
import xml.etree.ElementTree as ET
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class BaseballData(Dataset):
    def __init__(self, video_dir, annotation_dir, transform = None, frameRate = 1):
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.frameRate = frameRate

        self.samples, self.label_map = self.loadDaVids()

    def getFrames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return print(f"Uh oh. Can't open {video_path}")
        
        frames = []
        i = 0
        while True:
            pull, frame = cap.read()
            if not pull:
                break
            if i % self.frameRate == 0: #had to look this part up
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() #permute reorders the dimensions of a tensor according to a specified ordering.
                frames.append(frame)
            i+= 1

        cap.release() #this part drove me nuts
        return frames
    
    def readDaAnnotations(self, xml_path):
        #I know Dusty talked about doing this with another library, but I chose xml.etree.ElementTree
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frameBoxes = {}
        label_map = {}
        next_label_id = 0

        for track in root.findall("track"): #had AI debug the rest of this function
            label = track.attrib["label"]
            if label not in label_map:
                label_map[label] = next_label_id
                next_label_id +=1

            for box in track.findall("box"):
                frame_id = int(box.attrib["frame"])
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])

                if frame_id not in frameBoxes:
                    frameBoxes[frame_id] = []
                frameBoxes[frame_id].append({
                    "bbox": [xtl, ytl, xbr, ybr],
                    "label": label_map[label]
                })
        return frameBoxes, label_map
    
    def matchVid(self, xmlName, video_files):
        originName = os.path.splitext(xmlName)[0].lower()
        #way better than regex here. tysm stack

        for k in video_files:
            if originName in k.lower():
                return k
            
        for k in video_files:
            if k.lower().startswith(originName):
                return k
            
        return None
    
    def loadDaVids(self):
        totalPulls = []
        big_label_map = {}
        next_label = 0

        video_files = [l for l in os.listdir(self.video_dir)
                       if l.lower().endswith((".mp4", ".mov"))]
        xml_files = [l for l in os.listdir(self.annotation_dir)
                     if l.lower().endswith(".xml")]
        
        for xfile in xml_files:
            xml_path = os.path.join(self.annotation_dir, xfile) #I hate os so much
            pair = self.matchVid(xfile, video_files = video_files)
            if pair is None:
                print(f"no pair found for {xfile}")
                continue

        video_path = os.path.join(self.video_dir, pair)
        frames = self.getFrames(video_path=video_path)
        frameBoxes, miniLabelMap = self.readDaAnnotations(xml_path=xml_path)

        for label, miniID in miniLabelMap.items(): #omg .items(), duh
            if label not in big_label_map:
                big_label_map[label] = next_label #this lets us iterate
                next_label += 1

            for frameID, frame in enumerate(frames): #thanks Ben
                if frameID not in frameBoxes:
                    continue
                boxes = []
                labels = []
                for m in frameBoxes[frameID]:
                    boxes.append(m["bbox"])
                    labels.append(big_label_map[list(miniLabelMap.keys())[m["label"]]])
                totalPulls.append({ #had to look this up
                    "frame": frame,
                    "boxes": torch.tensor(boxes, dtype = torch.float32),
                    "labels": torch.tensor(labels, dtype = torch.int64)})
                
        if not totalPulls:
            print("Uh oh. No pulls gathered")

        return totalPulls, big_label_map
    
    #omg I really thought it was going to be this simple: (pulled from slides)

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["frame"]

        if self.transform:
            image = self.transform(image)

        label = sample["labels"][0] #just for testing, not full 
        label = int(label) #makes this thing an array

        return image, label
    def makebbox(self, bbox, img_w, img_h):
        xtl, ytl, xbr, ybr = bbox
        x_center = ((xtl + xbr) / 2) / img_w
        y_center = ((ytl + ybr) / 2) / img_h
        width = (xbr - xtl) / img_w
        height = (ybr - ytl) / img_h

        return x_center, y_center, width, height
    '''
    The function export_yolo_dataset modifies your output directory from CVAT annotations to YOLO annotations. You need to get the dataset using BaseballData() first, then use dataset.export_yolo_dataset(outputdir = "yolo_dataset")
    '''
    def export_yolo_dataset(self, output_dir, image_ext=".jpg"):
        images_out = os.path.join(output_dir, "images")
        labels_out = os.path.join(output_dir, "labels")

        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        video_files = [v for v in os.listdir(self.video_dir)
                    if v.lower().endswith((".mp4", ".mov"))]
        xml_files = [x for x in os.listdir(self.annotation_dir)
                    if x.lower().endswith(".xml")]

        global_label_map = {}
        next_label_id = 0
        image_counter = 0

        for xml_file in xml_files:
            xml_path = os.path.join(self.annotation_dir, xml_file)
            video_name = self.matchVid(xml_file, video_files)

            if video_name is None:
                print(f"Damn. No matching video for {xml_file}")
                continue

            video_path = os.path.join(self.video_dir, video_name)
            frames = self.getFrames(video_path)
            frame_boxes, local_label_map = self.readDaAnnotations(xml_path)

            # Sync local labels into global YOLO labels, THE REST OF THIS METHOD IS TAKEN FROM COLAB
            for label in local_label_map:
                if label not in global_label_map:
                    global_label_map[label] = next_label_id
                    next_label_id += 1

            for frame_id, frame in enumerate(frames):
                if frame_id not in frame_boxes:
                    continue

                _, img_h, img_w = frame.shape

                image_name = f"frame_{image_counter:06d}{image_ext}"
                label_name = f"frame_{image_counter:06d}.txt"

                image_path = os.path.join(images_out, image_name)
                label_path = os.path.join(labels_out, label_name)

                # Save image
                img_np = frame.permute(1, 2, 0).byte().numpy()
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, img_bgr)

                # Write YOLO labels
                with open(label_path, "w") as f:
                    for obj in frame_boxes[frame_id]:
                        label_id = global_label_map[
                            list(local_label_map.keys())[obj["label"]]
                        ]

                        x, y, w, h = self.cvat_to_yolo_bbox(
                            obj["bbox"], img_w, img_h
                        )

                        f.write(f"{label_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                image_counter += 1

        print("YOLO export complete.")
        print("Label map:")
        for k, v in global_label_map.items():
            print(f"{v}: {k}")
'''
The commented out section below is if you would like to get the dataset on your own computer. You need to change video_dir (a folder of raw vids) and annotation_dir(a folder of annotation xml files).
'''

# video_dir = r"C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project_Extra\Raw Videos"
# annotation_dir = r"C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project_Extra\CVAT Annotations"

# transform = transforms.Compose([ #not necessary, but it helped me train the model way quicker
#     transforms.Resize((224, 224)),
# ])

# dataset = BaseballData(video_dir, annotation_dir, transform=transform, frameRate=1)

# train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# dataset_size = len(dataset)
# train_size = int(dataset_size * 0.8)
# remaining_size = dataset_size - train_size

# lengths = [train_size, remaining_size]


# torch.manual_seed(1811441513)
# print(f"Total samples in dataset: {len(dataset)}")

# train_dataset, val_dataset = random_split(dataset, lengths)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)







