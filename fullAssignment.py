import os
import xml.etree.ElementTree as ET
import cv2
import torch
from torch.utils.data import Dataset
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
'''
The commented out section below is if you would like to get the dataset on your own computer. You need to change video_dir (a folder of raw vids) and annotation_dir(a folder of annotation xml files).
'''

from torchvision import transforms
from torch.utils.data import DataLoader, random_split

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



