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

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.Resize((224, 224)), #dampens resolution. 
    transforms.ToTensor()           
])
dataset = CustomBaseballDataset(root_dir = "C:\\Users\jared\OneDrive\Grad Year Two\Forecasting\Project",
                                transform=transform)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
remaining_size = dataset_size - train_size

lengths = [train_size, remaining_size]


torch.manual_seed(67)
train_dataset, val_dataset = random_split(dataset, lengths)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# For reading data
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For visualizing
import plotly.express as px

# For model building
import torch
import torch.nn as nn
import torch.nn.functional as F
class FirstNet(nn.Module):
    def __init__(self):
      # We define the components of our model here
      super(FirstNet, self).__init__()
      # Function to flatten our image
      self.flatten = nn.Flatten()
      # Create the sequence of our network
      self.linear_relu_model = nn.Sequential(
            # Add a linear output layer w/ 10 perceptrons
            nn.LazyLinear(10),
        )

    def forward(self, x):
      # We construct the sequencing of our model here
      x = self.flatten(x)
      # Pass flattened images through our sequence
      output = self.linear_relu_model(x)

      # Return the evaluations of our ten 
      #   classes as a 10-dimensional vector
      return output

# Create an instance of our model
model = FirstNet()
# Define some training parameters
learning_rate = 1e-2
batch_size = 16
epochs = 1

# Define our loss function
#   This one works for multiclass problems
loss_fn = nn.CrossEntropyLoss()
# Build our optimizer with the parameters from
#   the model we defined, and the learning rate
#   that we picked
optimizer = torch.optim.SGD(model.parameters(),
     lr=learning_rate)
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    # important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # Loop over batches via the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and looking for improved gradients
        loss.backward()
        optimizer.step()
        # Zeroing out the gradient (otherwise they are summed)
        #   in preparation for next round
        optimizer.zero_grad()

        # Print progress update every few loops
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    # important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    print("Running test loop...")  # indicator that testing has started

    # Evaluating the model with torch.no_grad() ensures
    # that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations
    # and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if batch % 10 == 0:
                print(f"  [Batch {batch+1}/{num_batches}] running...")

    # Printing some output after a testing round
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {
        (100*correct):>0.1f}%, Avg loss: {
            test_loss:>8f} \n")
    
# Need to repeat the training process for each epoch.
#   In each epoch, the model will eventually see EVERY
#   observations in the data
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(val_loader, model, loss_fn=loss_fn)
print("Done!")

# Save our model for later, so we can train more or make predictions

EPOCH = epochs
# We use the .pt file extension by convention for saving
#    pytorch models
PATH = "model.pt"

# The save function creates a binary storing all our data for us
torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
# Specify our path
PATH = "model.pt"

# # Create a new "blank" model to load our information into
# model = FirstNet()

# # Recreate our optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Load back all of our data from the file
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# EPOCH = checkpoint['epoch']