"""
Text 2 Image with VQGAN Code
Author: Nicholas Mesa-Cucalon
10-623 Generative AI
"""

# Perform Imports
import sys
sys.path.append(".")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from t2i_model import T2IEncoderInput
from PIL import Image

# Define helper functions
def normalize_output(x):
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  return x

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Variables
batch_size  = 1
d_proj      = 512
d_hidden    = 512

# Initialize model
config_path = "logs/vqgan_gumbel_f8/configs/model.yaml"
chkpt_path = "logs/vqgan_gumbel_f8/checkpoints/last.ckpt"
model = T2IEncoderInput(d_proj, d_hidden, config_path, chkpt_path, True)
model.to(device)


# Initialize dataloader
transform = T.Compose([
    T.PILToTensor(), # Convert all PIL objects to tensors
    T.Resize((512, 512)),  # Resize all images to 640x640
])

dataset = datasets.CocoCaptions(root = 'coco_data/content/val2017/val2017',
                                annFile = 'coco_data/content/annotations/annotations/captions_val2017.json',
                                transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Setup MSE Loss
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for (img, txt) in tqdm(dataloader):
        # Take the first set of captions
        txt = txt[0]

        # Convert captions into a list
        txt = [t for t in txt]

        # Forward pass
        optimizer.zero_grad()

        # Pass it through our model and get the logits
        txt_to_img = model(x = txt)
        txt_to_img = torch.stack([normalize_output(t) for t in txt_to_img], dim = 0)

        # Convert both gt to float and scale
        img = img.to(torch.float32) / 255.0

        # Compute loss
        loss   = loss_fn(txt_to_img, img.to(device))

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()

        # Clear cache if necessary
        # print(f"GPU Memory in Use = {torch.cuda.memory_allocated() / 1024**3}")
        torch.cuda.empty_cache()

        # Print current loss for this batch
        # print(f"Loss for this batch: {loss.item()}")

        # Increment number of batches seen
        num_batches += 1

        # Make a checkpoint every 100 batches
        if (num_batches % 100 == 0):
            torch.save(model.state_dict(), "encoder_input.pth")

        # Print the average loss every 10 batches
        if (num_batches % 10 == 0):
            print(f"Loss after {num_batches} batches: {total_loss / num_batches}")

    print(f"Total Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "encoder_input.pth")
