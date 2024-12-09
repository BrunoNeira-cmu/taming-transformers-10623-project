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
batch_size  = 256
d_proj      = 512
d_hidden    = 32

# Initialize model
config_path = "logs/vqgan_gumbel_f8/configs/model.yaml"
chkpt_path = "logs/vqgan_gumbel_f8/checkpoints/last.ckpt"
model = T2IEncoderInput(d_proj, d_hidden, config_path, chkpt_path, True)
model_path = "embedding_input.pth"
# model.load_state_dict(torch.load(model_path))
model.to(device)


# Initialize dataloader
transform = T.Compose([
    T.PILToTensor(), # Convert all PIL objects to tensors
    T.Resize((d_hidden, d_hidden)),  # Resize all images to d_h x d_h
    T.ConvertImageDtype(torch.float),
])

# Define a custom collate_fn
def custom_collate_fn(batch):
    images, captions = zip(*batch)

    # Find the max height and width in the batch
    max_height = max(img.shape[1] for img in images)
    max_width  = max(img.shape[2] for img in images)

    # Pad images to the max dimensions
    padded_images = [
        F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]))
        for img in images
    ]

    # Stack images and return with captions
    return torch.stack(padded_images), captions

dataset = datasets.CocoCaptions(root = 'coco_data/content/train2017/train2017',
                                annFile = 'coco_data/content/annotations/annotations/captions_train2017.json',
                                transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = custom_collate_fn)

# Setup optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Setup MSE Loss
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for (img, txt) in tqdm(dataloader):
        # Convert caption tuple options into a list
        txt = [t[0] for t in txt]

        # Forward pass
        optimizer.zero_grad()

        # Pass it through our model and get the logits
        txt_to_img = model(x = txt)
        txt_to_img = torch.stack([normalize_output(t) for t in txt_to_img], dim = 0)

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
        del loss

        # Print current loss for this batch
        # print(f"Loss for this batch: {loss.item()}")

        # Increment number of batches seen
        num_batches += 1

        # Make a checkpoint every 100 batches
        if (num_batches % 100 == 0):
            torch.save(model.state_dict(), model_path)

        # Print the average loss every 10 batches
        if (num_batches % 10 == 0):
            print(f"Loss after {num_batches} batches: {total_loss / num_batches}")

    print(f"Total Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), model_path)
