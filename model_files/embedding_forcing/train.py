"""
Text 2 Image Training for Embedding Forcing
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
from embedding_forcing_t2i import T2IEmbeddingForcing
from PIL import Image

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Variables
batch_size  = 32
d_proj      = 512
d_embd      = 32
num_chans   = 256
d_crop      = 256
lr          = 2e-4

# Initialize model
config_path = "logs/vqgan_gumbel_f8/configs/model.yaml"
chkpt_path  = "logs/vqgan_gumbel_f8/checkpoints/last.ckpt"
model       = T2IEmbeddingForcing(d_proj, d_embd, num_chans, config_path, chkpt_path, True)
model_path  = "embedding_forcing.pth"
# model.load_state_dict(torch.load(model_path))
model.to(device)

# Initialize dataloader
transform = T.Compose([
    T.PILToTensor(), # Convert all PIL objects to tensors
    T.Resize((d_crop, d_crop)),  # Resize all images to d_crop x d_crop
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
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Setup loss function
"""
Source: https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/clip/modeling_clip.py#L65
"""
def contrastive_loss(logits):
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def loss_fn(x,y):
    logit_scale = model.logit_scale.exp().to(device)
    txt_logits = logit_scale * torch.matmul(x, y.t())
    img_logits = txt_logits.t()
    l_contrastive = (contrastive_loss(txt_logits) + contrastive_loss(img_logits)) / 2.0
    return l_contrastive

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for (img, txt) in tqdm(dataloader):
        # Convert caption tuple options into a list
        txt = [t[0] for t in txt]
        img = img.to(device)

        # Forward pass
        optimizer.zero_grad()

        # Pass through model
        z_txt, z_img = model(txt,img)

        # Compute loss
        loss = loss_fn(z_txt,z_img)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()

        # Clear cache if necessary
        torch.cuda.empty_cache()
        # Delete loss item to save on space
        del loss

        # Increment number of batches seen
        num_batches += 1

        # Make a checkpoint every 3000 batches
        if (num_batches % 3000 == 0):
            torch.save(model.state_dict(), model_path)

        # Print the average loss every 100 batches
        if (num_batches % 100 == 0):
            print(f"Loss after {num_batches} batches: {total_loss / num_batches}")

    print(f"Total Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), model_path)
