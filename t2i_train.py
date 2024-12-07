"""
Text 2 Image with VQGAN Code
Author: Nicholas Mesa-Cucalon
10-623 Generative AI
"""

# Perform Imports
import sys
sys.path.append(".")
import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython.display import display, display_markdown

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Variables
batch_size  = 8
d_proj      = 512
d_hidden    = 512

# Initialize model
config_path = "logs/vqgan_gumbel_f8/configs/model.yaml"
chkpt_path = "logs/vqgan_gumbel_f8/checkpoints/last.ckpt"
model = T2IEncoderInput(d_proj, d_hidden, config_path, chkpt_path, True)
model.to(device)

"""
TODO: FIX THIS DATALOADER TO WORK AS WE WANT
"""
dataset = MemeCapDataset(json_path="../data/memes-trainval-filtered.json",
                         image_path="../data/images/images-trainval-filtered/",
                         ocr_path="./memes-trainval-ocr.json")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Setup Loss Function
loss_fn = torch.nn.MSE()

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for (images, texts) in tqdm(enumerate(dataloader)):
        """
        DATA SETUP HERE
        """
        # Forward pass
        optimizer.zero_grad()

        # Pass it through our model and get the logits
        _, logits = model(x = texts, y = images, prompt = prompts, num_logits = 0)

        # Calculate loss (assuming LLaMA is generating tokens and using CrossEntropyLoss)
        tokens = tokens[:,1:].contiguous()    # Model does not predict the first token, so we remove it
        logits = logits[:,:-1,:].contiguous() # No label for the token that follows the full input seq, so remove it

        logits = logits.view(-1, logits.shape[-1]) # Shape -> [b * (seq_len), d_vocab]
        tokens = tokens.view(-1) # Shape -> [b * seq_len]

        # Create padding masks so we don't learn to the padding tokens
        mask = (tokens != pad_token_id)
        logits, tokens = logits[mask], tokens[mask]

        # Compute cross entropy loss
        loss   = loss_fn(logits, tokens.to(device))

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()

        # Clear cache if necessary
        print(f"GPU Memory in Use = {torch.cuda.memory_allocated() / 1024**3}")
        torch.cuda.empty_cache()

        # Print urrent loss for this batch
        print(f"Loss for this batch: {loss.item()}")

        # Increment number of batches seen
        num_batches += 1

        # Make a checkpoint every 100 batches
        if (num_batches % 100 == 0):
            torch.save(model.state_dict(), "multimodal_model.pth")

        # Print the average loss every 10 batches
        if (num_batches % 10 == 0):
            print(f"Loss after {num_batches} batches: {total_loss / num_batches}")

        # Break early to see performance after 150 batches
        if (num_batches == 150):
            torch.save(model.state_dict(), "multimodal_model.pth")
            print(f"Total Loss: {total_loss/50}")
            exit(0)

    print(f"Total Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "multimodal_model.pth")
