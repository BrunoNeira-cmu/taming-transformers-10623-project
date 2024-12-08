"""
Text 2 Image with VQGAN Code
Author: Nicholas Mesa-Cucalon
10-623 Generative AI
"""

# Perform Imports
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from t2i_model import T2IEncoderInput

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


# Initialize dataloader
dataset = datasets.CocoCaptions(root = '/data/coco_annotations_100/val2017',
                                annFile = '/data/annotations/stuff_val2017.json',
                                transform=T.PILToTensor())

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Setup L2 Distance Norm Function
loss_fn = lambda x,y: (torch.linalg.norm(x) - torch.linalg.norm(y)) ** 2

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for (txt, img) in tqdm(dataloader):
        """
        DATA SETUP HERE
        """
        print(txt)
        print(img)
        exit(0)

        # Forward pass
        optimizer.zero_grad()

        # Pass it through our model and get the logits
        txt_to_img = model(x = txt)

        """
        TODO: Compute L2 Loss Fn
        """
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
