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
from t2i_model import T2IEmbeddingInputRedux
from t2i_clip import T2ICLIPVQGan
from PIL import Image

# Define helper functions
def normalize_output(x):
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  return x

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Variables
batch_size  = 4
d_proj      = 512
d_embd      = 32
num_chans   = 256
d_crop      = 256
# ddconfig   = {'double_z': False, 'z_channels': 256, 'resolution': 256,
#     'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 1, 2, 4], 'num_res_blocks': 2,
#     'attn_resolutions': [32], 'dropout': 0.1}

# Initialize model
config_path = "logs/vqgan_gumbel_f8/configs/model.yaml"
chkpt_path  = "logs/vqgan_gumbel_f8/checkpoints/last.ckpt"
model       = T2IEmbeddingInputRedux(d_proj, d_embd, num_chans, config_path, chkpt_path, True)
model_path  = "embedding_input.pth"
model.load_state_dict(torch.load(model_path))
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
optimizer = optim.AdamW(model.parameters(), lr=2.5e-4)

# Setup MSE Loss
# loss_fn = nn.CosineEmbeddingLoss()
# mse_loss = nn.CrossEntropyLoss()
# kl_loss = nn.KLDivLoss(reduction="batchmean")
# def loss_fn(x,y):
#     l_mse = mse_loss(x,y)
#     x = x.reshape(batch_size,3,-1)
#     y = y.reshape(batch_size,3,-1)
#     x = F.log_softmax(x,dim=-1)
#     y = F.softmax(y,dim=-1)
#     l_kl = kl_loss(x,y)
#     return l_mse + l_kl

mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")
cosine_loss = nn.CosineEmbeddingLoss()
def loss_fn(z_txt, z_img):
    # # Compute MSE loss between gt image and generated img
    # l_mse = mse_loss(t2i_img,img)
    # # Compute KL Loss Between gt image and generated img
    # t2i_img = t2i_img.reshape(batch_size,3,-1)
    # img = img.reshape(batch_size,3,-1)
    # t2i_img = F.log_softmax(t2i_img,dim=-1)
    # img = F.softmax(img,dim=-1)
    # l_kl = kl_loss(t2i_img,img)
    # Compute Cosine Similarity Loss Between clip_in_clip and cnn_in_clip embeddings
    b, _, = z_txt.shape
    targets = torch.ones(b).to(device)
    # l_cosine_clip = cosine_loss(z_clip_in_clip, z_cnn_in_clip, targets)
    # Compute Cosine Similarity Loss Between clip_in_cnn and cnn_in_cnn embeddings
    # z_clip_in_cnn = z_clip_in_cnn.reshape(batch_size,-1)
    # z_cnn_in_cnn  = z_cnn_in_cnn.reshape(batch_size,-1)
    # l_cosine_cnn = cosine_loss(z_clip_in_cnn,z_cnn_in_cnn,targets)
    tmp_txt = F.log_softmax(z_txt,dim=1)
    tmp_img = F.softmax(z_img,dim=1)
    l_cosine = cosine_loss(z_txt,z_img,targets)
    return l_cosine + kl_loss(tmp_txt,tmp_img)

# Training loop
num_epochs = 1
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

        # Pass it through our model and get the logits
        # txt_to_img = model(x = txt)
        # txt_to_img = torch.stack([normalize_output(t) for t in txt_to_img], dim = 0)
        # t2i_out, img_recon = model(x = txt, y = img)
        # (z_clip_in_clip, z_clip_in_cnn, z_cnn_in_cnn, z_cnn_in_clip), t2i_img = model(txt,img)
        z_txt, z_img = model(txt,img)
        # targets = torch.ones(batch_size).to(device)

        # Reshape outputs
        # z_txt = z_txt.reshape(batch_size,-1)
        # z_img = z_img.reshape(batch_size,-1)

        # Compute loss
        # loss   = loss_fn(z_clip_in_clip, z_clip_in_cnn, z_cnn_in_cnn, z_cnn_in_clip)
        loss = loss_fn(z_txt,z_img)
        # loss = loss_fn(z_txt, z_img, targets)

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
        if (num_batches % 3000 == 0):
            torch.save(model.state_dict(), model_path)

        # Print the average loss every 10 batches
        if (num_batches % 10 == 0):
            print(f"Loss after {num_batches} batches: {total_loss / num_batches}")

        # Break after 500 batches to see progress
        # if (num_batches == 500):
        #     torch.save(model.state_dict(), model_path)
        #     print(f"Loss after {num_batches} batches: {total_loss / num_batches}")
        #     exit(0)

    print(f"Total Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), model_path)
