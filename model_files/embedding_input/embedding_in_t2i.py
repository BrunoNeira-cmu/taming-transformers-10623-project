"""
Text 2 Image via Encoder Output Space Grounding
Authors: Nicholas Mesa-Cucalon, Bruno Neira, Deon D Kouatchou-Ngongang
10-623 Generative AI
"""

"""
Imports
"""
import sys
import os
os.chdir("/home/ubuntu/taming-transformers-10623-project")
sys.path.append(os.getcwd())
import yaml
from taming.models.vqgan import VQModel, GumbelVQ
import numpy as np
import torch
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import CLIPTextModelWithProjection, AutoTokenizer

"""
Helper Functions
"""
def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

"""
Model 3: Embedding Input
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class T2IEmbeddingInput(nn.Module):
    def __init__(self, d_proj : int,
                       d_embd : int,
                       num_chans : int,
                       config_path : str,
                       ckpt_path : str,
                       is_gumbel : bool,
    ):
        super(T2IEmbeddingInput, self).__init__()
        # Store variables
        self.d_proj = d_proj
        self.d_embd = d_embd
        self.num_channels = num_chans
        # CLIP Text Model
        self.clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize grounding layers
        self.grounding_layer = nn.Linear(d_proj, num_chans * (d_embd ** 2))
        # Initialize VQGAN Config
        config = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
        self.vqgan = load_vqgan(config, ckpt_path=ckpt_path, is_gumbel=is_gumbel).to(device)
        # Freeze CLIP Modules
        for param in self.clip_txt.parameters():
            param.requires_grad = False
        # Freeze VQGAN modules
        for param in self.vqgan.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        # Tokenize text
        inputs = self.clip_tokenizer(x, padding = "max_length",
            max_length = 77, truncation = True, return_tensors = "pt").to(device)
        # Return text features
        txt_features = self.clip_txt(**inputs).text_embeds
        return txt_features

    def forward(self, x, y):
        # Length of x array is the batch size
        b = len(x)
        # Encode text and project to CNN Encoder Space
        z_txt = self.encode_text(x)
        z_txt = self.grounding_layer(z_txt).reshape(b,-1)
        # Encode image in CNN Encoder Space
        z_img = self.vqgan.encoder(y).reshape(b,-1)
        # Return both embeddings
        return z_txt, z_img

    def generate(self, x):
        # Length of x array is the batch size
        b = len(x)
        # Encode text
        z = self.encode_text(x)
        # Project into the embedding space
        cnn_shape = (b,self.num_channels,self.d_embd,self.d_embd)
        z = self.grounding_layer(z).reshape(cnn_shape)
        return self.vqgan.decode(z)
