"""
Text 2 Image via Encoder Output with Contrastive Loss
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
Module: Adapter with Residual
Source: https://github.com/SiddhantBikram/MemeCLIP
"""
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x) + x
        return x

"""
Model 4: Contrastive Loss Embedding Input
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class T2IEmbeddingForcing(nn.Module):
    def __init__(self, d_proj : int,
                       d_embd : int,
                       num_chans : int,
                       config_path : str,
                       ckpt_path : str,
                       is_gumbel : bool,
    ):
        super(T2IEmbeddingForcing, self).__init__()
        # Store variables
        self.d_proj = d_proj
        self.d_embd = d_embd
        self.num_channels = num_chans
        # CLIP Text Model
        self.clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize grounding layers and adapters
        self.adapter  = Adapter(d_proj,4)
        self.clip2cnn = nn.Linear(d_proj, num_chans * (d_embd ** 2))
        # Initialize VQGAN Config
        config = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
        self.vqgan = load_vqgan(config, ckpt_path=ckpt_path, is_gumbel=is_gumbel).to(device)
        # Initialize logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # Freeze CLIP Modules
        for param in self.clip_txt.parameters():
            param.requires_grad = False
        # Freeze VQGAN modules
        for param in self.vqgan.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        # Tokenize text
        inputs = self.clip_tokenizer(x, padding="longest", max_length = 77, truncation = True, return_tensors = "pt").to(device)
        # Return text features
        return self.clip_txt(**inputs).text_embeds

    def forward(self, x, y):
        # Length of x array is the batch size
        b = len(x)
        # Encode text in CLIP and CNN Encoder Space
        z_txt = self.clip2cnn(self.adapter(self.encode_text(x)))
        z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
        # Encode image in CLIP and CNN Encoder Space
        z_img = self.vqgan.encoder(y).reshape(b,-1)
        z_img = z_img / z_img.norm(dim=-1, keepdim=True)
        # Return both embeddings
        return z_txt, z_img

    def generate(self, x):
        # Length of x array is the batch size
        b = len(x)
        # Encode text
        z = self.encode_text(x)
        # Project into the embedding space
        cnn_shape = (b,self.num_channels,self.d_embd,self.d_embd)
        z = self.clip2cnn(z).reshape(cnn_shape)
        # Decode with the VQGAN decoder
        return self.vqgan.decode(z)
