"""
Text 2 Image via Multimodal Fusion
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
from transformers import CLIPModel, AutoProcessor

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
Model 2: Fusion Encoder Input
"""
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class T2IFusionEncoderInput(nn.Module):
    def __init__(self, d_proj : int,
                       d_embd : int,
                       num_chans : int,
                       config_path : str,
                       ckpt_path : str,
                       is_gumbel : bool,
    ):
        super(T2IFusionEncoderInput, self).__init__()
        # Store variables
        self.d_proj = d_proj
        self.d_embd = d_embd
        self.num_channels = num_chans
        # CLIP Model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize Linear Layer
        self.grounding_layer = nn.Linear(d_proj, self.num_channels * (self.d_embd ** 2))
        # Initialize Fusion Function
        self.fuse_fn = lambda x,y : x + y
        # Initialize VQGAN Config
        config = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
        self.vqgan = load_vqgan(config, ckpt_path=ckpt_path, is_gumbel=is_gumbel).to(device)
        # Freeze CLIP Modules
        for param in self.clip.parameters():
            param.requires_grad = False
        # Freeze VQGAN modules
        for param in self.vqgan.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Batch size is the length of the text list
        b = len(x)
        # Process text and image into features
        txt_in = self.clip_processor.tokenizer(
            text=x, return_tensors="pt", padding=True, truncation = True,
        ).to(device)
        img_in = self.clip_processor.image_processor(
            images=y, do_rescale = False, return_tensors="pt",
        ).to(device)
        txt_features = self.clip.get_text_features(**txt_in)
        img_features = self.clip.get_image_features(**img_in)
        # Fuse features
        fused_features = self.fuse_fn(txt_features,img_features)
        # Project it into the encoder input dimension
        embd = self.grounding_layer(fused_features)
        # Create a [b, 3, d_embd, d_embd] version of our grounded text features
        embd = embd.reshape((b,self.num_channels,self.d_embd,self.d_embd))
        # Preprocess embedding for the VQGAN Encoder
        embd = preprocess_vqgan(embd)
        # Encode embedding with VQGAN Encoder
        z, _, _ = self.vqgan.encode(embd)
        # Return a reconstructed version of the encoded embedding
        return self.vqgan.decode(z)

    def encode_text(self, x):
        # Tokenize text
        inputs = self.clip_processor.tokenizer(x, padding = "max_length",
            max_length = 77, truncation = True, return_tensors = "pt").to(device)
        # Get text features
        txt_features = self.clip.get_text_features(**inputs)
        # Get the batch size
        b, _ = txt_features.shape
        # Project it into the encoder input dimension
        text_embd = self.grounding_layer(txt_features)
        # Return a [b, 3, d_embd, d_embd] view of our grounded text features
        return text_embd.view((b,self.num_channels,self.d_embd,self.d_embd))

    def generate(self,x):
        # Encode text
        x = self.encode_text(x)
        # Preprocess encoded text for the VQGAN
        x = preprocess_vqgan(x)
        # Encode text with VQGAN Encoder
        z, _, _ = self.vqgan.encode(x)
        # Return a reconstructed version of the encoded text embedding
        return self.vqgan.decode(z)
