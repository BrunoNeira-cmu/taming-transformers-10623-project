"""
Text 2 Image with VQGAN Model
Author: Nicholas Mesa-Cucalon
10-623 Generative AI
"""

"""
Imports
"""
import sys
sys.path.append(".")
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
Model 1: Encoder Input
"""
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class T2IEncoderInput(nn.Module):
    def __init__(self, d_proj : int,
                       res_h_dim : int,
                       config_path : str,
                       ckpt_path : str,
                       is_gumbel : bool,
    ):
        super(T2IEncoderInput, self).__init__()
        # Store variables
        self.d_proj = d_proj
        self.res_h_dim = res_h_dim
        self.num_channels = 3
        # CLIP Text Model
        self.clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize Linear Layer
        self.grounding_layer = nn.Linear(d_proj, self.num_channels * (self.res_h_dim ** 2))
        # self.grounding_stack = nn.ModuleList([
        #             nn.Linear(d_proj, res_h_dim ** 2),
        #             nn.Linear(d_proj, res_h_dim ** 2),
        #             nn.Linear(d_proj, res_h_dim ** 2)
        #         ])
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
        # Get text features
        txt_features = self.clip_txt(**inputs).text_embeds
        # Get the batch size
        b, _ = txt_features.shape
        # Project it into the encoder input dimension
        # text_embd = torch.stack([layer(txt_features) for layer in self.grounding_stack],dim = 1)
        text_embd = self.grounding_layer(txt_features)
        # Return a [b, 3, res_h_dim, res_h_dim] view of our grounded text features
        return text_embd.view((b,self.num_channels,self.res_h_dim,self.res_h_dim))

    def forward(self, x):
        # Encode text
        x = self.encode_text(x)
        # Preprocess encoded text for the VQGAN
        x = preprocess_vqgan(x)
        # Encode text with VQGAN Encoder
        z, _, _ = self.vqgan.encode(x)
        # Return a reconstructed version of the encoded text embedding
        return self.vqgan.decode(z)

"""
Model 2: Embedding Input
"""
class T2IEmbeddingInput(nn.Module):
    def __init__(self, d_proj : int,
                       res_h_dim : int,
                       config_path : str,
                       ckpt_path : str,
                       is_gumbel : bool,
    ):
        super(T2IEmbeddingInput, self).__init__()
        # Store variables
        self.d_proj = d_proj
        self.res_h_dim = res_h_dim
        self.num_channels = 256
        # CLIP Text Model
        self.clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize grounding layer
        self.grounding_layer = nn.Linear(d_proj, self.num_channels * (self.res_h_dim ** 2))
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
        # Get text features
        txt_features = self.clip_txt(**inputs).text_embeds
        # Get the batch size
        b, _ = txt_features.shape
        # Project it into the encoder input dimension
        text_embd = self.grounding_layer(txt_features)
        # Return a [b, 256, 32, 32] view of our grounded text features
        return text_embd.view((b,self.num_channels,self.res_h_dim,self.res_h_dim))

    def forward(self, x):
        # Encode text and project into embedding space
        z = self.encode_text(x)
        # Return a reconstructed version of the encoded text embedding
        return self.vqgan.decode(z)
