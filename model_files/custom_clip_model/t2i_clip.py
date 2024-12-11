"""
Text 2 Image with CLIP and VQGAN Arch
Authors: Nicholas Mesa-Cucalon, Bruno Neira, Deon D Kouatchou-Ngongang
10-623 Generative AI
"""

# This model didn't originally fit on a GPU, so we abandoned it, though we considered training from
# scratch. Future Work maybe?

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
from transformers import CLIPModel, AutoProcessor

"""
Helper Functions
"""
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
Model 1: CLIP Model
"""
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class T2ICLIPVQGan(nn.Module):
    def __init__(self, d_proj : int,
                       d_embd : int,
                       num_channels : int,
                       config_path : str,
                       ckpt_path : str,
                       is_gumbel : bool
    ):
        super(T2ICLIPVQGan, self).__init__()
        # Needed variables
        self.num_channels = num_channels
        self.d_embd = d_embd
        # VQGAN
        config = load_config(config_path, display=False)
        self.vqgan = load_vqgan(config, ckpt_path=ckpt_path, is_gumbel=is_gumbel).to(device)
        # CLIP Model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Grounding Layer
        self.grounding_layer = nn.Linear(d_proj,num_channels * (d_embd ** 2))
        # Freeze CLIP and VQGAN
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.vqgan.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Encode text
        x = self.clip_processor.tokenizer(x, padding = "max_length",
            max_length = 77, truncation = True, return_tensors = "pt").to(device)
        x = self.clip.get_text_features(**x)
        # Encode img
        y = self.clip_processor(images = y, do_rescale = False, return_tensors = "pt").to(device)
        y = self.clip.get_image_features(**y)
        # Ground both txt and img features into quant space
        x, y = self.grounding_layer(x), self.grounding_layer(y)
        # Get batch size
        b, _ = x.shape
        # Reshape to be an image like feature
        x = x.reshape(b, self.num_channels, self.d_embd, self.d_embd)
        y = y.reshape(b, self.num_channels, self.d_embd, self.d_embd)
        # Quantize encoded txt
        x = self.vqgan.quant_conv(x)
        x, _, _ = self.vqgan.quantize(x)
        # Quantize encoded img
        y = self.vqgan.quant_conv(y)
        y, _, _ = self.vqgan.quantize(y)
        # Return the decoded text and img
        return self.decoder(x), self.decoder(y)
