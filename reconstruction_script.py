######################################################
#                     IMPORTS                        #
######################################################

import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython.display import display, display_markdown

#########################################################################
#                              CONSTANTS                                #
#########################################################################

torch.set_grad_enabled(False)
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
titles=["Input", "DALL-E dVAE (f8, 8192)", "VQGAN (f8, 8192)", 
        "VQGAN (f16, 16384)", "VQGAN (f16, 1024)"]

#########################################################################
#                            HELPER FUNCTIONS                           #
#########################################################################

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

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model, embedding_tensor=None, replace_latent=False):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)

  if embedding_tensor is not None and replace_latent:
    z = embedding_tensor
  elif embedding_tensor is not None and not replace_latent:
    z += embedding_tensor
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

def preprocess(img, target_image_size=256):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)

    return img

def stack_reconstructions(input, x0, x1, x2, x3, titles=[]):
  assert input.size == x1.size == x2.size == x3.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (5*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  img.paste(x1, (2*w,0))
  img.paste(x2, (3*w,0))
  img.paste(x3, (4*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img

def reconstruction_pipeline(image_tensor, pretrained_model, embedding_tensor = None, replace_latent = False, size=320):
  
  print(f"input is of size: {image_tensor.shape}")
  x0 = reconstruct_with_vqgan(preprocess_vqgan(image_tensor), pretrained_model, embedding_tensor, replace_latent)
  
  img = stack_reconstructions(custom_to_pil(preprocess_vqgan(image_tensor[0])),
                              custom_to_pil(x0[0]), titles=titles)
  return img

