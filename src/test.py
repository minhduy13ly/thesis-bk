from models import GeneratorRRDB
#from datasets_vgg import denormalize, mean, std
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F

import numpy as np 
from numpy import savez_compressed

import cv2

import matplotlib.pyplot as plt

from torch.autograd import Variable
import argparse
import os

from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)



original = "test/bounding_box/"
output = "test/sr_image/fft-large"
bicubic = "test/bicubic"

#---------------------------------------------------------------------------


os.makedirs(output, exist_ok=True)
os.makedirs(bicubic, exist_ok=True)
def denormalize(tensors):
    """ Denormalizes image tensors for lpips """
    return (tensors + 1)/2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model),strict=False)
generator.eval()

transform = transforms.Compose([transforms.ToTensor()])

# Prepare input
image = Image.open(opt.image_path)
image_tensor = Variable(transform(image)).to(device).unsqueeze(0)

image_tensor = 2 * image_tensor - 1


# Upsample image
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor)).cpu()



bicubic_image = torch.nn.functional.interpolate(image_tensor,scale_factor = 4, mode='bicubic')
bicubic_image = denormalize(bicubic_image)
# Save image
fn = opt.image_path.split("/")[-1]
save_image(sr_image, output + "/" + fn)
save_image(bicubic_image,bicubic + "/" + fn)