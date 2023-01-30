from models import GeneratorRRDB
#from datasets_vgg import denormalize, mean, std
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F

import numpy as np 
from numpy import savez_compressed

import matplotlib.pyplot as plt

from torch.autograd import Variable
import argparse
import os

from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image



lr_default = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/resize_lr"
hr_default = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/resize_hr"
checkpoint_dir = "/content/gdrive/MyDrive/THESIS RESOURCES/TESTING/GENERATOR"
parser = argparse.ArgumentParser()
parser.add_argument("--lr_path", type=str, default = lr_default,required=False, help="Path to low resolution image")
parser.add_argument("--hr_path", type=str, default = hr_default,required=False, help="Path to high resolution image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
parser.add_argument("--model_name", type=str, default="original", help="Model name")
opt = parser.parse_args()
print(opt)


def denormalize(tensors):
    """ Denormalizes image tensors for lpips """
    return (tensors + 1)/2.0

def get_image_path(i):
  if(i < 9):
    fn = "/" + str(opt.model_name) + "-080" + str(i+1) + str(".png")
  elif(i >= 9 and i < 99):
    fn = "/" + str(opt.model_name) + "-08" + str(i + 1) + str(".png")
  else:
    fn = "/" + str(opt.model_name) + "-0900.png"
  return  fn

def get_average(lst):
  return sum(lst)/len(lst)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(checkpoint_dir + opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor()])


# Prepare input
lr_input = torchvision.datasets.ImageFolder(root= opt.lr_path)
hr_input = torchvision.datasets.ImageFolder(root= opt.hr_path)

output = "gen_hr/Model/" + str(opt.model_name) 
os.makedirs(output, exist_ok=True)
sr_list = []
hr_list = []

print("Using model at :" + opt.checkpoint_model)
for i in range(0,len(lr_input)):
  print("Generating HR image " + str(i + 1))
  lr_image = lr_input[i][0]
  lr_tensor = Variable(transform(lr_image)).to(device).unsqueeze(0)
  
  lr_tensor = 2 * lr_tensor - 1
  #Unfold
  kc, kh, kw = 3, 128, 128
  dc, dh, dw = 3, 128, 128

  patches = lr_tensor.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
  patch = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)


  sr_patch = []
  #Get the patch of LR image
  for j in range(0, patch.shape[1]):
    patch_j = patch[:,j,:,:,:]
    with torch.no_grad():
      sr_tensor_patch = denormalize(generator(patch_j)).cpu()
      sr_patch.append(sr_tensor_patch)
 
  #Get the new tensor from the patch
  sr_tensor = torch.stack(sr_patch,dim = 1)
  sr_tensor = torch.flatten(sr_tensor,start_dim = 2)
  sr_tensor = sr_tensor.permute(0,2,1)
  

  # Fold back
  fold = torch.nn.Fold(output_size = (lr_tensor.shape[-2] * 4,lr_tensor.shape[-1] * 4), kernel_size = (512 ,512), stride = (512,512))
  sr_tensor = fold(sr_tensor)
  
  sr_list.append(sr_tensor)

  
  save_image(sr_tensor, output + get_image_path(i))

  hr = hr_input[i][0]
  hr_cuda = Variable(transform(hr)).to(device).unsqueeze(0)
  hr_tensor = hr_cuda.to(device=torch.device("cpu"))
  hr_tensor = 2 * hr_tensor - 1 
  hr_list.append(hr_tensor)
  




