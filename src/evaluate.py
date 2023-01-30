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

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from brisque import BRISQUE
import lpips



lr_default = "/content/gdrive/MyDrive/Data/lr"
hr_default = "/content/hr"


parser = argparse.ArgumentParser()
parser.add_argument("--lr_path", type=str, default = lr_default,required=False, help="Path to low resolution image")
parser.add_argument("--hr_path", type=str, default = hr_default,required=False, help="Path to high resolution image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)


def denormalize(tensors):
    """ Denormalizes image tensors for lpips """
    return (tensors + 1)/2.0


def get_average(lst):
  return sum(lst)/len(lst)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor()])


# Prepare input
lr_input = torchvision.datasets.ImageFolder(root= opt.lr_path)
hr_input = torchvision.datasets.ImageFolder(root= opt.hr_path)

output = "all/output"
os.makedirs(output, exist_ok=True)
sr_list = []
hr_list = []
name = []
print("Using model at :" + opt.checkpoint_model)
for i in range(0,len(lr_input)):
  print("Generating HR image " + str(i + 1))
  lr_image = lr_input[i][0]

  #Name
  lr_name = lr_input.imgs[i][0]
  lr_name = lr_name.split('/')[-1]
  lr_name = lr_name.split('.')[0]
  name.append(lr_name)
  lr_tensor = Variable(transform(lr_image)).to(device).unsqueeze(0)
  
  lr_tensor = 2 * lr_tensor - 1



  with torch.no_grad():
    sr_tensor = denormalize(generator(lr_tensor)).cpu()
    sr_list.append(sr_tensor)
  
  

  save_image(sr_tensor, output + "/" + name[i] + ".png")

  hr = hr_input[i][0]
  hr_cuda = Variable(transform(hr)).to(device).unsqueeze(0)
  hr_tensor = hr_cuda.to(device=torch.device("cpu"))
  hr_tensor = 2 * hr_tensor - 1 
  hr_list.append(hr_tensor)
  




psnr_list = []
ssim_list = []
brisque_list = []
lpips_list = []

sr_input = torchvision.datasets.ImageFolder(root= "all")


for i in range(0,len(lr_input)):
  print("Image " + str(i + 1))
  sr_image = sr_input[i][0]
  sr_tensor = sr_list[i]

  hr_image = hr_input[i][0]

  hr_tensor = hr_list[i]

  psnr_score = psnr(np.array(hr_image),np.array(sr_image))
  psnr_list.append(psnr_score)

  ssim_score, diff = ssim(np.array(hr_image),np.array(sr_image), full=True, multichannel=True)
  
  ssim_list.append(ssim_score)

  bris = BRISQUE()
  bris_score = bris.get_score(output + "/" + name[i] + ".png")
  brisque_list.append(bris_score)

  loss_fn_alex = lpips.LPIPS(net='alex')
  sr_tensor = 2 * sr_tensor - 1
  lpips_score = loss_fn_alex(sr_tensor, hr_tensor)
  lpips_list.append(lpips_score)

  




psnr_average = get_average(psnr_list)
ssim_average = get_average(ssim_list)

brisque_average = get_average(brisque_list)
lpips_average = get_average(lpips_list)



np.savez_compressed('/content/gdrive/MyDrive/all_metric.npz', PSNR = psnr_list, SSIM =ssim_list, 
BRISQUE = brisque_list, LPIPS = lpips_list, aver = list([psnr_average,ssim_average,brisque_average,lpips_average]))