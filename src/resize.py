
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

root_dir = "/content/gdrive/MyDrive/Data/Testing/Original/"


parser = argparse.ArgumentParser()
parser.add_argument("--lr_path", type=str, required=False, help="Path to low resolution image")
parser.add_argument("--hr_path", type=str, required=False, help="Path to high resolution image")
parser.add_argument("--folder_name", type=str, required=True, help="Folder name")
opt = parser.parse_args()
print(opt)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Prepare input
hr_input = torchvision.datasets.ImageFolder(root= (root_dir + opt.hr_path))
hr_resize = "/content/gdrive/MyDrive/Data/Testing/Cut/" + opt.folder_name
os.makedirs(hr_resize, exist_ok=True)

lr_list = []
hr_list = []
name = []
for i in range(len(hr_input)):
  hr_name = hr_input.imgs[i][0]
  hr_name = hr_name.split('/')[-1]
  hr_name = hr_name.split('.')[0]
  name.append(hr_name)

def img_resize(img_input, folder_resize, result,crop_size):
  for i in range(0,len(img_input)):
    print("Resizing image " + str(i + 1))
    image = img_input[i][0]
    print(image.size)
    crop = transforms.Compose([transforms.RandomCrop((crop_size,crop_size))])
    crop_image = crop(image)
    result.append(crop_image)

  for i in range(0,len(img_input)):
      print("After resizing image " + str(i + 1))
      resize_img = result[i]  
      print(resize_img.size)
      resize_img.save(folder_resize + "/" + name[i] + '.png','png')

img_resize(hr_input,hr_resize,hr_list, 128)
