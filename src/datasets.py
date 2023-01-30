import glob
import os
import numpy as np
from random import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



def denormalize(tensors):
    """ Denormalizes image tensors for lpips """
    return (tensors + 1)/2.0

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Crop image first
        
        self.crop = transforms.Compose(
          [
            transforms.RandomCrop(size = (128,128))
          ]
        )

        # Transforms for low resolution images and high resolution images

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                #transforms.Resize((hr_height // 8, hr_width // 8), Image.BICUBIC),
                transforms.ToTensor()
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor()
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_crop = self.crop(img)
        img_lr = self.lr_transform(img_crop)
        img_hr = self.hr_transform(img_crop)

        img_lr = 2 * img_lr - 1
        img_hr = 2 * img_hr - 1 


        

        if random() > 0.5:
            img_hr = transforms.functional.vflip(img_hr)
            img_lr = transforms.functional.vflip(img_lr)

        if random() > 0.5:
            img_hr = transforms.functional.hflip(img_hr)
            img_lr = transforms.functional.hflip(img_lr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)