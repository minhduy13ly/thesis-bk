
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

import radialProfile
import glob
import cv2
from scipy.interpolate import griddata

parser = argparse.ArgumentParser()
opt = parser.parse_args()
print(opt)

## real data
N = 128
epsilon = 1e-8
number_iter = 1000
psd1D_total = np.zeros([number_iter, N])
y = []
error = []

def RGB2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# real
psd1D_org_mean = np.zeros(N)
psd1D_org_std = np.zeros(N)

cont = 0

rootdir = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/gen_hr/Visualize/Real/"

#rootdir = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/resize_hr/hr_resize/"
#-----------------
# ----Original------
#-----------------
for filename in glob.glob(rootdir+"*.png"):
    print(filename)    
    img = cv2.imread(filename,0)
  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    
     # Calculate the azimuthally averaged 1D power spectrum
    points = np.linspace(0,N,num=psd1D.size) # coordinates of a
    xi = np.linspace(0,N,num=N) # coordinates for interpolation
    interpolated = griddata(points,psd1D,xi,method='cubic')
    
    interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
    psd1D_total[cont,:] = interpolated  
    
    cont+=1
    
    if cont == number_iter:
        break


for x in range(N):
    psd1D_org_mean[x] = np.mean(psd1D_total[:,x])
    psd1D_org_std[x] = np.std(psd1D_total[:,x])


#-----------------
# ----RaGAN------
#-----------------
psd1D_org_mean2= np.zeros(N)
psd1D_org_std2= np.zeros(N)
cont = 0

print("Finish original")

firstdir = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/gen_hr/Visualize/No FFT/"

for filename in glob.glob(firstdir+"*.png"):
    print(filename)
    img = cv2.imread(filename,0)
  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    
     # Calculate the azimuthally averaged 1D power spectrum
    points = np.linspace(0,N,num=psd1D.size) # coordinates of a
    xi = np.linspace(0,N,num=N) # coordinates for interpolation
    interpolated = griddata(points,psd1D,xi,method='cubic')
    
    interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
    psd1D_total[cont,:] = interpolated  
    
    cont+=1
    
    if cont == number_iter:
        break


for x in range(N):
    psd1D_org_mean2[x] = np.mean(psd1D_total[:,x])
    psd1D_org_std2[x] = np.std(psd1D_total[:,x])


print("Finish FFT")
#------------------------------------------------
psd1D_org_mean3= np.zeros(N)
psd1D_org_std3= np.zeros(N)
cont = 0

#seconddir = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/gen_hr/Datasets/Urban100/RaGP/"
seconddir = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/gen_hr/Visualize/FFT/"
for filename in glob.glob(seconddir+"*.png"):
    print(filename)
    img = cv2.imread(filename,0)
  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    
     # Calculate the azimuthally averaged 1D power spectrum
    points = np.linspace(0,N,num=psd1D.size) # coordinates of a
    xi = np.linspace(0,N,num=N) # coordinates for interpolation
    interpolated = griddata(points,psd1D,xi,method='cubic')
    
    interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
    psd1D_total[cont,:] = interpolated  
    
    cont+=1
    
    if cont == number_iter:
        break


for x in range(N):
    psd1D_org_mean3[x] = np.mean(psd1D_total[:,x])
    psd1D_org_std3[x] = np.std(psd1D_total[:,x])

print("Finish FFT")

#------------------------------------------------
psd1D_org_mean4= np.zeros(N)
psd1D_org_std4= np.zeros(N)
cont = 0

seconddir = "/content/gdrive/MyDrive/THESIS RESOURCES/ESRGAN code/gen_hr/Visualize/Spectral/"
for filename in glob.glob(seconddir+"*.png"):
    print(filename)
    img = cv2.imread(filename,0)
  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    
     # Calculate the azimuthally averaged 1D power spectrum
    points = np.linspace(0,N,num=psd1D.size) # coordinates of a
    xi = np.linspace(0,N,num=N) # coordinates for interpolation
    interpolated = griddata(points,psd1D,xi,method='cubic')
    
    interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
    psd1D_total[cont,:] = interpolated  
    
    cont+=1
    
    if cont == number_iter:
        break


for x in range(N):
    psd1D_org_mean4[x] = np.mean(psd1D_total[:,x])
    psd1D_org_std4[x] = np.std(psd1D_total[:,x])


print("Finish spectral")

# #----------------------
y.append(psd1D_org_mean)
y.append(psd1D_org_mean2)
y.append(psd1D_org_mean3)
y.append(psd1D_org_mean4)




x = np.arange(0, N, 1)

fig, ax = plt.subplots(figsize=(15, 9))

ax.plot(x, y[0], alpha=0.5, color='red', label='A: Real', linewidth =2.0)


ax.plot(x, y[1], alpha=0.5, color='blue', label='B: Model train without FFT/spectral loss', linewidth = 2.0)
print("No FFT difference")
print(np.mean(np.abs(y[0]-y[1])))

ax.plot(x, y[2], alpha=0.5, color='black', label='C: Model train with FFT loss', linewidth = 2.0)

print("FFT difference")
print(np.mean(np.abs(y[0]-y[2])))

ax.plot(x, y[3], alpha=0.5, color='green', label='D: Model train with spectral loss', linewidth = 2.0)

print("spectral difference")
print(np.mean(np.abs(y[0]-y[3])))

# ax.plot(x, y[4], alpha=0.5, color='black', label='Config D', linewidth = 2.0)
# print("FFT difference")
# print(np.mean(np.abs(y[0]-y[4])))

#ax.fill_between(x, y[3] - error[3], y[3] + error[3], color='black', alpha=0.2)


plt.xlabel('Spatial Frequency', fontsize=25)
plt.ylabel('Power Spectrum', fontsize=25)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)


ax.legend(loc='best', prop={'size': 25})
fig.savefig("test.jpg")
