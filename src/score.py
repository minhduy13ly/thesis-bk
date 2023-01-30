import numpy as np
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


parser = argparse.ArgumentParser()
parser.add_argument("--sr_path", type=str,required=False, help="Path to low resolution image")
parser.add_argument("--hr_path", type=str,required=False, help="Path to high resolution image")
opt = parser.parse_args()


  
sr_image = Image.open(opt.sr_path)
hr_image = Image.open(opt.hr_path)

psnr_score = psnr(np.array(hr_image),np.array(sr_image))
ssim_score, diff = ssim(np.array(hr_image),np.array(sr_image), full=True, multichannel=True)

  



