# Single Image Non-Uniform Blur Kernel Estimation Via Adaptiva Basis Decomposition
# CVPR 2021 Submission 10063

## Script to compute kernels from an image

python compute_kernels.py -i image_path -m model_path

Model can be anonimously downloaded from [here](https://www.dropbox.com/s/410672buqv3a881/ADE_L1_LeakyRelu_epoch200_epoch150_epoch150_epoch200_epoch200.pkl?dl=0)

## Script to deblur an image or a list of images

python image_deblurring.py -b blurry_img_path --reblur_model model_path --denoiser_model '/path/to/drunet_color.pth'  --output_folder results

Denoiser network (drunet color) can be downloaded from [here](https://drive.google.com/file/d/1KDn0ok5Q6dJtAAIBBkiFbHl1ms9kVezz/view?usp=sharing)

Additional options:   
  --blurry_images: may be a singe image path or a .txt with a list of images.
  
  --n_iters: number of iterations in the optimization (default=30)     
  
  --n_den_iters: number of denoiser steps (default=15)     
  
  --lambda_reblur: learning rate used in the optimization (default=1)     
  
  --resize_factor: input image resize factor (default=1)     
  
  --alpha: alpha parameter in Hybrid Steepest Descent algorithm (default = 0.9)    
  
  --ratio_sigma: controls the amount of denoising (default=2)      


## Script to detect blur in an image or a list of images

python detect_blur.py -b blurry_img_path --reblur_model model_path   --output_folder results
