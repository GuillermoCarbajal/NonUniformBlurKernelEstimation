# Non-Uniform Blur Kernel Estimation Via Adaptiva Basis Decomposition
# ICCV 2021 Submission 9229

## Script to compute kernels from an image

python compute_kernels.py -i image_path -m model_path

Model can be anonimously downloaded from [here](https://www.dropbox.com/s/ei4rhu7di8qpgml/TwoHeads.pkl?dl=0)

## Script to deblur an image or a list of images

python image_deblurring.py -b blurry_img_path --reblur_model model_path --output_folder results


Additional options:   
  --blurry_images: may be a singe image path or a .txt with a list of images.
  
  --n_iters: number of iterations in the RL optimization (default=30)       
  
  --resize_factor: input image resize factor (default=1)     
  
  --saturation_method: 'combined' or 'basic'. When 'combined' is passed RL in the presence of saturated pixels is applied. Otherwise,  simple RL update rule is applied in each iteration. For Kohler images, 'basic' is applied. For RealBlur images 'combined' is better.
  
  --gamma_factor: gamma correction factor. By default is assummed gamma_factor=2.2. For Kohler dataset images gamma_factor=1.0.

