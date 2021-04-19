import argparse
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte

from models.TwoHeadsNetwork import TwoHeadsNetwork

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from utils.visualization import save_kernels_grid


parser = argparse.ArgumentParser()
parser.add_argument('--img','-i', type=str, help='path of blurry image', required=True)
parser.add_argument('--model','-m', type=str, help='path of the model', required=True)
parser.add_argument('--output_dir','-o', type=str, help='path of the output directory', default='testing_results', required=False)
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--gamma_factor', type=float, default=2.2, help='gamma correction factor')

K = 25 # number of elements en the base
opt = parser.parse_args()
blurry_image_filename = opt.img
model_file = opt.model

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)


img_name = blurry_image_filename.split('/')[-1]
img_name, ext = img_name.split('.')

print('loading image ',blurry_image_filename)
blurry_image = imread(blurry_image_filename)

two_heads = TwoHeadsNetwork(K).cuda(opt.gpu_id)
print('loading weight\'s model')
two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % opt.gpu_id))

two_heads.eval()

# Blurry image is transformed to pytorch format
transform = transforms.Compose([
    transforms.ToTensor()
])
blurry_tensor = transform(blurry_image).cuda(opt.gpu_id)


# Kernels and masks are estimated
blurry_tensor_to_compute_kernels = blurry_tensor**opt.gamma_factor - 0.5
kernels_estimated, masks_estimated = two_heads(blurry_tensor_to_compute_kernels[None,:,:,:])

kernels_val_n = kernels_estimated[0, :, :, :]
kernels_val_n_ext = kernels_val_n[:, np.newaxis, :, :]

blur_kernel_val_grid = make_grid(kernels_val_n_ext, nrow=K,
                                                   normalize=True, scale_each=True,pad_value=1)
mask_val_n = masks_estimated[0, :, :, :]
mask_val_n_ext = mask_val_n[:, np.newaxis, :, :]
blur_mask_val_grid = make_grid(mask_val_n_ext, nrow=K, pad_value=1)

imsave(os.path.join(opt.output_dir, img_name + '_kernels.png' ),
           img_as_ubyte(blur_kernel_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))
print('Kernels saved in ',os.path.join(opt.output_dir, img_name + '_kernels.png') )

imsave(os.path.join(opt.output_dir, img_name + '_masks.png' ),
           img_as_ubyte(blur_mask_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))
print('Mixing coefficients saved in ',os.path.join(opt.output_dir, img_name + '_mask.png' ))

win_kernels_grid = save_kernels_grid(blurry_tensor, torch.flip(kernels_estimated[0], dims=(1,2)), masks_estimated[0], os.path.join(opt.output_dir, img_name + '_kernels_grid.png'))
print('Kernels grid saved in ',os.path.join(opt.output_dir, img_name + '_kernels_grid.png' ))
