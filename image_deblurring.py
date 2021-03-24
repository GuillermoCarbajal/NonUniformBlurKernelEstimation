import numpy as np
import argparse
from models.TwoHeadsNetwork_ICCV import TwoHeadsNetwork

import torch
from scipy.io import savemat

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2ycbcr, ycbcr2rgb, gray2rgb, rgb2gray


from torchvision import transforms
import os
import json

from utils.visualization import save_image, tensor2im, save_kernels_grid
from utils.reblur import forward_reblur
from utils.restoration import RL_restore, combined_RL_restore

parser = argparse.ArgumentParser()
parser.add_argument('--blurry_images', '-b', type=str, required=True, help='list with the original blurry images or path to a blurry image')
parser.add_argument('--initial_images', '-r', type=str, required=True, help='list with the initial restorations or path to initial image')
parser.add_argument('--sharp_images', '-s', type=str, required=False, help='list with the original sharp images or path to sharp image')
parser.add_argument('--reblur_model', '-m', type=str, required=True, help='two heads reblur model')
parser.add_argument('--K', '-k', type=int, default=25, help='number of kernels in two heads model')
parser.add_argument('--blur_kernel_size', '-bks', type=int, default=33, help='blur_kernel_szie')
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--n_iters', '-n', type=int, default=30)
parser.add_argument('--output_folder','-o', type=str, help='output folder', default='sharp_opt_output')
parser.add_argument('--root_dir','-rd', type=str, required=False, default='')
parser.add_argument('--resize_factor','-rf', type=float, default=1)
parser.add_argument('--saturation_method', type=str, default='combined')
parser.add_argument('--regularization','-reg', type=str, help='regularization method')
parser.add_argument('--reg_factor', type=float, default=1e-3, help='regularization factor')
parser.add_argument('--sat_threshold','-sth', type=float, default=0.99)
parser.add_argument('--gamma_factor', type=float, default=1.0, help='gamma correction factor')
parser.add_argument('--optim_iters', action='store_true', default=False, help='stop iterating when reblur loss is 1e-6')
parser.add_argument('--smoothing', action='store_true', default=False, help='apply smoothing to the saturated region mask')
parser.add_argument('--erosion', action='store_true', default=False, help='apply erosion to the non-saturated region')
parser.add_argument('--dilation', action='store_true', default=False, help='apply dilation to the saturated region using the kernel as structural element')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')

'''
-b /media/carbajal/OS/data/datasets/cvpr16_deblur_study_real_dataset/real_dataset/coke.jpg -r /media/carbajal/OS/data/datasets/cvpr16_deblur_study_real_dataset/real_dataset/coke.jpg -m /media/carbajal/OS/data/models/ade_dataset/NoFC/gamma_correction/L1/L2_epoch150_epoch150_L1_epoch900.pkl -n 20 -o coke --saturation_method 'combined'
'''

args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

with open(os.path.join(args.output_folder, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def get_images_list(list_path):

    with open(list_path) as f:
        images_list = f.readlines()
        images_list = [l[:-1] for l in images_list]
    f.close()

    return images_list

def compute_kernels_from_base(base_kernels, masks, GPU=0):
    output_kernel = torch.empty((masks.shape[-2],masks.shape[-1],base_kernels.shape[-2]*base_kernels.shape[-1])).cuda(GPU)
    for k in range(base_kernels.shape[0]):
        kernel_k = base_kernels[ k, :, :].view(-1)
        masks_k = masks[k, :, :]
        aux = masks_k[:, :, None] * kernel_k[None, None, :]
        output_kernel += aux
        del aux
        torch.cuda.empty_cache()
    return output_kernel


if args.blurry_images.endswith('.txt'):
    blurry_images_list = get_images_list(args.blurry_images)
    initial_images_list = get_images_list(args.initial_images)
else:
    blurry_images_list = [args.blurry_images]
    initial_images_list = [args.initial_images]



two_heads = TwoHeadsNetwork(args.K).cuda(args.gpu_id)
two_heads.load_state_dict(torch.load(args.reblur_model, map_location='cuda:%d' % args.gpu_id))
two_heads.eval()

PSNR_gains, PSNRs, reblur_losses_val = [], [], []
SSIM_gains, SSIMs = [], []

for i,(blurry_path, restored_path) in enumerate(zip(blurry_images_list, initial_images_list)):

    img_name, ext = blurry_path.split('/')[-1].split('.')
    blurry_image =  imread(os.path.join(args.root_dir, blurry_path))
    blurry_image = blurry_image[:,:,:3]
    #blurry_image = blurry_image[-288: ,:288,:]
    M, N, C = blurry_image.shape
    if args.resize_factor != 1:
        if len(blurry_image.shape) == 2:
            blurry_image = gray2rgb(blurry_image)
        new_shape = (int(args.resize_factor*M), int(args.resize_factor*N), C )
        blurry_image = resize(blurry_image,new_shape).astype(np.float32)

    blurry_tensor = transforms.ToTensor()(blurry_image)
    blurry_tensor = blurry_tensor[None,:,:,:]
    blurry_tensor = blurry_tensor.cuda(args.gpu_id)


    initial_image = imread(os.path.join(args.root_dir, restored_path))
    initial_image = initial_image[:, :, :3]


    #initial_image = initial_image[-288: ,:288, :]
    if args.resize_factor != 1:
        if len(initial_image.shape) == 2:
            initial_image = gray2rgb(initial_image)
        new_shape = (int(args.resize_factor*M), int(args.resize_factor*N), C )
        initial_image = resize(initial_image,new_shape).astype(np.float32)



    initial_restoration_tensor = transforms.ToTensor()(initial_image)
    initial_restoration_tensor = initial_restoration_tensor[None, :, :, :]
    initial_restoration_tensor = initial_restoration_tensor.cuda(args.gpu_id)

    save_image(tensor2im(initial_restoration_tensor[0] - 0.5), os.path.join(args.output_folder,
                                                       img_name + '.png' ))

    with torch.no_grad():

        blurry_tensor_to_compute_kernels = blurry_tensor**args.gamma_factor - 0.5
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels)
        save_kernels_grid(blurry_tensor[0],kernels[0], masks[0], os.path.join(args.output_folder, img_name + '_kernels'+'.png'))


    output = initial_restoration_tensor


    with torch.no_grad():

        if args.saturation_method == 'combined':
            output = combined_RL_restore(blurry_tensor, output, kernels, masks, args.n_iters,
                                         args.gpu_id, SAVE_INTERMIDIATE=True, saturation_threshold=args.sat_threshold,
                                         reg_factor=args.reg_factor, optim_iters=args.optim_iters, gamma_correction_factor=args.gamma_factor,
                                         apply_dilation=args.dilation, apply_smoothing=args.smoothing, apply_erosion=args.erosion, isDebug=args.debug)
        else:
            output = RL_restore(blurry_tensor, output, kernels, masks, args.n_iters,
                                args.gpu_id, SAVE_INTERMIDIATE=True,
                                method=args.saturation_method,gamma_correction_factor=args.gamma_factor,
                                saturation_threshold=args.sat_threshold, reg_factor=args.reg_factor, isDebug=args.debug)


    output_img = tensor2im(torch.clamp(output[0],0,1) - 0.5)
    save_image(output_img, os.path.join(args.output_folder, img_name + '_restored.png' ))

