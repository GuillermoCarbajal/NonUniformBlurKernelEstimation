
import os
import argparse
import torch
from torchvision import transforms

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

from models.TwoHeadsNetwork_CVPR import TwoHeadsNetwork
from utils.visualization import save_image, tensor2im, save_kernels_grid
from utils.reblur import forward_reblur
from utils.restoration import get_DRUNet, run_DRUNet, deep_prior_initialization



# from sklearn.metrics import precision_recall_curve, average_precision_score # used to compute scores in the paper on CUHK dataset


parser = argparse.ArgumentParser()
parser.add_argument('--blurry_images', '-b', type=str, required=True, help='list with the original blurry images or path to a blurry image')
parser.add_argument('--reblur_model', '-m', type=str, required=True, help='two heads reblur model')
parser.add_argument('--K', '-k', type=int, default=25, help='number of kernels in two heads model')
parser.add_argument('--blur_kernel_size', '-bks', type=int, default=33, help='blur_kernel_szie')
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--output_folder','-o', type=str, help='output folder', default='sharp_opt_output')
parser.add_argument('--root_dir','-rd', type=str, required=False, default='')
parser.add_argument('--n_images', type=int, default=0)
parser.add_argument('--resize_factor','-rf', type=float, default=1)
parser.add_argument('--rescale_factor','-sf', type=float, default=1)

args = parser.parse_args()
BLUR_KERNEL_SIZE = args.blur_kernel_size
SAVE_INTERMIDIATE = args.save_intermidiate


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
else:
    blurry_images_list = [args.blurry_images]

if args.n_images > 0:
    blurry_images_list = blurry_images_list[:args.n_images]

two_heads = TwoHeadsNetwork(B).cuda(args.gpu_id)
two_heads.load_state_dict(torch.load(args.reblur_model, map_location='cuda:%d' % args.gpu_id))
two_heads.eval()



if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


blur_type =  'motion'

fnames = glob.glob('/data/blurdetect/image/%s*.jpg' % blur_type)

norm_threshold = 0.25

for i,blurry_path in enumerate(fnames):
    img_name, ext = blurry_path.split('/')[-1].split('.')
    blurry_image =  imread(os.path.join(args.root_dir, blurry_path))

    if args.resize_factor != 1:
        if len(blurry_image.shape) == 2:
            blurry_image = gray2rgb(blurry_image)
        M, N, C = blurry_image.shape
        new_shape = (int(args.resize_factor*M), int(args.resize_factor*N), C )
        blurry_image = resize(blurry_image,new_shape).astype(np.float32)

    blurry_tensor = transforms.ToTensor()(blurry_image)
    blurry_tensor = blurry_tensor[None,:,:,:]
    blurry_tensor = blurry_tensor.cuda(args.gpu_id) - 0.5



    with torch.no_grad():
        blurry_tensor_to_compute_kernels = args.rescale_factor * blurry_tensor
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels)

        masks = masks[0].detach().cpu().numpy()
        kernels = kernels[0].detach().cpu().numpy()




        blur_mask = np.zeros_like(masks[0])
        sharp_mask = np.zeros_like(blur_mask)
        
        for k in range(kernels.shape[0]):
            norm_k = np.linalg.norm(kernels[k].squeeze())
            mask = masks[k]


            if norm_k > norm_threshold:
                sharp_mask += mask
            else:
                blur_mask += mask

        imsave('masks_out/blur_mask_%f.png' % blur_threshold, blur_mask)
