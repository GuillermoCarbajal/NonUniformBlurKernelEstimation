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

parser = argparse.ArgumentParser()
parser.add_argument('--blurry_images', '-b', type=str, required=True, help='list with the original blurry images or path to a blurry image')
parser.add_argument('--reblur_model', '-m', type=str, required=True, help='two heads reblur model')
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--n_iters', '-n', type=int, default=30)
parser.add_argument('--n_den_iters', '-nd', type=int, default=15)
parser.add_argument('--lambda_reblur','-l', type=float, default=1)
parser.add_argument('--ratio_sigma', type=float, default=2)
parser.add_argument('--output_folder','-o', type=str, help='output folder', default='sharp_opt_output')
parser.add_argument('--resize_factor','-rf', type=float, default=1)
parser.add_argument('--denoiser_model', '-dm', type=str, default='/home/guillermo/github/DPIR/model_zoo/drunet_color.pth', help='denoiser model')
parser.add_argument('--use_clamp', type=int, default=0, help='Use clamp in the output')
parser.add_argument('--alpha', type=float, default=0.9)


B = 25 # number of kernels
BLUR_KERNEL_SIZE = 33
SAVE_INTERMIDIATE = False
args = parser.parse_args()



def get_images_list(list_path):

    with open(list_path) as f:
        images_list = f.readlines()
        images_list = [l[:-1] for l in images_list]
    f.close()

    return images_list




if args.blurry_images.endswith('.txt'):
    blurry_images_list = get_images_list(args.blurry_images)
else:
    blurry_images_list = [args.blurry_images]



two_heads = TwoHeadsNetwork(B).cuda(args.gpu_id)
two_heads.load_state_dict(torch.load(args.reblur_model, map_location='cuda:%d' % args.gpu_id))
two_heads.eval()

rhos, sigmas = deep_prior_initialization(num_iters=args.n_den_iters, GPU=args.gpu_id)

# denoiser model
denoiser = get_DRUNet(args.denoiser_model,args.gpu_id)


for name, param in two_heads.named_parameters():
    if param.requires_grad:
        param.requires_grad = False

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

PSNR_gains, PSNRs, reblur_losses_val = [], [], []

for i, blurry_path in enumerate(blurry_images_list):
    print('---')
    print('Processsing ', blurry_path)
    img_name, ext = blurry_path.split('/')[-1].split('.')
    blurry_image =  imread(os.path.join(blurry_path))
    blurry_image = blurry_image[:,:,:3]
    M, N, C = blurry_image.shape
    if args.resize_factor != 1:
        new_shape = (int(args.resize_factor*M), int(args.resize_factor*N), C )
        blurry_image = resize(blurry_image, new_shape).astype(np.float32)


    blurry_tensor = transforms.ToTensor()(blurry_image)
    blurry_tensor = blurry_tensor[None,:,:,:]
    blurry_tensor = blurry_tensor.cuda(args.gpu_id) - 0.5
    save_image(tensor2im(blurry_tensor[0].detach()), os.path.join(args.output_folder,
                                                       img_name + '.png' ))

    # initialization
    padding = torch.nn.ReflectionPad2d(BLUR_KERNEL_SIZE // 2)
    output = padding(blurry_tensor)
    output.requires_grad = True
    output_optimizer = torch.optim.Adam([output, ], lr=1e-6)

    # Compute kernels
    with torch.no_grad():
        kernels, masks = two_heads(blurry_tensor)
        save_kernels_grid(blurry_tensor[0]+0.5,kernels[0], masks[0], os.path.join(args.output_folder, img_name + '_kernels.png'))
        output_reblurred = forward_reblur(output, kernels, masks, args.gpu_id)


    reblur_loss = torch.mean((output_reblurred - blurry_tensor) ** 2)

    psnr_val = 10 * np.log10(1 / reblur_loss.item())
    print('PSNR gt_reblurred-real_A', psnr_val)
    reblur_losses_val.append(psnr_val)

    for it in range(args.n_iters):

        output_reblurred = forward_reblur(output, kernels, masks, args.gpu_id)


        # compute ||K*I -B||
        reblur_loss = torch.mean((output_reblurred - blurry_tensor) ** 2)

        # do optimization step
        output_optimizer.zero_grad()
        output.retain_grad()
        reblur_loss.backward(retain_graph=True)


        # clamp I to valid range
        if args.use_clamp:
            output.data = (output.data - output.grad.data  * args.lambda_reblur * (args.resize_factor*M)*(args.resize_factor*N)).clamp(-0.5, 0.5)
        else:
            output.data = (output.data - output.grad.data  * args.lambda_reblur * (args.resize_factor*M)*(args.resize_factor*N))


        denoiser_period = np.ceil(args.n_iters / len(sigmas)).astype(np.int)
        is_denoiser_step = (it+1) % denoiser_period == 0
        if (it > 0 and is_denoiser_step) or it==(args.n_iters-1):

            denoiser_call_counter = int((it+1) / denoiser_period)
            print('iter=%d/%d , periodicity=%d, call counter =%d' % (it+1, args.n_iters, denoiser_period, denoiser_call_counter))

            z = output + 0.5
            z = run_DRUNet(denoiser, z, sigmas[denoiser_call_counter-1]/args.ratio_sigma, compute_grad=False)
            z = z - 0.5

            output.data = (1 - args.alpha)*output.data + args.alpha * z.data
            output.requires_grad = True




        if ( ((SAVE_INTERMIDIATE and it % (args.n_iters//10) == 0) or  it == (args.n_iters - 1)) ): #''' '''

            if it == (args.n_iters - 1):
                filename = os.path.join(args.output_folder, img_name + '_restored_l_%f_nd_%f_n_%f_r_%f_clamp_%d.png' % (args.lambda_reblur, args.n_den_iters, args.n_iters,args.ratio_sigma, int(args.use_clamp)))
            else:
                filename = os.path.join(args.output_folder,img_name + '_iter_%06i.png' % it )


            save_image(tensor2im(output[0, :, BLUR_KERNEL_SIZE // 2:-(BLUR_KERNEL_SIZE // 2),
                                              BLUR_KERNEL_SIZE // 2:-(BLUR_KERNEL_SIZE // 2)].detach().clamp(-0.5, 0.5)), filename)


            print(i, it, 'reblur: ', reblur_loss.item())

            torch.cuda.empty_cache()
