import torch
from utils.utils_sisr import pre_calculate, data_solution
from utils.visualization import save_image, tensor2im
import os
from utils.reblur import saturate_image
import numpy as np
from DPIR.models.network_unet import UNetRes as drunet
from DPIR.models.network_dncnn import IRCNN as net
from DPIR.utils import utils_pnp as pnp
from DPIR.utils import utils_model

def restore(blurry_tensor, kernels, masks, GPU, arho= 0.0056):
    kernels_conv = torch.flip(kernels,[2,3])
    N, C, H, W = blurry_tensor.shape
    K = kernels.shape[1]
    y = torch.zeros((N, C, H, W)).cuda(GPU)
    rho = torch.tensor([arho]).cuda(GPU)
    tau = rho.repeat(1, 1, 1, 1)
    sf = 1
    for n in range(N):
        for k in range(K):
            FB, FBC, F2B, FBFy = pre_calculate(blurry_tensor[n:n+1, :, :, :],
                                               kernels_conv[n:n+1, k:k + 1, :, :], sf)
            x_k = data_solution(blurry_tensor[n:n+1, :, :, :], FB, FBC, F2B, FBFy, tau, sf)
            y[n:n+1, :,:,:] += masks[n:n+1, k:k + 1, :, :] * x_k

    sat_y = saturate_image(y)
    return sat_y

def restore_with_GT(blurry_tensor, sharp_image, kernels, masks, GPU, arho= 0.0056):
    kernels_conv = torch.flip(kernels,[2,3])
    N, C, H, W = blurry_tensor.shape
    K = kernels.shape[1]
    y = torch.zeros((N, C, H, W)).cuda(GPU)
    rho = torch.tensor([arho]).cuda(GPU)
    tau = rho.repeat(1, 1, 1, 1)
    sf = 1
    for n in range(N):
        for k in range(K):
            FB, FBC, F2B, FBFy = pre_calculate(blurry_tensor[n:n+1, :, :, :],
                                               kernels_conv[n:n+1, k:k + 1, :, :], sf)
            x_k = data_solution(sharp_image[n:n+1, :, :, :], FB, FBC, F2B, FBFy, tau, sf)
            y[n:n+1, :,:,:] += masks[n:n+1, k:k + 1, :, :] * x_k

    sat_y = saturate_image(y)
    return sat_y



def deep_prior_initialization(num_iters=8, GPU=0):

    noise_level_img = 7.65 / 255.0  # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level of model, default 0
    # model_name = 'drunet_color'  # 'drunet_gray' | 'drunet_color' | 'ircnn_gray' | 'ircnn_color'
    x8 = True  # default: False, x8 to boost performance
    modelSigma1 = 49
    modelSigma2 = noise_level_model * 255.

    rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=num_iters,
                                     modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
    rhos, sigmas = torch.tensor(rhos).to(GPU), torch.tensor(sigmas).to(GPU)

    return rhos, sigmas

def get_DRUNet(model_path, GPU, n_channels=3):

    denoiser = drunet(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                      downsample_mode="strideconv", upsample_mode="convtranspose")
    denoiser.load_state_dict(torch.load(model_path), strict=True)
    denoiser.eval()
    for _, v in denoiser.named_parameters():
        v.requires_grad = False
    denoiser = denoiser.to(GPU)

    return denoiser

def get_IRCNN(model_path, n_channels=3):
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
    model25 = torch.load(model_path)
    former_idx = 0
    return model, model25, former_idx

def run_IRCNN(model, img, model25, sigma, former_idx, GPU, compute_grad=True):
    current_idx = np.int(np.ceil(sigma.cpu().numpy() * 255. / 2.) - 1)
    if current_idx != former_idx:
        model.load_state_dict(model25[str(current_idx)], strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(GPU)
    former_idx = current_idx
    if compute_grad:
        deblurred_img = model(img)
    else:
        with torch.no_grad():
            deblurred_img = model(img)
    return deblurred_img, model, model25, former_idx


def run_DRUNet(denoiser, inp_img, sigma, compute_grad=True):

    aux = torch.cat((inp_img, sigma.repeat(1, 1, inp_img.shape[2], inp_img.shape[3])), dim=1)
    aux = aux.float()  # added GC
    if compute_grad:
        denoised_img = utils_model.test_mode(denoiser, aux, mode=1, refield=32, min_size=256, modulo=16)
    else:
        with torch.no_grad():
            denoised_img = utils_model.test_mode(denoiser, aux, mode=1, refield=32, min_size=256, modulo=16)

    return denoised_img