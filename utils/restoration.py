import torch

from utils.visualization import save_image, tensor2im
from utils.reblur import forward_reblur, apply_saturation_function
import numpy as np
import os
from torch.nn import functional as F


def gradient_v2(img):

    G_x = torch.zeros_like(img).to(img.device)
    G_y = torch.zeros_like(img).to(img.device)
    C = img.shape[1]

    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).to(img.device)

    a = a.view((1, 1, 3, 3))

    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).to(img.device)

    b = b.view((1, 1, 3, 3))

    for c in range(C):

        G_x[0,c,:,:] = F.conv2d(img[:,c:c+1,:,:], a, padding=1)

        G_y[0,c,:,:]  = F.conv2d(img[:,c:c+1,:,:], b, padding=1)

    return G_y, G_x

def gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dy, dx


def normalised_gradient_divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    gy, gx = gradient_v2(F)
    eps = 1e-9
    ngy = gy.clone().to(F.device)
    ngx = gx.clone().to(F.device)
    norm = torch.sqrt(gx**2 + gy**2)
    ngy[norm < eps]=  eps
    ngx[norm < eps] = eps
    ngy[norm >= eps] = torch.div(gy[norm >= eps], norm[norm >= eps])
    ngx[norm >= eps] = torch.div(gx[norm >= eps], norm[norm >= eps])
    gxy, gxx = gradient_v2(ngx)
    gyy, gyx = gradient_v2(ngy)
    return gxx + gyy



def RL_restore(blurry_tensor, initial_output, kernels, masks, n_iters, GPU,
               SAVE_INTERMIDIATE=True, method='basic', saturation_threshold=0.7,
               reg_factor=1e-3, gamma_correction_factor=1.0, isDebug=False):

    epsilon = 1e-6

    deblug_folder = './RL_deblug'
    if not os.path.exists(deblug_folder):
        os.makedirs(deblug_folder)

    blurry_tensor_ph = blurry_tensor ** gamma_correction_factor  # to photons space
    output = initial_output
    kernels_flipped = torch.flip(kernels, dims=(2, 3))
    for it in range(n_iters):
        output_ph = output** gamma_correction_factor
        output_reblurred_ph = forward_reblur(output_ph, kernels, masks, GPU, size='same',
                                          padding_mode='reflect', manage_saturated_pixels=False, max_value=1)
        #output_reblurred = output_reblurred_ph**(1.0/gamma_correction_factor)  # to pixels space
        if method=='basic':
            relative_blur = torch.div(blurry_tensor_ph, output_reblurred_ph + epsilon)

        elif method=='function':
            R = apply_saturation_function(output_reblurred_ph, max_value=1)
            R_prima = apply_saturation_function(output_reblurred_ph, max_value=1, get_derivative=True)
            relative_blur =  torch.div(blurry_tensor_ph * R_prima, R + epsilon) + 1 - R_prima
        elif method=='masked':
            mask = blurry_tensor < saturation_threshold
            relative_blur =  torch.div(blurry_tensor_ph * mask, output_reblurred_ph + epsilon) + 1 - mask.float()

        error_estimate = forward_reblur(relative_blur, kernels_flipped, masks, GPU, size='same',
                                        padding_mode='reflect', manage_saturated_pixels=False, max_value=1)

        output_ph = output_ph * error_estimate
        J_reg_grad = reg_factor * normalised_gradient_divergence(output)
        output = output_ph**(1.0/gamma_correction_factor)*(1.0/(1-J_reg_grad))


        if isDebug:
            # compute ||K*I -B||
            # reblur_loss = model.reblurLoss(2*output_reblurred,  model.real_A[:,:, K//2:-K//2+1, K//2:-K//2+1]) * opt.lambda_reblur
            reblur_loss = torch.mean((output_reblurred_ph**(1.0/gamma_correction_factor) - blurry_tensor) ** 2)
            PSNR_reblur = 10 * np.log10(1 / reblur_loss.item())
            print('PSNR_reblur', PSNR_reblur)
            if ( ((SAVE_INTERMIDIATE and it % np.max([1, n_iters // 10]) == 0)or  it == (n_iters - 1)) ):

                if it == (n_iters - 1):
                    filename = os.path.join(deblug_folder,'iter_%06i_restored.png' % (n_iters))
                else:
                    filename = os.path.join(deblug_folder, 'iter_%06i.png' % it )

                save_image(tensor2im(output[0].detach().clamp(0, 1)-0.5), filename)

                print(it, 'PSNR_reblur: ', PSNR_reblur.item())



    return output


def combined_RL_restore(blurry_tensor, initial_output, kernels, masks, n_iters, GPU,
                        SAVE_INTERMIDIATE=True, saturation_threshold=0.9, reg_factor=1e-3,
                        optim_iters=False, gamma_correction_factor=1.0, apply_smoothing=True, apply_erosion=True,
                        apply_dilation=True, isDebug=False):
    epsilon = 1e-6
    output = initial_output

    erosion_kernel = torch.ones(1, 1, 3, 3).to(output.device)
    smooth_kernel = 1.0 / 9 * torch.ones(1, 1, 3, 3).to(output.device)

    # smooth_kernel = torch.sum(kernels, dim=1, keepdim=True)/kernels.shape[1]
    # smooth_kernel =  torch.Tensor([[[[0.9,0.94,0.9],[0.94,1,0.94], [0.9,0.94,0.9]]]]).to(output.device)
    # smooth_kernel = smooth_kernel/smooth_kernel.sum()
    previous_reblur_loss = 1000
    blurry_tensor_ph = blurry_tensor**gamma_correction_factor
    kernels_flipped = torch.flip(kernels, dims=(2, 3))
    for it in range(n_iters):

        u = (output < saturation_threshold).float()

        # erosion
        if apply_erosion:
            for c in range(u.shape[1]):
                u[0:1, c:c + 1, :, :] = 1 - (F.conv2d(1 - u[:, c:c + 1, :, :], erosion_kernel,
                                                      padding=erosion_kernel.shape[2] // 2) > 0).float()

        # smoothing
        if apply_smoothing:
            for c in range(u.shape[1]):
                u[0:1, c:c + 1, :, :] = F.conv2d(u[:, c:c + 1, :, :], smooth_kernel, padding=smooth_kernel.shape[2] // 2)


        f_u = u * output
        f_s = output - f_u


        if apply_dilation:
            non_zero_elements = torch.sum(kernels, dim=1, keepdim=True)
            dilation_kernel = (non_zero_elements > 0.25 * non_zero_elements.max()).float()
            # dilation_kernel = torch.repeat_interleave(dilation_kernel/dilation_kernel.sum(), 3, dim=0)
            v = torch.zeros_like(u).to(output.device)
            for c in range(u.shape[1]):
                v[0:1, c:c + 1, :, :] = F.conv2d(1 - u[:, c:c + 1, :, :], dilation_kernel,
                                                 padding=dilation_kernel.shape[2] // 2)
            v = 1 - (v > 0).float()
        else:
            v=u

        output_ph = output ** gamma_correction_factor  # from pixels to photons
        output_reblurred_ph = forward_reblur(output_ph, kernels, masks, GPU, size='same',
                                          padding_mode='reflect', manage_saturated_pixels=False, max_value=1)
        #output_reblurred = output_reblurred_ph ** (1.0 / gamma_correction_factor)  # from photons to pixels space

        R = apply_saturation_function(output_reblurred_ph, max_value=1)
        R_prima = apply_saturation_function(output_reblurred_ph, max_value=1, get_derivative=True)
        #print(output_reblurred_ph.device, R.device, R_prima.device)
        relative_blur_u = torch.div(blurry_tensor_ph * R_prima * v, R + epsilon) + 1 - R_prima * v
        relative_blur_s = torch.div(blurry_tensor_ph * R_prima, R + epsilon) + 1 - R_prima

        error_estimate_u = forward_reblur(relative_blur_u,kernels_flipped , masks, GPU, size='same',
                                          padding_mode='reflect', manage_saturated_pixels=False, max_value=1)
        error_estimate_s = forward_reblur(relative_blur_s, kernels_flipped, masks, GPU, size='same',
                                          padding_mode='reflect', manage_saturated_pixels=False, max_value=1)


        f_u_ph = ((f_u**gamma_correction_factor) *  error_estimate_u)
        f_s_ph = ((f_s**gamma_correction_factor) * error_estimate_s)
        f_u = f_u_ph **(1.0/gamma_correction_factor)
        f_s = f_s_ph ** (1.0 / gamma_correction_factor)
        output_it = (f_u + f_s)

        J_reg_grad = reg_factor * normalised_gradient_divergence(output)
        output_it *= (1.0 / (1 - J_reg_grad))

        with torch.no_grad():
            reblur_loss = torch.mean((output_reblurred_ph**(1.0/gamma_correction_factor) - blurry_tensor) ** 2)
            # print(previous_reblur_loss, reblur_loss.item(), previous_reblur_loss - reblur_loss.item())
            if ((previous_reblur_loss - reblur_loss.item()) < epsilon and optim_iters):
                break
            else:
                output = output_it

            previous_reblur_loss = reblur_loss.item()

        if isDebug:
            E = (output**2).mean()
            print('Energy=%f' % E)
            print('Range=[%f, %f]' % (output.min(), output.max()))
            if (((SAVE_INTERMIDIATE and it % np.max([1, n_iters // 10]) == 0) or it == (n_iters - 1))):  # ''' '''
                PSNR_reblur = 10 * np.log10(1 / reblur_loss.item())
                print(it, 'PSNR_reblur: ', PSNR_reblur.item())





        del output_reblurred_ph, u, v, f_u, f_s, R, R_prima, relative_blur_u, relative_blur_s, error_estimate_u, error_estimate_s
        torch.cuda.empty_cache()




    return output








