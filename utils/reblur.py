import torch
import numpy as np
import torch.nn.functional as F


def saturate_image(img, max_value=0.5):
    '''
    Implements the saturated function proposed by Whyte
    https://www.di.ens.fr/willow/research/saturation/whyte11.pdf
    :param img: input image may have values above max_value
    :param max_value: maximum value
    :return:
    '''

    a=50
    img[img>1]=1  # to avoid overflow in exponential
    saturated_image = img - 1.0/a*torch.log(1+torch.exp(a*(img - max_value)))
    output_image = F.relu(saturated_image + 0.5) - 0.5
    return output_image

def forward_reblur(sharp_estimated, kernels, masks, GPU=0, manage_saturated_pixels=True):
    n_kernels = kernels.size(1)
    K = kernels.size(2)
    N = sharp_estimated.size(0)
    C = sharp_estimated.size(1)
    H = sharp_estimated.size(2) - K + 1
    W = sharp_estimated.size(3) - K + 1

    output_reblurred = torch.empty(N, n_kernels, C, H, W).cuda(GPU)

    for num in range(N):  # print('n = ',n)
        for c in range(C):
            # print('gt padded one channel shape: ', gt_n_padded_c.shape)
            conv_output = F.conv2d(sharp_estimated[num:num + 1, c:c + 1, :, :],
                                   kernels[num][:, np.newaxis, :, :])
            # print('conv output shape: ', conv_output.shape)
            output_reblurred[num:num + 1, :, c, :, :] = conv_output * masks[num:num + 1]
            del conv_output


    # print('reblur_image shape before sum:', reblurred_images.shape)
    output_reblurred = torch.sum(output_reblurred, (1))

    if manage_saturated_pixels:
        output_reblurred = saturate_image(output_reblurred)

    return output_reblurred

def compute_Lp_norm(kernels_tensor, p):
    N, K, S, S = kernels_tensor.shape
    output = 0
    for n in range(N):
        for k in range(K):
            kernel = kernels_tensor[n, k, :, :]
            p_norm = torch.pow(torch.sum(torch.pow(torch.abs(kernel), p)), 1. / p)
            output = output + p_norm
    return output/(N*K)

def compute_total_variation_loss(img):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).abs()).sum()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).abs()).sum()
    return  (tv_h + tv_w)/(img.shape[0]*img.shape[1])


def compute_masks_regularization_loss(masks, MASKS_REGULARIZATION_TYPE, MASKS_REGULARIZATION_LOSS_FACTOR):

    masks_regularization_loss = 0.
    if MASKS_REGULARIZATION_TYPE == 'L2':
        masks_regularization_loss = MASKS_REGULARIZATION_LOSS_FACTOR * torch.mean(masks * masks)
    elif MASKS_REGULARIZATION_TYPE == 'TV':
        masks_regularization_loss = compute_total_variation_loss(masks, MASKS_REGULARIZATION_LOSS_FACTOR)

    return masks_regularization_loss

def compute_kernels_regularization_loss(kernels, KERNELS_REGULARIZATION_TYPE, KERNELS_REGULARIZATION_LOSS_FACTOR):

    kernels_regularization_loss = 0.

    if KERNELS_REGULARIZATION_TYPE == 'L2':
        kernels_regularization_loss = KERNELS_REGULARIZATION_LOSS_FACTOR * torch.mean(kernels ** 2)
    if KERNELS_REGULARIZATION_TYPE == 'L1':
        kernels_regularization_loss = KERNELS_REGULARIZATION_LOSS_FACTOR * torch.mean(
            torch.abs(kernels))
    elif KERNELS_REGULARIZATION_TYPE == 'TV':
        kernels_regularization_loss = compute_total_variation_loss(kernels,
                                                                    KERNELS_REGULARIZATION_LOSS_FACTOR)
    elif KERNELS_REGULARIZATION_TYPE == 'Lp':
        kernels_regularization_loss = KERNELS_REGULARIZATION_LOSS_FACTOR * compute_Lp_norm(kernels,
                                                                                            p=0.5)

    return kernels_regularization_loss


def compute_reblurred_image_and_kernel_loss(sharp_image, kernels, masks, gt_kernels, gt_masks,
                        masks_weights, kernels_loss_type,  GPU=0, manage_saturated_pixels=True):

    n_kernels = kernels.size(1)
    K = kernels.size(2)
    N = sharp_image.size(0)
    C = sharp_image.size(1)
    H = sharp_image.size(2) - K + 1
    W = sharp_image.size(3) - K + 1
    reblurred_images = torch.empty(N, n_kernels, C, H, W).cuda(GPU)

    Kgt = gt_kernels.shape[1]
    Wk = gt_kernels.shape[2]  # kernel side

    kernels_loss = torch.Tensor([0.]).cuda(GPU);
    for n in range(N):
        gt_masks_nn = gt_masks[n].view(Kgt, H * W)  # *(1/(masks_sums[n][nonzero]))
        gt_kernels_nn = gt_kernels[n].view(Kgt, Wk * Wk)
        gt_kernels_per_pixel = torch.mm(gt_kernels_nn.t(), gt_masks_nn)

        predicted_kernels_per_pixel = torch.mm(kernels[n].contiguous().view(n_kernels, Wk * Wk).t(),
                                               masks[n].contiguous().view(n_kernels, H * W))

        if kernels_loss_type == 'L2':
            per_pixel_kernel_diff = (predicted_kernels_per_pixel - gt_kernels_per_pixel)**2
        elif kernels_loss_type == 'L1':
            per_pixel_kernel_diff = (predicted_kernels_per_pixel - gt_kernels_per_pixel).abs()

        kernels_loss += (per_pixel_kernel_diff.sum(0) * masks_weights[n].view(H * W)).sum() / N

        for c in range(C):
            conv_output = F.conv2d(sharp_image[n:n + 1, c:c + 1, :, :], kernels[n][:, np.newaxis, :, :])
            reblurred_images[n:n + 1, :, c, :, :] = conv_output * masks[n:n + 1]

    reblurred_images = torch.sum(reblurred_images, (1))

    if manage_saturated_pixels:
        output_reblurred = saturate_image(reblurred_images)


    return output_reblurred, kernels_loss


def compute_gradient_and_kernel_loss(blur_image, sharp_image, kernels, masks, gt_kernels, gt_masks,
                        masks_weights, kernels_loss_type, GPU=0):

    N = sharp_image.size(0)
    gt_sharp_y = torch.zeros_like(sharp_image)
    gt_sharp_y[:, :, 1:, :] = sharp_image[:, :, 1:, :] - sharp_image[:, :, :-1, :]
    gt_sharp_x = torch.zeros_like(sharp_image)
    gt_sharp_x[:, :, :, 1:] = sharp_image[:, :, :, 1:] - sharp_image[:, :, :, :-1]

    blurry_y = blur_image[:, :, 1:, :] - blur_image[:, :, :-1, :]
    blurry_x = blur_image[:, :, :, 1:] - blur_image[:, :, :, :-1]
    reblurred_images_y, kernels_loss_y = compute_reblurred_image_and_kernel_loss(
        gt_sharp_y, kernels, masks, gt_kernels, gt_masks, masks_weights,
        kernels_loss_type, GPU)
    reblurred_images_x, kernels_loss_x = compute_reblurred_image_and_kernel_loss(
        gt_sharp_x, kernels, masks, gt_kernels, gt_masks, masks_weights,
        kernels_loss_type, GPU)
    reblur_diff_y = (reblurred_images_y[:, :, 1:, :] - blurry_y) ** 2 * masks_weights[:, np.newaxis, 1:, :]
    reblur_diff_x = (reblurred_images_x[:, :, :, 1:] - blurry_x) ** 2 * masks_weights[:, np.newaxis, :, 1:]
    # grad_diff = compute_square_gradient_magnitudes(blur_image, reblurred_images)  * masks_weights[:,np.newaxis,:-1,:-1]
    grad_loss = (reblur_diff_x.sum() + reblur_diff_y.sum()) / N
    kernels_loss = kernels_loss_x + kernels_loss_y

    return grad_loss, kernels_loss


def compute_gradient_loss(blur_image, sharp_image, kernels, masks, GPU=0):

    N = sharp_image.size(0)
    gt_sharp_y = torch.zeros_like(sharp_image)
    gt_sharp_y[:, :, 1:, :] = sharp_image[:, :, 1:, :] - sharp_image[:, :, :-1, :]
    gt_sharp_x = torch.zeros_like(sharp_image)
    gt_sharp_x[:, :, :, 1:] = sharp_image[:, :, :, 1:] - sharp_image[:, :, :, :-1]

    blurry_y = blur_image[:, :, 1:, :] - blur_image[:, :, :-1, :]
    blurry_x = blur_image[:, :, :, 1:] - blur_image[:, :, :, :-1]
    reblurred_images_y = forward_reblur( gt_sharp_y, kernels, masks, GPU)
    reblurred_images_x = forward_reblur( gt_sharp_x, kernels, masks, GPU)
    reblur_diff_y = (reblurred_images_y[:, :, 1:, :] - blurry_y) ** 2
    reblur_diff_x = (reblurred_images_x[:, :, :, 1:] - blurry_x) ** 2
    # grad_diff = compute_square_gradient_magnitudes(blur_image, reblurred_images)  * masks_weights[:,np.newaxis,:-1,:-1]
    grad_loss = (reblur_diff_x + reblur_diff_y).mean()

    return grad_loss

