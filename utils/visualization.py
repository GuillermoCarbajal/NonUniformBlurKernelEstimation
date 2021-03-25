import torchvision
import numpy as np
import torch
from PIL import Image
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.color import rgb2gray



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor.cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
	return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
	image_pil = None
	if image_numpy.shape[2] == 1:
		image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
		image_pil = Image.fromarray(image_numpy, 'L')
	else:
		image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)

def save_kernels_grid_(blurry_image, kernels, masks, image_name):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    blurry_image = blurry_image.cpu().numpy()
    kernels = kernels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    grid_to_draw = blurry_image.copy()
    for i in range(kernel_size, M - kernel_size // 2, 2 * kernel_size):
        for j in range(kernel_size, N - kernel_size // 2, 2 * kernel_size):
            kernel_ij = np.zeros((kernel_size, kernel_size))
            for k in range(K):
                kernel_ij += masks[k, i, j] * kernels[k]
            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
            grid_to_draw[:, i - kernel_size // 2:i + kernel_size // 2 + 1,
            j - kernel_size // 2:j + kernel_size // 2 + 1] = kernel_ij_norm[None, ::-1, ::-1]

    #self.add_image(image_name, grid_to_draw, step)

    imsave(image_name, img_as_ubyte(grid_to_draw.transpose((1, 2, 0))))


def save_kernels_grid_green(blurry_image, kernels, masks, image_name):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    blurry_image = blurry_image.cpu().numpy()
    kernels = kernels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    grid_to_draw = blurry_image.copy()
    for i in range(kernel_size, M - kernel_size // 2, kernel_size):
        for j in range(kernel_size, N - kernel_size // 2, kernel_size):
            kernel_ij = np.zeros((3, kernel_size, kernel_size))
            for k in range(K):
                kernel_ij[1, :, :] += masks[k, i, j] * kernels[k]
            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
            grid_to_draw[:, i - kernel_size // 2:i + kernel_size // 2 + 1,
            j - kernel_size // 2:j + kernel_size // 2 + 1] = \
                0.5 * grid_to_draw[:, i - kernel_size // 2:i + kernel_size // 2 + 1,
                      j - kernel_size // 2:j + kernel_size // 2 + 1] + 0.5 * kernel_ij_norm[:, ::-1, ::-1]

    imsave(image_name, img_as_ubyte(grid_to_draw.transpose((1, 2, 0))))

def save_kernels_grid(blurry_image, kernels, masks, image_name):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    blurry_image = blurry_image.cpu().numpy()
    kernels = kernels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    grid_to_draw = 0.4*1 + 0.6*rgb2gray(blurry_image.transpose(1,2,0)).copy()
    grid_to_draw = np.repeat(grid_to_draw[None,:,:], 3, axis=0)
    for i in range(kernel_size, M - kernel_size // 2, kernel_size):
        for j in range(kernel_size, N - kernel_size // 2, kernel_size):
            kernel_ij = np.zeros((3, kernel_size, kernel_size))
            for k in range(K):
                kernel_ij[None, :, :] += masks[k, i, j] * kernels[k]
            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
            grid_to_draw[0, i - kernel_size // 2:i + kernel_size // 2 + 1,
            j - kernel_size // 2:j + kernel_size // 2 + 1] = 0.5 * kernel_ij_norm[0, ::-1, ::-1] + (1- kernel_ij_norm[0, ::-1, ::-1]) * grid_to_draw[0, i - kernel_size // 2:i + kernel_size // 2 + 1,
                      j - kernel_size // 2:j + kernel_size // 2 + 1]
            grid_to_draw[1:, i - kernel_size // 2:i + kernel_size // 2 + 1,
            j - kernel_size // 2:j + kernel_size // 2 + 1] = (1- kernel_ij_norm[1:, ::-1, ::-1]) * grid_to_draw[1:, i - kernel_size // 2:i + kernel_size // 2 + 1,
                      j - kernel_size // 2:j + kernel_size // 2 + 1]


    grid_to_draw = np.clip(grid_to_draw, 0, 1)
    imsave(image_name, img_as_ubyte(grid_to_draw.transpose((1, 2, 0))))

   
