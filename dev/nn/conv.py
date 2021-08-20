import numpy as np
import torch
from torch import nn
import time
import math

from utils import HE_array_sum, modify_ndarray

# from seal import *

# Use cipher dummy to approximate cipher zero
FIX_CIPHER_ZERO = True

def HE_naive_conv2d(x, kernel, evaluator, cipher_dummy, padding=0, stride=1):     
    # cipher_zero, check SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT option in
    #     https://github.com/microsoft/SEAL/tree/607801221be3f8499d9d8bd93c06f8b201c98e0b#advanced-cmake-options 
    #     for detail.
    # Using cipher dummy value to approximate.
    cipher_zero = cipher_dummy

    npad = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    pad_x = np.pad(x, npad, mode='constant', constant_values=cipher_zero)
    w_out = math.floor((x.shape[2]+2*padding-(kernel.shape[2]-1)-1)/stride+1)
    h_out = math.floor((x.shape[3]+2*padding-(kernel.shape[3]-1)-1)/stride+1)
    output_shape = (x.shape[0], kernel.shape[0], w_out, h_out)
    result = np.full(output_shape, cipher_zero, dtype='object')

    for b in range(result.shape[0]):
        for c in range(result.shape[1]):
            for w in range(result.shape[2]):
                for h in range(result.shape[3]):
                    window = pad_x[b][:, w: w + kernel.shape[2], h: h + kernel.shape[3]]
                    # result[b][c, w, h] = HE_array_sum(HE_dot_product_plain(window, kernel[c], evaluator), evaluator)
                    result[b][c, w, h] = HE_array_sum(modify_ndarray(evaluator.multiply_plain, [window, kernel[c]], dim=3), evaluator)
    return result

def HE_conv2d(x, kernel, *args, **kwarg):
    return HE_naive_conv2d(x, kernel, *args, **kwarg)

def naive_conv2d(x, kernel, padding=0, stride=1):
    '''
        All of parameters are type of ndarray
    '''
    npad = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    pad_x = np.pad(x, npad, mode='constant', constant_values=0)
    w_out = math.floor((x.shape[2]+2*padding-(kernel.shape[2]-1)-1)/stride+1)
    h_out = math.floor((x.shape[3]+2*padding-(kernel.shape[3]-1)-1)/stride+1)
    output_shape = (x.shape[0], kernel.shape[0], w_out, h_out)
    result = np.zeros(output_shape)

    for b in range(result.shape[0]):
        for c in range(result.shape[1]):
            for w in range(result.shape[2]):
                for h in range(result.shape[3]):
                    window = pad_x[b][:, w: w + kernel.shape[2], h: h + kernel.shape[3]]
                    result[b][c][w][h] = np.sum(np.multiply(kernel[c], window))
    return result

def numpy_conv2d(x, kernel, *args, **kwarg):
    return naive_conv2d(x, kernel, **kwarg)
