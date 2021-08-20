import numpy as np
import torch
from torch import nn
import time
import math

import sys
sys.path.append('..')

from seal import *
from examples.seal_helper import *

FIX_CIPHER_ZERO = False

def HE_naive_conv2d(x, kernel, evaluator, cipher_dummy, padding=0, stride=1):     
    # TODO: how to get cipher_zero, 
    #       check SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT option at
    #               https://github.com/microsoft/SEAL/tree/607801221be3f8499d9d8bd93c06f8b201c98e0b#advanced-cmake-options
    # cipher_zero = evaluator.sub(x[0][0][0][0], x[0][0][0][1])
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
                    result[b][c, w, h] = HE_array_sum(HE_dot_product_plain(window, kernel[c], evaluator), evaluator)
    return result

def HE_dot_product_plain(cipher_x, plain_y, evaluator):
    '''
        Maybe there are some apis to speedup
    '''
    assert(cipher_x.shape == plain_y.shape)

    result = np.empty(cipher_x.shape, dtype='object')
    for c in range(cipher_x.shape[0]):
        for w in range(cipher_x.shape[1]):
            for h in range(cipher_x.shape[2]):
                result[c][w][h] = evaluator.multiply_plain(cipher_x[c][w][h], plain_y[c][w][h])
    return result

def HE_array_sum(cipher_x, evaluator):
    assert(len(cipher_x.shape) == 3)

    result = cipher_x[0][0][0]
    for c in range(cipher_x.shape[0]):
        for w in range(cipher_x.shape[1]):
            for h in range(cipher_x.shape[2]):
                result = evaluator.add(result, cipher_x[c][w][h])
    result = evaluator.sub(result, cipher_x[0][0][0])
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
