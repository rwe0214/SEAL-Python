from .utils import HE_array_sum, modify_ndarray, get_mean, set_scale, kScale
from seal import *

import numpy as np
import torch
from torch import nn
import time
import math


# Use cipher dummy to approximate cipher zero
FIX_CIPHER_ZERO = True

def HE_naive_conv2d(x, kernel, evaluator, cipher_dummy, relin_keys, padding=0, stride=1):     
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
                    result[b][c, w, h] = modify_ndarray(evaluator.multiply_plain, [window, kernel[c]], dim=3)
                    result[b][c, w, h] = modify_ndarray(evaluator.relinearize, [result[b][c, w, h], relin_keys], dim=3)
                    result[b][c, w, h] = modify_ndarray(evaluator.rescale_to_next, [result[b][c, w, h]], dim=3)
                    result[b][c][w][h] = HE_array_sum(result[b][c, w, h], evaluator)
                    # TODO: BatchNorm
                    # (x - a.running_mean) * (1 / a.running_var) * weight + bias
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


import time
import argparse

OUTPUT_VISIBLE = False

def test(name, test_conv2d, **kwarg):
    return _test(name, test_conv2d, **kwarg)

def _test(name, test_conv2d, batch=1, in_channel=1, out_channel=1, width=4, height=4, kernel_size=3, padding=1, stride=1):
    x = np.random.rand(batch, in_channel, width, height)*255.0
    kernel = np.random.rand(out_channel, in_channel, kernel_size, kernel_size)*255.0
    dummy_value = 1e-10

    torch_conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    torch_conv2d.weight = nn.Parameter(torch.from_numpy(kernel))

    print(f'Input size: {x.shape}')
    tensor_x = torch.from_numpy(x)
    print(f'==> torch.nn.Conv2d forwarding...')
    start = time.time()
    torch_conv2d_result = torch_conv2d(tensor_x)
    end = time.time()
    if OUTPUT_VISIBLE:
        print(f'Output: \n{torch_conv2d_result.detach().numpy()}')
    print(f'Output size: {torch_conv2d_result.shape}')
    print(f'torch.nn.Conv2d time cost: {((end-start)*1000.):3f} ms\n')

    evaluator = None
    relin_keys = None
    print(f'==> {name} forwarding...')
    if 'HE' in name:
        parms = EncryptionParameters(scheme_type.ckks)
        poly_modulus_degree = 16384
        parms.set_poly_modulus_degree(poly_modulus_degree)
        parms.set_coeff_modulus(CoeffModulus.Create(
            poly_modulus_degree, [60, 40, 40, 40, 40, 60]))
        scale = 2.0**40
        context = SEALContext(parms)
        # print_parameters(context)

        ckks_encoder = CKKSEncoder(context)
        slot_count = ckks_encoder.slot_count()
        # print(f'Number of slots: {slot_count}')

        keygen = KeyGenerator(context)
        public_key = keygen.create_public_key()
        secret_key = keygen.secret_key()
        relin_keys = keygen.create_relin_keys()

        encryptor = Encryptor(context, public_key)
        evaluator = Evaluator(context)
        decryptor = Decryptor(context, secret_key)

        start = time.time()
        plain_x = modify_ndarray(ckks_encoder.encode, [x, scale], dim=4)
        plain_dummy = ckks_encoder.encode(dummy_value, scale)
        plain_kernel = modify_ndarray(ckks_encoder.encode, [kernel, scale], dim=4)
        cipher_x = modify_ndarray(encryptor.encrypt, [plain_x], dim=4)
        cipher_dummy = encryptor.encrypt(plain_dummy)
        # cipher_kernel = modify_ndarray(encryptor.encrypt, [plain_kernel])
        end = time.time()
        print(f'Encrypt time: {(end-start):3f} s')
        x = cipher_x
        kernel = plain_kernel
        dummy_value = cipher_dummy
    
    start = time.time()
    test_conv2d_result = test_conv2d(x, kernel, evaluator, dummy_value, relin_keys, padding=padding, stride=stride)
    end = time.time()
    print(f'{name} time cost: {((end-start)*1000.):3f} ms')
    
    if 'HE' in name:
        plain_result = modify_ndarray(decryptor.decrypt, [test_conv2d_result], dim=4)
        test_conv2d_result = modify_ndarray(ckks_encoder.decode, [plain_result], dim=4)
        end = time.time()
        test_conv2d_result = modify_ndarray(get_mean, [test_conv2d_result], dim=4)
        print(f'Decrypt time: {(end-start):3f} s')

    diff = np.subtract(test_conv2d_result, torch_conv2d_result.detach().numpy())
    print(f'Output size: {test_conv2d_result.shape}')
    print(f'Difference (mean, std): {(diff.mean()):.2e}, {(diff.std()):.2e}')
    if OUTPUT_VISIBLE:
        print(f'Output: \n{test_conv2d_result}')

    if 'HE' in name:
        if not FIX_CIPHER_ZERO:
            print('[Warning] The problem of cipher zero did not be solved!', file=sys.stderr)
            print('          It would increases accuracy loss!', file=sys.stderr)
            exit(1)

if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', help='batch size', default=1, type=int)
    parser.add_argument('--in_channel', '-ic', help='in channel', default=1, type=int)
    parser.add_argument('--out_channel', '-oc', help='out channel', default=3, type=int)
    parser.add_argument('--width', '-w', help='width', default=4, type=int)
    parser.add_argument('--height', '-he', help='height', default=4, type=int)
    parser.add_argument('--kernel_size', '-k', help='kernel size', default=3, type=int)
    parser.add_argument('--padding', '-p', help='padding size', default=1, type=int)
    parser.add_argument('--stride', '-s', help='stride', default=1, type=int)
    args = parser.parse_args()

    libs = ['numpy', 'HE']
    test_methods = {
            'numpy': {
                'name': 'numpy_conv2d',
                'body': numpy_conv2d
                },
            'HE': {
                'name': 'HE_conv2d',
                'body': HE_conv2d
                }
            }    
    for lib in libs:
        print('-'*18+lib+'-'*18)
        test(test_methods[lib]['name'], test_methods[lib]['body'], 
                batch=args.batch, 
                in_channel=args.in_channel, 
                out_channel=args.out_channel, 
                width=args.width, 
                height=args.height, 
                kernel_size=args.kernel_size, 
                padding=args.padding, 
                stride=args.stride)




