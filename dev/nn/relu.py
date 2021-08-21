from nn.utils import modify_ndarray, get_mean, set_scale, kScale
from seal import *

import numpy as np
import torch
from torch import nn
import argparse
import time
import math


# reference to https://ieeexplore.ieee.org/document/9378372
coefs_pool = {
        2: [0.563059, 0.5, 0.078047],
        4: [0.119782, 0.5, 0.147298, 0.0, -0.002015]
        }


def _numpy_relu(x: np.ndarray, deg=4) -> np.ndarray:
    assert(len(x.shape) == 4)
    result = np.zeros(x.shape)
    coefs = coefs_pool[deg]
    for e, c in enumerate(coefs):
        result += c*x**e
    return result

def numpy_relu(x, *args):
    return _numpy_relu(x)

def _HE_relu(cipher_x: np.ndarray, evaluator, encoder, scale, relin_keys, deg=4) -> np.ndarray:
    assert(len(cipher_x.shape) == 4)
    
    # encode coefs
    coefs = coefs_pool[deg]
    plain_coefs = []
    for coef in coefs:
        plain_coefs.append(encoder.encode(coef, cipher_x.item(0).parms_id(), scale))

    cipher_x1 = modify_ndarray(evaluator.multiply_plain, [cipher_x, plain_coefs[1]], dim=4)
    cipher_x1 = modify_ndarray(evaluator.relinearize, [cipher_x1, relin_keys], dim=4)
    cipher_x1 = modify_ndarray(evaluator.rescale_to_next, [cipher_x1], dim=4)
    cipher_x1 = modify_ndarray(set_scale, [cipher_x1, kScale], dim=4)
    
    cipher_xns = []
    for i, coef in enumerate(coefs[2:]):
        i += 2
        if coef == 0.0:
            continue
        cipher_xn = modify_ndarray(evaluator.square, [cipher_x], dim=4)
        cipher_xn = modify_ndarray(evaluator.relinearize, [cipher_xn, relin_keys], dim=4)
        cipher_xn = modify_ndarray(evaluator.rescale_to_next, [cipher_xn], dim=4)
    
        if i == 2:
            evaluator.mod_switch_to_inplace(plain_coefs[i], cipher_xn.item(0).parms_id())
            cipher_xn = modify_ndarray(evaluator.multiply_plain, [cipher_xn, plain_coefs[i]], dim=4)
            cipher_xn = modify_ndarray(evaluator.relinearize, [cipher_xn, relin_keys], dim=4)
            cipher_xn = modify_ndarray(evaluator.rescale_to_next, [cipher_xn], dim=4)
        elif i == 3:
            cipher_x_coef = modify_ndarray(evaluator.multiply_plain, [cipher_x, plain_coefs[i]], dim=4)
            cipher_x_coef = modify_ndarray(evaluator.relinearize, [cipher_x_conf, relin_keys], dim=4)
            cipher_x_coef = modify_ndarray(evaluator.rescale_to_next, [cipher_x_coef], dim=4)

            cipher_xn = modify_ndarray(evaluator.multiply, [cipher_xn, cipher_x_coef], dim=4)
            cipher_xn = modify_ndarray(evaluator.relinearize, [cipher_xn, relin_keys], dim=4)
            cipher_xn = modify_ndarray(evaluator.rescale_to_next, [cipher_xn], dim=4)
        elif i == 4:
            cipher_xn = modify_ndarray(evaluator.square, [cipher_xn], dim=4)
            cipher_xn = modify_ndarray(evaluator.relinearize, [cipher_xn, relin_keys], dim=4)
            cipher_xn = modify_ndarray(evaluator.rescale_to_next, [cipher_xn], dim=4)
    
            evaluator.mod_switch_to_inplace(plain_coefs[i], cipher_xn.item(0).parms_id())
            cipher_xn = modify_ndarray(evaluator.multiply_plain, [cipher_xn, plain_coefs[i]], dim=4)
            cipher_xn = modify_ndarray(evaluator.relinearize, [cipher_xn, relin_keys], dim=4)
            cipher_xn = modify_ndarray(evaluator.rescale_to_next, [cipher_xn], dim=4)
        
        cipher_xn = modify_ndarray(set_scale, [cipher_xn, kScale])
        cipher_xns.append(cipher_xn)

    evaluator.mod_switch_to_inplace(plain_coefs[0], cipher_x1.item(0).parms_id())
    result = modify_ndarray(evaluator.add_plain, [cipher_x1, plain_coefs[0]], dim=4)

    for cipher_xn in cipher_xns:
        result = modify_ndarray(evaluator.mod_switch_to, [result, cipher_xn], dim=4)
        result = modify_ndarray(evaluator.add, [result, cipher_xn], dim=4)
    return result

def HE_relu(x, *args, **kwargs):
    return _HE_relu(x, *args, **kwargs)

def test(**kwarg):
    return _test(**kwarg)

def _test(batch=1, channel=1, width=4, height=4, degree=4, bound=10.0):
    _range = bound * 2
    print(f'Input is sample from [{-bound}, {bound}]')
    x = np.random.rand(batch, channel, width, width)*_range - bound

    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, 40, 40, 40, 40, 60]))
    scale = kScale
    context = SEALContext(parms)

    ckks_encoder = CKKSEncoder(context)
    slot_count = ckks_encoder.slot_count()

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    start = time.time()
    plain_x = modify_ndarray(ckks_encoder.encode, [x, scale], dim=4)
    cipher_x = modify_ndarray(encryptor.encrypt, [plain_x], dim=4)
    encrypt_time = time.time()-start
    
    
    start = time.time()
    cipher_result = HE_relu(cipher_x, evaluator, ckks_encoder, scale, relin_keys, deg=degree)
    exec_time = time.time() - start

    start = time.time()
    plain_result = modify_ndarray(decryptor.decrypt, [cipher_result], dim=4)
    value = modify_ndarray(ckks_encoder.decode, [plain_result], dim=4)
    decrypt_time = time.time()-start

    value = modify_ndarray(get_mean, [value], dim=4)

    relu6 = nn.ReLU6()
    true_result = relu6(torch.from_numpy(x)).detach().numpy()
    np_result = numpy_relu(x)

    loss_np_relu6 = np.subtract(np_result, true_result)
    loss_relu6 = np.subtract(value, true_result)
    loss_np = np.subtract(value, np_result)
    print(f'Input size: {x.shape}')
    print(f'Approximate degree: {degree}')
    print(f'Encrypt time: {(encrypt_time*1000.):.3f} ms')
    print(f'HE_relu time cost: {(exec_time*1000.):3f} ms')
    print(f'Decrypt time: {(decrypt_time*1000.):.3f} ms')
    print(f'Loss b/w np_relu & ReLU6 (mean, std): {loss_np_relu6.mean():.3e}, {loss_np_relu6.std():.3e}')
    print(f'Loss b/w HE & ReLU6 (mean, std): {loss_relu6.mean():.3e}, {loss_relu6.std():.3e}')
    print(f'Loss b/w HE & np_relu (mean, std): {loss_np.mean():.3e}, {loss_np.std():.3e}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', help='batch size', default=1, type=int)
    parser.add_argument('--channel', '-c', help='channel', default=1, type=int)
    parser.add_argument('--width', '-w', help='width', default=4, type=int)
    parser.add_argument('--height', '-he', help='height', default=4, type=int)
    parser.add_argument('--degree', '-d', help='approximate degree', default=4, type=int)
    parser.add_argument('--range', '-r', help='input range', default=10.0, type=float)
    args = parser.parse_args()

    test(batch = args.batch,
            channel = args.channel,
            width = args.width,
            height = args.height,
            degree = args.degree,
            bound = args.range)
