import torch
from torch import nn

import numpy as np

import time

from nn.conv import HE_conv2d
from nn.relu import HE_relu
from nn.utils import *
from seal import *

def relu(x: torch.Tensor, deg=4):
    coefs = { 
            2: [0.563059, 0.5, 0.078047],
            4: [0.119782, 0.5, 0.147298, 0.0, -0.002015]
        }
    result = torch.zeros(x.shape)
    for exp, coef in enumerate(coefs[deg]):
        result += coef*(x**exp)
    return result


batch = 1
in_channel = 1
out_channel = 3
width = 4
height = 4
kernel_size = 3
stride = 1
padding = 1
degree = 4

x = np.random.rand(batch, in_channel, width, height)*255.0
kernel = np.random.rand(out_channel, in_channel, kernel_size, kernel_size)*10.0-5.0
dummy = 1e-10

print(f'==> torch forwarding...')
conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
conv1.weight = nn.Parameter(torch.from_numpy(kernel))
conv_tensor = conv1(torch.from_numpy(x))
# print(conv_tensor)
relu_tensor = relu(conv_tensor, deg=degree)


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
plain_dummy = ckks_encoder.encode(dummy, scale)
plain_kernel = modify_ndarray(ckks_encoder.encode, [kernel, scale], dim=4)
cipher_x = modify_ndarray(encryptor.encrypt, [plain_x], dim=4)
cipher_dummy = encryptor.encrypt(plain_dummy)
end = time.time()

print('==> HE forwarding...')
print(f'Encrypt time: {(end-start):3f} s')
start = time.time()
cipher_conv = HE_conv2d(cipher_x, plain_kernel, evaluator, cipher_dummy, relin_keys, padding=padding, stride=stride)
cipher_relu = HE_relu(cipher_conv, 
                        evaluator,
                        ckks_encoder,
                        scale, 
                        relin_keys,
                        deg=degree)
end = time.time()

plain_conv = modify_ndarray(decryptor.decrypt, [cipher_conv], dim=4)
val_conv = modify_ndarray(ckks_encoder.decode, [plain_conv], dim=4)
val_conv = modify_ndarray(get_mean, [val_conv], dim=4)

plain_relu = modify_ndarray(decryptor.decrypt, [cipher_relu], dim=4)
val_relu = modify_ndarray(ckks_encoder.decode, [plain_relu], dim=4)
val_relu = modify_ndarray(get_mean, [val_relu], dim=4)

true_conv = conv_tensor.detach().numpy()
true_relu = relu_tensor.detach().numpy()

loss_conv = np.subtract(val_conv, true_conv)
loss_relu = np.subtract(val_relu, true_relu)

print(f'Forwarding time: {((end-start)*1000.):.3f} ms')
print(f'Conv Loss (mean, std): {loss_conv.mean():.3e}, {loss_conv.std():.3e}')
print(f'ReLu Loss (mean, std): {loss_relu.mean():.3e}, {loss_relu.std():.3e}')
