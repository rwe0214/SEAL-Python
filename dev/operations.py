import time
import sys
import numpy as np

sys.path.append('..')
from seal import *
from examples.seal_helper import *

from matplotlib import pyplot as plt
import random

def main():    
    vec_size = 3
    a = random.uniform(0.0, 255.0)
    b = random.uniform(0.0, 255.0)
    a_vec = list(np.random.uniform(0.0, 255.0, vec_size))
    b_vec = list(np.random.uniform(0.0, 255.0, vec_size))

    print(f'A: {a}')
    print(f'B: {b}')
    print(f'A_vecor: {a_vec}')
    print(f'B_vector: {b_vec}')

    print('='*12+' Calculations '+'='*12)
    print('True result:')
    print(' '*4+f'- Add: {(a+b)}')
    print(' '*4+f'- Multiply: {(a*b)}')
    print(' '*4+f'- Add_many: {[a+b for a, b in zip(a_vec, b_vec)]}')
    print(' '*4+f'- Multiply_many: {(np.multiply(a_vec, b_vec))}')

    print('='*12+' HE calculations '+'='*12)
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, 40, 40, 40, 40, 60]))
    scale = 2.0**40
    context = SEALContext(parms)
    print_parameters(context)

    ckks_encoder = CKKSEncoder(context)
    slot_count = ckks_encoder.slot_count()
    print(f'Number of slots: {slot_count}')

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)


    plain_a = ckks_encoder.encode(a, scale)
    plain_b = ckks_encoder.encode(b, scale)
    cipher_a = encryptor.encrypt(plain_a)
    cipher_b = encryptor.encrypt(plain_b)
    
    plain_a_vec = ckks_encoder.encode(a_vec, scale)
    plain_b_vec = ckks_encoder.encode(b_vec, scale)
    cipher_a_vec = encryptor.encrypt(plain_a_vec)
    cipher_b_vec = encryptor.encrypt(plain_b_vec)
    
    operations = ['Add', 
            'Add_many', 
            'Multiply',
            'Multiply_many', 
            'Multiply w/ plain', 
            'Multiply w/ plain many']
    functions = {
            'Add': evaluator.add,
            'Multiply': evaluator.multiply,
            'Add_many': evaluator.add,
            'Multiply_many': evaluator.multiply,
            'Multiply w/ plain': evaluator.multiply_plain,
            'Multiply w/ plain many': evaluator.multiply_plain
            }
    true_results = {
            'Add': a+b,
            'Multiply': a*b,
            'Add_many': [a+b for a, b in zip(a_vec, b_vec)],
            'Multiply_many': [a*b for a, b in zip(a_vec, b_vec)],
            'Multiply w/ plain': a*b,
            'Multiply w/ plain many': [a*b for a, b in zip(a_vec, b_vec)]
            }
    args = {
            'Add': (cipher_a, cipher_b), 
            'Multiply': (cipher_a, cipher_b), 
            'Add_many': (cipher_a_vec, cipher_b_vec), 
            'Multiply_many': (cipher_a_vec, cipher_b_vec), 
            'Multiply w/ plain': (cipher_a, plain_b),
            'Multiply w/ plain many': (cipher_a_vec, plain_b_vec), 
            }

    print('Cipher result:')
    for op in operations:
        start = time.time()
        cipher_result = functions[op](*args[op])
        end = time.time()
        plain_result = decryptor.decrypt(cipher_result)
        value = ckks_encoder.decode(plain_result)
        print(' '*4 + f'- {op}: {value[:vec_size] if "many" in op else value.mean()}')
        print(' '*8 + f'- Time cost: {((end-start)*1000):.3f} ms')
        if 'many' in op:
            loss = np.array([v-p for v, p in zip(value[:vec_size], true_results[op][:vec_size])])
            print(' '*8 + f'- Loss: {loss.mean():.3e}, {loss.std():.3e}')
        else:
            loss = value.mean()-(true_results[op])
            print(' '*8 + f'- Loss: {loss:.3e}')

if __name__=='__main__':
    main()
