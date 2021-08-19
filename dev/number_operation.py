import time
import sys
from seal import *

sys.path.append('..')
from examples.seal_helper import *

from matplotlib import pyplot as plt

def main():    
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

    import random
    a = random.uniform(0.0, 255.0)
    b = random.uniform(0.0, 255.0)
    print('Plain result:')
    print(f'A: {a}')
    print(f'B: {b}')
    print(' '*4+f'- Add: {(a+b):.3e}')
    print(' '*4+f'- Multiply: {(a*b):.3e}')

    print('Cipher result:')
    plain_a = ckks_encoder.encode(a, scale)
    plain_b = ckks_encoder.encode(b, scale)
    cipher_a = encryptor.encrypt(plain_a)
    cipher_b = encryptor.encrypt(plain_b)
    
    operations = ['Add', 'Multiply', 'Multiply w/ plain']
    functions = {
            'Add': evaluator.add,
            'Multiply': evaluator.multiply,
            'Multiply w/ plain': evaluator.multiply_plain
            }
    plain_results = {
            'Add': a+b,
            'Multiply': a*b
            }
    args = {
            'Add': (cipher_a, cipher_b), 
            'Multiply': (cipher_a, cipher_b), 
            'Multiply w/ plain': (cipher_a, plain_b)
            }

    for op in operations:
        start = time.time()
        cipher_result = functions[op](*args[op])
        end = time.time()
        plain_result = decryptor.decrypt(cipher_result)
        value = ckks_encoder.decode(plain_result)
        print(' '*4 + f'- {op}(mean, std): {value.mean():.3e}, {value.std():.3e}')
        op = 'Multiply' if 'Multiply' in op else op
        print(' '*8 + f'- Time cost: {((end-start)*1000):.3f} ms')
        print(' '*8 + f'- Loss: {(value.mean()-(plain_results[op])):.3e}')

if __name__=='__main__':
    main()
