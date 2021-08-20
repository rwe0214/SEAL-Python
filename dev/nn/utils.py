import numpy as np
import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../..'))

from seal import *

def get_mean(x: np.ndarray):
    return x.mean()

def _modify_ndarray4d(func, arg_list) -> np.ndarray:
    assert(len(arg_list) > 0)
    assert(len(arg_list[0].shape) == 4)
    x = arg_list[0]
    y = None if len(arg_list) == 1 else arg_list[1]
    result = np.empty(x.shape, dtype='object')

    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for w in range(x.shape[2]):
                for h in range(x.shape[3]):
                    arg_list[0] = x[b][c][w][h]
                    if type(y) is np.ndarray:
                        arg_list[1] = y[b][c][w][h]
                    _args = tuple(arg_list)
                    result[b][c][w][h] = func(*_args)
    return result

def _modify_ndarray3d(func, arg_list) -> np.ndarray:
    assert(len(arg_list) > 0)
    assert(len(arg_list[0].shape) == 3)
    x = arg_list[0]
    y = None if len(arg_list) == 1 else arg_list[1]
    result = np.empty(x.shape, dtype='object')

    for c in range(x.shape[0]):
        for w in range(x.shape[1]):
            for h in range(x.shape[2]):
                    arg_list[0] = x[c][w][h]
                    if type(y) is np.ndarray:
                        arg_list[1] = y[c][w][h]
                    _args = tuple(arg_list)
                    result[c][w][h] = func(*_args)
    return result

'''
This method is the interface of manipulating on ndarray type,
    result = func(arg_list[0], arg_list[1], ...)

arg_list[0] is ndarray type,
arg_list[1] might not be ndarray type,
    if it is ndarray, then would apply `func` on both array element.
    else, apply `func` on arg_list[0] element with arg_list[1]

'''
def modify_ndarray(func, arg_list, dim=4) -> np.ndarray:
    assert(dim==4 or dim==3)
    if dim == 4:
        return _modify_ndarray4d(func, arg_list)
    return _modify_ndarray3d(func, arg_list)

def HE_array_sum(cipher_x: np.ndarray, evaluator):
    assert(len(cipher_x.shape) == 3)

    result = cipher_x[0][0][0]
    for c in range(cipher_x.shape[0]):
        for w in range(cipher_x.shape[1]):
            for h in range(cipher_x.shape[2]):
                result = evaluator.add(result, cipher_x[c][w][h])
    result = evaluator.sub(result, cipher_x[0][0][0])
    return result
