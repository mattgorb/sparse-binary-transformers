import numpy as np
import torch


def nonzero(tensor):
    """Returns absolute number of values different from 0
    Arguments:
        tensor {numpy.ndarray} -- Array to compute over
    Returns:
        int -- Number of nonzero elements
    """
    return np.sum(tensor != 0.0)


# https://pytorch.org/docs/stable/tensor_attributes.html
dtype2bits = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
    torch.qint8:8
}

from numpy import dtype
# https://pytorch.org/docs/stable/tensor_attributes.html
dtype2bits_np = {
    'float32': 32,
    'float': 32,
    'float64': 64,
    'double': 64,
    'float16': 16,
    'half': 16,
    'uint8': 8,
    'int8': 8,
    'int16': 16,
    'short': 16,
    'int32': 32,
    'int': 32,
    'int64': 64,
    'long': 64,
    'bool': 1,
}