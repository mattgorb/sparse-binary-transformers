import numpy as np
import torch
from models.layers.sparse_type import *
import torch.nn.functional as nnF
import math


def subnet_norm_flops(module, input,):
    batch_flops = np.prod(input[0].shape)

    #if (getattr(module, 'affine', False)
     #       or getattr(module, 'elementwise_affine', False)):
      #  batch_flops *= 2
    return batch_flops, batch_flops*module.prune_rate




def norm_flops(module, input,):
    batch_flops = np.prod(input[0].shape)


    if (getattr(module, 'affine', False)
            or getattr(module, 'elementwise_affine', False)):
        batch_flops *= 2
    return batch_flops


def posenc_flops(module, input,):

    flops = input.shape[0]*input.shape[2]

    return flops

def multihead_attention_nonzero_flops(multihead_attention_module,lin_q,lin_k,lin_v):
    #https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/pytorch_ops.py
    flops = 0

    q, k, v = lin_q,lin_k,lin_v

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]


    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    #print(flops)

    # Initial projections
    '''flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )'''

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads


    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    return flops


def multihead_attention_flops(multihead_attention_module, input,):
    #https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/pytorch_ops.py
    #https://github.com/google-research/electra/blob/master/flops_computation.py
    #https: // www.stat.cmu.edu / ~ryantibs / convexopt - F18 / scribes / Lecture_19.pdf
    flops = 0

    q, k, v = input, input, input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]


    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    # Q scaling
    flops += qlen * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    #head_flops = (
        #(qlen * klen * qk_head_dim)  # QK^T
        #+ (qlen * klen)  # softmax
        #+ (qlen * klen * v_head_dim)  # AV
    #)

    #https://math.stackexchange.com/questions/3512976/proof-of-of-flops-in-matrix-multiplication
    print(f'attention has source mask? {multihead_attention_module.args.has_src_mask}')


    if not multihead_attention_module.args.has_src_mask:
        head_flops=0
        head_flops+=(qlen * klen * (qk_head_dim))  # QK^T
        head_flops += (qlen * klen)  # softmax
        head_flops += (qlen * klen * (v_head_dim))  # AV #
        flops += num_heads * head_flops
        #flops *= batch_size
    else:

        head_flops=0
        #head_flops+=((qlen-1) * (qk_head_dim)+(klen-1) * (qk_head_dim))  # QK^T (n-1)(2p-1)+(m-1)(2p-1)
        #head_flops += (qlen + klen)  # softmax
        #head_flops +=((qlen-1) * (v_head_dim-1)+(klen-1) * (v_head_dim))  # AV

        head_flops+=((qlen-1) * (qk_head_dim))  # QK^T (n-1)(2p-1)+(m-1)(2p-1)
        head_flops += (qlen)  # softmax
        head_flops +=((qlen-1) * (qk_head_dim))  # AV
        flops += num_heads * head_flops
        #flops *= batch_size

    return flops










def sparse_multihead_attention_flops(multihead_attention_module, input,):
    #https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/pytorch_ops.py
    flops = 0
    bops=0

    q, k, v = input, input, input


    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]


    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim


    # Q scaling
    flops += qlen * qdim * multihead_attention_module.attention_prune_rate


    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads


    qlen_prate=int(qlen* multihead_attention_module.attention_prune_rate)
    klen_prate=int(klen* multihead_attention_module.attention_prune_rate)


    head_flops1=qlen_prate*qlen_prate*qk_head_dim# QK^T
    head_flops2=(qlen * klen)# softmax
    head_flops3=qlen*klen_prate*v_head_dim# AV

    head_flops=head_flops1+head_flops2+head_flops3

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    #flops += qlen * vdim * (vdim + 1)

    #flops *= batch_size
    return int(flops), int(flops)



def dense_flops(in_neurons, out_neurons,args, act):
    """Compute the number of multiply-adds used by a Dense (Linear) layer"""
    if len(act)==3:
        return in_neurons * out_neurons*args.window_size
    else:
        return in_neurons * out_neurons

def subnet_dense_flops(module, args,act):
    """Compute the number of multiply-adds used by a Dense (Linear) layer"""

    if len(act)==3:
        return module.in_features*module.out_features, int(module.prune_rate*module.in_features*module.out_features*args.window_size)
    else:
        return module.in_features*module.out_features, int(module.prune_rate*module.in_features*module.out_features)



def conv2d_flops(in_channels, out_channels, input_shape, kernel_shape,
                 padding='same', strides=1, dilation=1):
    """Compute the number of multiply-adds used by a Conv2D layer
    Args:
        in_channels (int): The number of channels in the layer's input
        out_channels (int): The number of channels in the layer's output
        input_shape (int, int): The spatial shape of the rank-3 input tensor
        kernel_shape (int, int): The spatial shape of the rank-4 kernel
        padding ({'same', 'valid'}): The padding used by the convolution
        strides (int) or (int, int): The spatial stride of the convolution;
            two numbers may be specified if it's different for the x and y axes
        dilation (int): Must be 1 for now.
    Returns:
        int: The number of multiply-adds a direct convolution would require
        (i.e., no FFT, no Winograd, etc)
    >>> c_in, c_out = 10, 10
    >>> in_shape = (4, 5)
    >>> filt_shape = (3, 2)
    >>> # valid padding
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, padding='valid')
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 4))
    True
    >>> # same padding, no stride
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, padding='same')
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * np.prod(in_shape))
    True
    >>> # valid padding, stride > 1
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, \
                       padding='valid', strides=(1, 2))
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 2))
    True
    >>> # same padding, stride > 1
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, \
                           padding='same', strides=2)
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 3))
    True
    """
    # validate + sanitize input
    assert in_channels > 0
    assert out_channels > 0
    assert len(input_shape) == 2
    assert len(kernel_shape) == 2
    padding = padding.lower()
    assert padding in ('same', 'valid', 'zeros'), "Padding must be one of same|valid|zeros"
    try:
        strides = tuple(strides)
    except TypeError:
        # if one number provided, make it a 2-tuple
        strides = (strides, strides)
    assert dilation == 1 or all(d == 1 for d in dilation), "Dilation > 1 is not supported"

    # compute output spatial shape
    # based on TF computations https://stackoverflow.com/a/37674568
    if padding in ['same', 'zeros']:
        out_nrows = np.ceil(float(input_shape[0]) / strides[0])
        out_ncols = np.ceil(float(input_shape[1]) / strides[1])
    else:  # padding == 'valid'
        out_nrows = np.ceil((input_shape[0] - kernel_shape[0] + 1) / strides[0])  # noqa
        out_ncols = np.ceil((input_shape[1] - kernel_shape[1] + 1) / strides[1])  # noqa
    output_shape = (int(out_nrows), int(out_ncols))

    # work to compute one output spatial position
    nflops = in_channels * out_channels * int(np.prod(kernel_shape))

    # total work = work per output position * number of output positions
    return nflops * int(np.prod(output_shape))


if __name__ == '__main__':
    import doctest
    doctest.testmod()



'''
import torch
vals=[]
#for i in range(2000):
q=torch.randn(200,64)
k=torch.randn(64,200)
prune_size = int(torch.flatten(q).size()[0] * .5)

q_sort_val, q_sort_ind = torch.sort(q.abs().flatten(), descending=True)
q.flatten()[q_sort_ind[prune_size:]] = 0
q.flatten()[q_sort_ind[:prune_size]] = 1

k_sort_val, k_sort_ind = torch.sort(k.abs().flatten(), descending=True)
k.flatten()[k_sort_ind[prune_size:]] = 0
k.flatten()[k_sort_ind[:prune_size]] = 1

output=torch.einsum('ij,jk->ikj', q,k)
print(output)
#print(torch.sum(output))
#print(torch.sum(torch.mm(q,k)))
#vals.append(torch.sum(output))

print(min(vals))
print(max(vals))
import torch
q = torch.zeros(20, 10)
k = torch.zeros(10, 20)
q[:,0]=1
k[0,:]=1
print(torch.sum(q))
print(torch.sum(k))
print(torch.sum(torch.mm(q,k)))
#20 20
'''