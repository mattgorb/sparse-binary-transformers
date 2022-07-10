import numpy as np
from torch import nn
from collections import OrderedDict, defaultdict
from models.layers.base_multihead_attention import MultiheadAttention
import torch
from .nonzero import *
from .abstract_flops import *
from .util import get_activations

def _multihead_attention_flops(module, activation):
    return multihead_attention_flops(multihead_attention_module=module,input=activation)

def _conv2d_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    # TODO Add support for dilation and padding size
    return conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)


def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    return dense_flops(module.in_features, module.out_features)


def flops(model, input):
    """Compute Multiply-add FLOPs estimate from model
    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        input {torch.Tensor} -- Input tensor needed for activations
    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """
    FLOP_fn = {
        nn.Conv2d: _conv2d_flops,
        nn.Linear: _linear_flops,
        MultiheadAttention: _multihead_attention_flops
    }

    total_flops = nonzero_flops = 0
    activations = get_activations(model, input)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            #if m.__class__==MultiheadAttention:
            #print(act.shape)
            module_flops = FLOP_fn[m.__class__](m, act)
            module_nonzero_flops=0
            total_flops += module_flops
            # For our operations, all weights are symmetric so we can just
            # do simple rule of three for the estimation
            if m.__class__!=MultiheadAttention:
                w = m.weight.detach().cpu().numpy().copy()
                module_nonzero_flops=module_flops * nonzero(w).sum() / np.prod(w.shape)
                nonzero_flops += module_nonzero_flops
            else:
                print('neeed multiheead attention nonzeero flops')
                print(m)
                act.shape
                sys.exit()
                module_nonzero_flops=0
            print(f'Module: {m}, FLOPs: {module_flops}, nonzeros: {module_nonzero_flops}')
        else:
            print(f'Module not found: {m.__class__}')

    sys.exit()
    return total_flops,nonzero_flops