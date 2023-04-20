import numpy as np
from torch import nn
from collections import OrderedDict, defaultdict
from models.layers.base_multihead_attention import MultiheadAttention
import torch
from .nonzero import *
from .abstract_flops import *
from .util import get_activations
from models.layers.positional_encoder import PositionalEncoding
from models.layers.sparse_type import SubnetLayerNorm,SubnetLinBiprop
from models.layers.sparsetopp_multihead_attention import SparseTopPMultiheadAttention
from models.layers.sparse_multihead_attention import SparseMultiheadAttention

def _multihead_attention_flops(module, activation, args):
    return multihead_attention_flops(multihead_attention_module=module,input=activation[0])



def _linear_flops(module, activation, args):
    # Auxiliary func to use abstract flop computation

    return dense_flops(module.in_features, module.out_features,args, activation)

def _subnet_linear_flops(module, activation, args):
    # Auxiliary func to use abstract flop computation
    return subnet_dense_flops(module,args,activation )



def _layernorm_flops(module, activation, args):
    return norm_flops(module,activation[0])

def _posenc_flops(module, activation, args):
    return posenc_flops(module,activation[0])





def _subnet_layernorm_flops(module, activation, args):
    return subnet_norm_flops(module,activation[0])


def _subnet_multihead_flops(module, activation, args):
    return sparse_multihead_attention_flops(module,activation[0])

def flops(model, input, args):
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

        nn.Linear: _linear_flops,
        MultiheadAttention: _multihead_attention_flops,
        nn.LayerNorm: _layernorm_flops,
        PositionalEncoding: _posenc_flops,
        SparseMultiheadAttention: _multihead_attention_flops
    }

    sbt_fn = {
        SubnetLinBiprop: _subnet_linear_flops,
        SubnetLayerNorm: _subnet_layernorm_flops,
        SubnetBatchNorm: _subnet_layernorm_flops,
        SparseTopPMultiheadAttention: _subnet_multihead_flops
    }

    flops_dict={}
    flops_dict['total_flops']=0
    #flops_dict['total_bops']=0

    activations = get_activations(model, input)


    modules_not_found=[]


    for m, act in activations.items():

        if m.__class__ in FLOP_fn:
            module_flops = FLOP_fn[m.__class__](m, act, args)
            flops_dict['total_flops'] += module_flops
            print(f'Module: {m._get_name()}, FLOPs: {module_flops:,}')#, nonzero FLOPS: {module_nonzero_flops}')
        elif m.__class__ in sbt_fn:
            module_bops, module_nonzero_flops = sbt_fn[m.__class__](m, act, args)
            flops_dict['total_flops'] += module_nonzero_flops
            #flops_dict['total_bops'] += module_bops
            print(act)
            print(f'Module: {m._get_name()},   Nonzero FLOPs: {module_nonzero_flops}')  # , nonzero FLOPS: {module_nonzero_flops}')

        else:
            #print(f'Module not found: {m._get_name()}')
            modules_not_found.append(m._get_name())

    return flops_dict,set(modules_not_found)