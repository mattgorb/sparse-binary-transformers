import numpy as np

from .nonzero import dtype2bits, nonzero,dtype2bits_np
from .util import get_activations
import torch
import os

def state_dict_size(mdl,):
    torch.save(mdl.state_dict(), "tmp.pt")
    #print("%.2f MB" %(os.path.getsize("tmp.pt") / 1e6))
    bit_size=os.path.getsize("tmp.pt")*8
    os.remove('tmp.pt')
    #print(bit_size)
    return bit_size

def model_size(model, as_bits=True):
    """Returns absolute and nonzero model size
    Arguments:
        model {torch.nn.Module} -- Network to compute model size over
    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype
    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """
    for (k, v) in model.state_dict().items():
        if 'dtype' in k:
            print("HERE2")
            print(k)
            print(v)
            continue
        if isinstance(v,tuple):
            print('here')
            print(k)
            print(v)
            #print(k,v)
            continue
        print(k, v.size())
    #return
    #print(model)

    total_params = 0
    nonzero_params = 0
    for name, tensor in model.named_parameters():
        print(name)
        #print(tensor.shape)
        t = np.prod(tensor.shape)
        nz = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            print(tensor.dtype)
            bits = dtype2bits[tensor.dtype]
            t *= bits
            nz *= bits
        total_params += t
        nonzero_params += nz
    return int(total_params), int(nonzero_params)

def memory(model, input, as_bits=True):
    """Compute memory size estimate
    Note that this is computed for training purposes, since
    all input activations to parametric are accounted for.
    For inference time you can free memory as you go but with
    residual connections you are forced to remember some, thus
    this is left to implement (TODO)
    The input is required in order to materialize activations
    for dimension independent layers. E.g. Conv layers work
    for any height width.
    Arguments:
        model {torch.nn.Module} -- [description]
        input {torch.Tensor} --
    Keyword Arguments:
        as_bits {bool} -- [description] (default: {False})
    Returns:
        tuple:
         - int -- Estimated memory needed for the full model
         - int -- Estimated memory needed for nonzero activations
    """
    batch_size = input.size(0)
    total_memory = nonzero_memory = np.prod(input.shape)

    activations = get_activations(model, input)

    # TODO only count parametric layers
    # Input activations are the ones we need for backprop
    input_activations = [i for _, (i, o) in activations.items()]

    for act in input_activations:
        t = np.prod(act.shape)
        nz = nonzero(act)
        if as_bits:
            bits = dtype2bits_np[str(act.dtype)]
            t *= bits
            nz *= bits
        total_memory += t
        nonzero_memory += nz

    return total_memory/batch_size, nonzero_memory/batch_size