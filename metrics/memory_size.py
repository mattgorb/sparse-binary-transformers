import numpy as np
import sys
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

def model_size(model,args,quantized=False, as_bits=True):
    """Returns absolute and nonzero model size
    Arguments:
        model {torch.nn.Module} -- Network to compute model size over
    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype
    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """
    params_dict={}
    params_dict['total_params']=0
    params_dict['total_nonzero_params']=0
    params_dict['int8_params']=0
    params_dict['float32_params']=0
    params_dict['binary_params']=0
    params_dict['total_bits']=0

    if quantized:
        #logic for quantized network
        for (k, v) in model.state_dict().items():
            if 'dtype' in k and '_packed' in k:
                continue
            if isinstance(v,tuple)  and '_packed' in k:
                temp=torch.int_repr(v[0]).numpy()
                assert(temp.dtype==np.int8)

                t = np.prod(v[0].shape)
                nz = nonzero(temp)

                bits = dtype2bits[torch.qint8]
                params_dict['total_bits']+=(bits*t)
                params_dict['total_params'] += t
                params_dict['total_nonzero_params'] += nz
                params_dict['int8_params']+=t
            elif not isinstance(v, tuple) and '_packed' in k:

                temp = torch.int_repr(v).numpy()
                assert(temp.dtype==np.uint8)
                t = np.prod(v.shape)
                nz = nonzero(temp)

                bits = dtype2bits[torch.qint8]
                params_dict['total_bits']+=(bits*t)
                params_dict['total_params'] += t
                params_dict['total_nonzero_params'] += nz
                params_dict['int8_params']+=t
    if args.model_type=='Dense':
        #logic for float32 and binary network
        for name, tensor in model.named_parameters():
            t = np.prod(tensor.shape)
            nz = nonzero(tensor.detach().cpu().numpy())
            bits = dtype2bits[tensor.dtype]
            params_dict['total_bits']+=(bits*t)
            params_dict['total_params'] += t
            params_dict['total_nonzero_params'] += nz
            params_dict['float32_params']+=t

    if args.model_type=='SparseBinary':
        #logic for float32 and binary network
        for name, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                #print(name)
                #print(m._get_name())
                b=t=nz=f=0
                print(f'Weights found for {m._get_name()}')
                if m._get_name()=='SubnetLinBiprop'  :
                    tensor=m.weight.detach().cpu().numpy()
                    b = np.prod(tensor.shape)
                    nz = b*m.prune_rate
                    dtype=torch.bool
                    bits = dtype2bits[dtype]
                    params_dict['total_bits'] += (bits * b)
                elif m._get_name()=='SubnetLayerNorm' or m._get_name()=='SubnetEmb':
                    tensor=m.weight.detach().cpu().numpy()
                    t = np.prod(tensor.shape)
                    nz = t*m.prune_rate
                    f=t*m.prune_rate
                    params_dict['total_bits'] += (nz*32+(t-nz)*1)
                else:
                    print(f'Class not found {m._get_name()}')
                    sys.exit()
                params_dict['total_params'] += t
                params_dict['total_nonzero_params'] += nz
                params_dict['float32_params'] += f
                params_dict['binary_params'] += b

    print(params_dict)
    sys.exit()
    return params_dict

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