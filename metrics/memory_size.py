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
            if 'bn' in name:
                if args.batch_norm == False:
                    print(f'{name} continuing...')
                    continue
            if 'norm' in name:
                if args.layer_norm == False:
                    print(f'{name} continuing...')
                    continue
            if 'in_proj' in name:
                continue
            t = np.prod(tensor.shape)
            print(f'Weights found for {name}, {t}')
            #print(t)
            nz = nonzero(tensor.detach().cpu().numpy())
            bits = dtype2bits[tensor.dtype]
            params_dict['total_bits']+=(bits*t)
            params_dict['total_params'] += t
            params_dict['total_nonzero_params'] += nz
            params_dict['float32_params']+=t

    if args.model_type=='SparseBinary' or args.model_type=='Sparse':
        #logic for float32 and binary network
        for name, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                b=t=nz=f=0
                print(f'Weights found for {m._get_name()}')
                if m._get_name()=='SubnetLinBiprop'  :
                    tensor=m.weight.detach().cpu().numpy()
                    b = np.prod(tensor.shape)
                    t=b
                    #print(t)
                    nz = b*m.prune_rate
                    dtype=torch.bool
                    bits = dtype2bits[dtype]
                    params_dict['total_bits'] += int(bits * b)

                elif m._get_name()=='Linear':
                    tensor=m.weight.detach().cpu().numpy()
                    t = np.prod(tensor.shape)
                    b=0
                    #print(t)
                    nz = nonzero(tensor)
                    dtype=torch.float
                    bits = dtype2bits[dtype]
                    params_dict['total_bits'] += int(bits * t)
                elif m._get_name()=='SubnetLayerNorm' or m._get_name()=='SubnetEmb' :
                    if args.layer_norm==False:
                        print('continuing...')
                        continue
                    else:
                        tensor = m.weight.detach().cpu().numpy()
                        b = np.prod(tensor.shape)
                        t = b
                        nz = b * m.prune_rate
                        dtype = torch.bool
                        bits = dtype2bits[dtype]
                        params_dict['total_bits'] += int(bits * b)
                elif m._get_name()=='SubnetBatchNorm' or m._get_name()=='SubnetEmb' :
                    if args.batch_norm==False:
                        print('continuing...')
                        continue
                    else:
                        tensor = m.weight.detach().cpu().numpy()
                        b = np.prod(tensor.shape)
                        t = b
                        nz = b * m.prune_rate
                        dtype = torch.bool
                        bits = dtype2bits[dtype]
                        params_dict['total_bits'] += int(bits * b)
                elif m._get_name()=='BatchNorm1d':
                    if args.batch_norm==False:
                        print('continuing...')
                        continue
                    else:
                        tensor=m.weight.detach().cpu().numpy()
                        t = np.prod(tensor.shape)
                        #t=b
                        b=0
                        #print(t)
                        nz = nonzero(tensor)
                        dtype=torch.float
                        bits = dtype2bits[dtype]
                        params_dict['total_bits'] += int(bits * t)
                elif m._get_name()=='LayerNorm':
                    if args.layer_norm==False:
                        print('continuing...')
                        continue
                    else:
                        tensor=m.weight.detach().cpu().numpy()
                        t = np.prod(tensor.shape)
                        #t=b
                        b=0
                        #print(t)
                        nz = nonzero(tensor)
                        dtype=torch.float
                        bits = dtype2bits[dtype]
                        params_dict['total_bits'] += int(bits * t)
                else:
                    print(f'Class not found {m._get_name()}')
                    #sys.exit()
                print(t)
                params_dict['total_params'] += int(t)
                params_dict['total_nonzero_params'] += int(nz)
                params_dict['float32_params'] += int(f)
                params_dict['binary_params'] += int(b)

    print(params_dict)
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