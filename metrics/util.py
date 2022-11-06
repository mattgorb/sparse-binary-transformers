import numpy as np
from torch import nn
from collections import OrderedDict, defaultdict

import torch
from .nonzero import *
from .abstract_flops import *

def hook_applyfn2(hook, model, forward=False, backward=False):
    """
    [description]
    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]
    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})
    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook,))
            if backward:
                hooks.append(module.register_backward_hook(hook))



    return register_hook, hooks

def get_activations(model, input):

    activations = OrderedDict()


    def register_hook(module):


        def store_activations(module, input, output,):
            if isinstance(module, nn.ReLU):
                # TODO ResNet18 implementation reuses a
                # single ReLU layer?
                return

            name=None
            for key, val in names_dict.items():

                if val == module:
                    name = key
            if name==None:
                print('cant find module name')
                sys.exit(0)
            assert module not in activations, \
                f"{module} already in activations"
            # TODO [0] means first input, not all models have a single input

            if "linear_Q" in name or "linear_K" in name or "linear_V" in name:

                activations[module] = (input[0].detach().cpu().numpy().copy(),
                                       output[0].detach().cpu().numpy().copy(),name)
            else:
                activations[module] = (input[0].detach().cpu().numpy().copy(),
                                       output[0].detach().cpu().numpy().copy(),)
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):

            hooks.append(module.register_forward_hook(store_activations,))

    hooks=[]
    names_dict={}
    for n,m in model.named_modules():

        names_dict[n]=m

    #fn, hooks = hook_applyfn(register_hook, model, forward=True)


    #sys.exit()

    model.apply(register_hook)



    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()



    return activations