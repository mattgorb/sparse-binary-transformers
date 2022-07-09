import torch
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Any, Dict, List, Tuple

import torch.fx
from torch.fx import Interpreter

def print_model_size(mdl,):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt") / 1e6))
    os.remove('tmp.pt')


def memory_profile(model, iterator, device):
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            label, text = batch
            label = label.to(device)
            text = text.to(device)
            break
        with profile(activities=[ProfilerActivity.CPU],
                     profile_memory=True, record_shapes=True) as prof:
            model(text)
        print(prof.key_averages().table())
        #predictions = model(text).squeeze(1)


def _count_relu(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a  ReLU activation.
    The function will count the comparison operation as a FLOP.
    :param node_string: an onnx node defining a ReLU op
    :return: number of FLOPs
    :rtype: `int`
    """
    total_ops = 2 * output.numel()  # also count the comparison
    return total_ops







def _count_maxpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a Max Pooling layer.
    :param node_string: an onnx node defining a max pooling layer
    :return: number of FLOPs
    :rtype: `int`
    """
    out_ops = output.numel()

    kernel_size = [module.kernel_size] * \
        (output.dim() - 2) if isinstance(module.kernel_size, int) else module.kernel_size
    #ops_add = reduce(lambda x, y: x * y, kernel_size) - 1
    #total_ops = ops_add * out_ops
    #return total_ops


def _count_bn(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a Batch Normalisation operation.
    :param node_string: an onnx node defining a batch norm op
    :return: number of FLOPs
    :rtype: `int`
    """
    total_ops = output.numel() * 2
    return total_ops


def _count_linear(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of a GEMM or linear layer.
    :param node_string: an onnx node defining a GEMM or linear layer
    :return: number of FLOPs
    :rtype: `int`
    """
    bias_ops = 0
    if isinstance(module, torch.nn.Linear):
        if module.bias is not None:
            bias_ops = output.shape[-1]
    total_ops = args[0].numel() * output.shape[-1] + bias_ops
    return total_ops


def _count_add_mul(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a summation op.
    :param node_string: an onnx node defining a summation op
    :return: number of FLOPs
    :rtype: `int`
    """
    return output.numel() * len(args)


def _undefined_op(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Default case for undefined or free (in terms of FLOPs) operations
    :param node_string: an onnx node
    :return: always 0
    :rtype: `int`
    """
    return 0


def count_operations(module: Any) -> Any:

    if isinstance(module, torch.nn.ReLU):
        return _count_relu
    elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return _count_bn
    elif isinstance(module, torch.nn.modules.pooling._MaxPoolNd):
        return _count_maxpool
    elif isinstance(module, torch.nn.Linear):
        return _count_linear
    elif 'add' == module or 'mul' == module:
        return _count_add_mul
    elif 'matmul' == module:
        return _count_linear
    else:
        return _undefined_op


class ProfilingInterpreter(Interpreter):
    def __init__(self, mod, custom_ops: Dict[str, Any] = {}):
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        self.custom_ops = custom_ops

        self.flops: Dict[torch.fx.Node, float] = {}
        self.parameters: Dict[torch.fx.Node, float] = {}

    def run_node(self, n: torch.fx.Node) -> Any:
        return_val = super().run_node(n)
        if isinstance(return_val, Tuple):
            self.flops[n] = return_val[1]
            self.parameters[n] = return_val[2]
            return_val = return_val[0]

        return return_val

    def call_module(self, target: 'Target', args, kwargs):
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        output = submod(*args, **kwargs)

        if submod in self.custom_ops:
            count_ops_funct = self.custom_ops[submod]
        else:
            count_ops_funct = count_operations(submod)
        current_ops = count_ops_funct(submod, output, args, kwargs)
        current_params = sum(p.numel() for p in submod.parameters())

        return output, current_ops, current_params

    def call_function(self, target: 'Target', args, kwargs):
        assert not isinstance(target, str)

        # Execute the function and return the result
        output = target(*args, **kwargs)

        current_ops = count_operations(target.__name__)(target, output, args, kwargs)

        return output, current_ops, 0


def count_ops_fx(model: torch.nn.Module,
                 input: torch.Tensor,

                 *args):
    r"""Estimates the number of FLOPs of an :class:`torch.nn.Module`
    :param model: the :class:`torch.nn.Module`
    :param input: a N-d :class:`torch.tensor` containing the input to the model
    :param custom_ops: :class:`dict` containing custom counting functions. The keys represent the name
    of the targeted aten op, while the value a lambda or callback to a function returning the number of ops.
    This can override the ops present in the package.
    :param ignore_layers: :class:`list` containing the name of the modules to be ignored.
    :param print_readable: boolean, if True will print the number of FLOPs. default is True
    :param verbose: boolean, if True will print all the non-zero OPS operations from the network
    :return: number of FLOPs
    :rtype: `int`
    """
    #model, input = same_device(model, input)

    # Place the model in eval mode, required for some models
    model_status = model.training
    model.eval()

    tracer = ProfilingInterpreter(model, custom_ops=None)
    tracer.run(input)

    ops = 0
    all_data = []

    for name, current_ops in tracer.flops.items():
        model_status = model.training

        #if any(name.name == ign_name for ign_name in ignore_layers):
            #continue

        ops += current_ops

        if current_ops :
            all_data.append(['{}'.format(name), current_ops])

    #if print_readable:
        #if verbose:
            #print_table(all_data)
    print("Input size: {0}".format(tuple(input.shape)))
    print("{:,} FLOPs or approx. {:,.2f} GFLOPs".format(ops, ops / 1e+9))

    if model_status:
        model.train()

    return ops, all_data












import numpy as np


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

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

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

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
    multihead_attention_module.__flops__ += int(flops)


CUSTOM_MODULES_MAPPING = {}

import torch.nn as nn
MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,

    nn.InstanceNorm1d: bn_flops_counter_hook,
    nn.InstanceNorm2d: bn_flops_counter_hook,
    nn.InstanceNorm3d: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_flops_counter_hook

import sys
from functools import partial

import torch
import torch.nn as nn

def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def get_flops_pytorch(model, input_res,
                      print_per_layer_stat=True,
                      input_constructor=None, ost=sys.stdout,
                      verbose=False, ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=3,
                      flops_units='GMac',
                      param_units='M'):
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty( *input_res,
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))
        #print(batch.size())
        print(batch)
        print(flops_model)
        _ = flops_model(batch.int())

    flops_count, params_count = flops_model.compute_average_flops_cost()

    if print_per_layer_stat:
        print_model_with_flops(
            flops_model,
            flops_count,
            params_count,
            ost=ost,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision
        )
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}

    return flops_count, params_count


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(model, total_flops, total_params, flops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        return ', '.join([params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Returns current mean flops consumption per image.
    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_variables)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
            module.__ptflops_backup_flops__ = module.__flops__
            module.__ptflops_backup_params__ = module.__params__
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def remove_flops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__'):
            del module.__flops__
            if hasattr(module, '__ptflops_backup_flops__'):
                module.__flops__ = module.__ptflops_backup_flops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__ptflops_backup_params__'):
                module.__params__ = module.__ptflops_backup_params__

def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              flops_units=None, param_units=None,
                              output_precision=2):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)

    if backend == 'pytorch':
        flops_count, params_count = get_flops_pytorch(model, input_res,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      flops_units=flops_units,
                                                      param_units=param_units)
    else:
        raise ValueError('Wrong backend name')

    if as_strings:
        flops_string = flops_to_string(
            flops_count,
            units=flops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return flops_string, params_string

    return flops_count, params_count
