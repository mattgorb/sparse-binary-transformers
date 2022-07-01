import copy
import torch
from torch import Tensor
from torch.nn import ModuleList
import torch.nn as nn

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class SparseTransformerEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SparseTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask = None, src_key_padding_mask = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
