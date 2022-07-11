import torch
import torch.nn.quantized.functional

class LayerNorm(torch.nn.LayerNorm):
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(self, normalized_shape, weight, bias, scale, zero_point, eps=1e-5,
                 elementwise_affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNorm, self).__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
            **factory_kwargs)
        self.weight = weight
        self.bias = bias
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias,
            eps=self.eps, output_scale=self.scale, output_zero_point=self.zero_point)

    def _get_name(self):
        return 'QuantizedLayerNorm'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.normalized_shape, mod.weight, mod.bias, float(scale),
            int(zero_point), mod.eps, mod.elementwise_affine)
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.normalized_shape, mod.weight, mod.bias, float(scale),
            int(zero_point), mod.eps, mod.elementwise_affine)