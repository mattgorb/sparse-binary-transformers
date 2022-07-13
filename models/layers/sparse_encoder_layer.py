import torch.nn as nn
from models.layers.sparse_type import linear_init, layernorm_init
import torch.nn.functional as F
from models.layers.sparse_multihead_attention import MultiheadAttention

class SparseTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None, args=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,args=args,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = linear_init(d_model, dim_feedforward,args=args, **factory_kwargs)
        self.linear2 = linear_init(dim_feedforward, d_model,args=args, **factory_kwargs)

        #self.norm1 = layernorm_init(d_model, eps=layer_norm_eps,args=args, **factory_kwargs)
        #self.norm2 = layernorm_init(d_model, eps=layer_norm_eps,args=args, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps,**factory_kwargs)


        self.activation = activation


    def forward(self, src, src_mask= None, src_key_padding_mask = None):
        x = src

        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        #x=x + self._sa_block(x, src_mask, src_key_padding_mask)
        #x=x + self._ff_block(x)

        return x

    # self-attention block
    def _sa_block(self, x,attn_mask, key_padding_mask) :
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    # feed forward block
    def _ff_block(self, x) :
        x=self.linear1(x)
        x=self.activation(x)
        #x=self.dropout(x)
        x = self.linear2(x)
        return x