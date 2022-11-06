import torch.nn as nn
from models.layers.sparse_type import linear_init,layernorm_init, batchnorm_init
import torch.nn.functional as F




class SparseTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None, args=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseTransformerEncoderLayer, self).__init__()

        self.args=args

        if self.args.attention=='SparseTopP':

            from models.layers.sparsetopp_multihead_attention import SparseTopPMultiheadAttention
            self.self_attn = SparseTopPMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,args=args,
                                                **factory_kwargs)
        elif self.args.attention=='Sparse':

            from models.layers.sparse_multihead_attention import SparseMultiheadAttention
            self.self_attn = SparseMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,args=args,
                                                **factory_kwargs)

        else:
            from models.layers.base_multihead_attention import MultiheadAttention
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,args=self.args,
                                                **factory_kwargs)

        self.dropout=dropout
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

        self.linear1 = linear_init(d_model, dim_feedforward,args=args, **factory_kwargs)
        self.linear2 = linear_init(dim_feedforward, d_model,args=args, **factory_kwargs)

        self.norm1 = layernorm_init(self.args.window_size, eps=layer_norm_eps,args=args, **factory_kwargs)
        self.norm2 = layernorm_init(self.args.window_size, eps=layer_norm_eps,args=args, **factory_kwargs)

        #self.norm1 = nn.LayerNorm(self.args.window_size, eps=layer_norm_eps, **factory_kwargs)
        #self.norm2 = nn.LayerNorm(self.args.window_size, eps=layer_norm_eps, **factory_kwargs)


        #self.bn1 = batchnorm_init(d_model, eps=layer_norm_eps,args=args, **factory_kwargs)
        #self.bn2 = batchnorm_init(d_model, eps=layer_norm_eps, args=args,**factory_kwargs)

        self.bn1 = nn.BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.bn2 = nn.BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.activation = activation


    def forward(self, src, src_mask= None, src_key_padding_mask = None):
        '''x = src

        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        #x=x + self._sa_block(x, src_mask, src_key_padding_mask)
        #x=x + self._ff_block(x)

        return x'''

        src2, attention = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        if self.args.layer_norm:
            src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
            src = self.norm1(src)
            src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        if self.args.batch_norm:
            src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
            src=self.bn1(src)
            src = src.permute(2, 0, 1)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)  # (seq_len, batch_size, d_model)

        if self.args.layer_norm:
            src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
            src = self.norm2(src)
            src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        if self.args.batch_norm:
            src = src.permute(1, 2, 0)
            src=self.bn2(src)
            src = src.permute(2, 0, 1)

        return src, attention

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