import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.positional_encoder import PositionalEncoding, LearnablePositionalEncoding
from models.layers.sparse_type import linear_init,emb_init

class TSSparseTransformerModel(nn.Module):

    def __init__(self, input_dim, ninp, nhead, nhid, nlayers=6, args=None):
        super(TSSparseTransformerModel, self).__init__()
        try:
            from models.layers.sparse_encoder import SparseTransformerEncoder
            from models.layers.sparse_encoder_layer import  SparseTransformerEncoderLayer
        except:
            raise ImportError("Had trouble importing transformer modules. ")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pad_mask = None
        self.args=args
        self.pos_encoder = LearnablePositionalEncoding(ninp, )
        encoder_layers = SparseTransformerEncoderLayer(ninp, nhead, nhid, args=self.args)
        self.transformer_encoder = SparseTransformerEncoder(encoder_layers, nlayers)

        self.embedding = linear_init(input_dim, ninp,args=args,)
        self.ninp = ninp
        self.decoder = linear_init(ninp, input_dim,bias=False,args=args, )

        #self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_src_mask=False, has_pad_mask=False):
        if has_src_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask

        else:
            self.src_mask = None
        if has_pad_mask:
            device = src.device
            #if self.pad_mask is None or self.src_mask.size(0) != len(src):
            mask = (src == 0).t()#.unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
            self.pad_mask = mask.to(device)
        else:
            self.pad_mask = None

        src = src.permute(1, 0, 2)
        src = self.embedding(src)*math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=self.pad_mask)

        output = output.permute(1, 0, 2)
        output = self.decoder(output)

        return output
