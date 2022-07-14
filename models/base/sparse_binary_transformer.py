import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.sparse_type import linear_init,emb_init
from models.layers.positional_encoder import PositionalEncoding


class SBTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers=6, args=None):
        super(SBTransformerModel, self).__init__()
        try:
            from models.layers.sparse_encoder import SparseTransformerEncoder
            from models.layers.sparse_encoder_layer import  SparseTransformerEncoderLayer
        except:
            raise ImportError("Had trouble importing transformer modules. ")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.args=args

        self.pos_encoder = PositionalEncoding(ninp, )
        encoder_layers = SparseTransformerEncoderLayer(ninp, nhead, nhid, args=self.args)
        self.transformer_encoder = SparseTransformerEncoder(encoder_layers, nlayers)
        self.embedding = emb_init(ntoken, ninp,args=args,)
        #self.encoder = nn.Embedding(ntoken, ninp,  )
        self.ninp = ninp
        self.decoder = linear_init(ninp, 2,bias=False,args=args, )


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        #nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):

                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None


        src = self.encoder(src)*math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = output.mean(dim=0)
        return F.log_softmax(output, dim=-1)