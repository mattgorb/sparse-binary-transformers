import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.positional_encoder import PositionalEncoding

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers=6, dropout=0.0):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder
            #from models.layers.base_transformer_encoder_layer import TransformerEncoderLayer
            from torch.nn import TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pad_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 2)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_src_mask=False, has_pad_mask=True):
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

        src = self.embedding(src)*math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=self.pad_mask)
        output = self.decoder(output)

        output = output.mean(dim=0)

        return F.log_softmax(output, dim=-1)