import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.positional_encoder import PositionalEncoding, LearnablePositionalEncoding



# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
    def __init__(self, feats):
        super(TranAD_Basic, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            #from models.layers.base_transformer_encoder_layer import TransformerEncoderLayer
            from torch.nn import TransformerDecoder, TransformerDecoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.name = 'TranAD_Basic'
        #self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 100
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        #self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        x = self.transformer_decoder(tgt, memory)
        #x = self.fcn(x)

        return x




class TSTransformerModel(nn.Module):

    def __init__(self, input_dim, ninp, nhead, nhid, args, nlayers=6, dropout=0.0):
        super(TSTransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder
            from models.layers.base_transformer_encoder_layer import TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pad_mask = None
        self.args=args
        if args.pos_enc=='Learnable':
            self.pos_encoder = LearnablePositionalEncoding(ninp, dropout)
        else:
            self.pos_encoder = PositionalEncoding(ninp,)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,args=self.args,)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.act=nn.ReLU()

        self.embedding = nn.Linear(input_dim, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, input_dim)

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

    def forward(self, src, has_src_mask=True, has_pad_mask=False):
        if has_src_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                #mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                size=src.size(1)
                mask=torch.zeros(size,size)
                mask=mask.masked_fill(mask == 0, float('-inf'))
                mask[-1,:]=0
                self.src_mask = mask.to(self.args.device)

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
        output=self.act(output)
        output = output.permute(1, 0, 2)
        output = self.decoder(output)

        return output
