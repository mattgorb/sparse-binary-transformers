import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.positional_encoder import PositionalEncoding, LearnablePositionalEncoding
from models.layers.sparse_type import linear_init,emb_init

class TSClsSparseTransformerModel(nn.Module):

    def __init__(self, input_dim, ninp, nhead, nhid, nlayers=6, args=None,classification_labels=None):
        super(TSClsSparseTransformerModel, self).__init__()
        try:
            from models.layers.sparse_encoder import SparseTransformerEncoder
            from models.layers.sparse_encoder_layer import  SparseTransformerEncoderLayer
            from models.layers.base_transformer_encoder import TransformerEncoder
            from models.layers.base_transformer_encoder_layer import TransformerEncoderLayer
        except:
            raise ImportError("Had trouble importing transformer modules. ")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pad_mask = None
        self.args=args
        if args.pos_enc=='Learnable':
            self.pos_encoder = LearnablePositionalEncoding(ninp,)
        else:
            self.pos_encoder = PositionalEncoding(ninp,)

        encoder_layers = SparseTransformerEncoderLayer(ninp, nhead, nhid, args=self.args)
        self.transformer_encoder = SparseTransformerEncoder(encoder_layers, nlayers)
        #encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid,args=self.args,)
        #self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.embedding = linear_init(input_dim, ninp,args=args,)
        #self.embedding = nn.Linear(input_dim, ninp,  )
        self.ninp = ninp
        self.decoder = linear_init(ninp, classification_labels,bias=False,args=args, )
        #self.decoder = nn.Linear(ninp, classification_labels,  )

        #self.init_weights()
        self.act=nn.ReLU()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        #nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_src_mask=False, pad_mask=None, ):
        if has_src_mask:
            size=src.size(1)
            mask=torch.eye(size,)
            mask=mask.masked_fill(mask == 0, float('-inf'))
            mask[-1,:]=0

            mask[-1,-1]=float('-inf')
            self.src_mask = mask.to(self.args.device)

        else:
            self.src_mask = None
        if pad_mask is not None:
            device = src.device
            self.pad_mask = pad_mask.to(device)

        src = src.permute(1, 0, 2)
        src = self.embedding(src)*math.sqrt(self.ninp)
        src = self.pos_encoder(src)


        output, attention_list = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=self.pad_mask)

        output = self.act(output)

        output = output.permute(1, 0, 2)
        output = self.decoder(output)


        output = output.mean(dim=1)
        output=F.log_softmax(output)

        return output, attention_list
