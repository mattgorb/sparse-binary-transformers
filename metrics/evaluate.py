import torch
from torchtext.datasets import IMDB
from models.base.dense_transformer import TransformerModel
from models.base.sparse_binary_transformer import SBTransformerModel
from models.layers.sparse_type import SubnetLinBiprop
from collections import Counter
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.model_utils import *
from torchtext.data.functional import to_map_style_dataset
import time
from torch import optim
from args import args
from torch.quantization import *
#from torch.quantization.quantize import *
from utils.model_size import get_model_complexity_info
from metrics.flops import flops
from metrics.memory_size import memory, model_size,state_dict_size
from metrics.accuracy import test
from utils.model_size import *
from models.layers.base_multihead_attention import MultiheadAttention as DenseMultiheadAttention
import warnings
warnings.filterwarnings("ignore")



def evaluate_flops_memory_size(model, test_dataloader, criterion,train_dataloader):
    device ='cpu'
    model = model.to(device)
    criterion=criterion.to(device)
    model.load_state_dict(torch.load(args.weight_file, map_location=torch.device('cpu')))
    model.eval()

    #valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if args.dataset=='imdb':
        max_len=2754
    else:
        sys.exit()
    '''max_len=0
    for batch in train_dataloader:
        _, text = batch
        max_len=max(max_len,text.size(0))
    print(max_len)'''

    model.eval()
    #valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
    #print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    #flops_dict,modules_not_found=flops(model,torch.ones(max_len,1).int() )
    #for k,v in flops_dict.items():
        #print(f'{k}: {v}')

    #print(f'Modules not found for FLOP measurement: {modules_not_found}')

    params_dict = model_size(model)
    for k,v in params_dict.items():
        print(f'{k}: {v}')
    sys.exit()
    if args.model_type == 'Dense':
        print('\n\n Running Quantized model...')

        qconfig_dict = {
            torch.nn.Embedding: float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig,
        }
        quantize_dynamic(model, qconfig_dict, inplace=True,)
        '''for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                print(name, layer)
                layer.qconfig=torch.quantization.default_qconfig
            if isinstance(layer, nn.LayerNorm):
                print(name, layer)
                layer.qconfig=torch.quantization.default_qconfig
            if isinstance(layer, nn.Embedding):
                print(name, layer)
                layer.qconfig=torch.quantization.float_qparams_weight_only_qconfig
        torch.quantization.prepare(model, inplace=True,)
        torch.quantization.convert(model ,inplace=True,)'''

        model.eval()
        valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
        print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        flops_dict, modules_not_found = flops(model, torch.ones(max_len, 1).int())
        for k, v in flops_dict.items():
            print(f'{k}: {v}')
        print(f'Modules not found for FLOP measurement: {modules_not_found}')
        params_dict = model_size(model)
        for k, v in params_dict.items():
            print(f'{k}: {v}')

        valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
        print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    else:
        sys.exit()
        print(model.transformer_encoder.layers[0].linear1.calc_alpha())
        for n,m in model.transformer_encoder.layers[0].linear1.named_parameters():
            print(n)
        print(model.transformer_encoder.layers[0].linear1.get_buffer('alpha'))
        sys.exit()
        #print(model)