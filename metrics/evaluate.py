import torch

from models.base.sparse_binary_transformer_nlp import SBTransformerModel
from models.layers.sparse_type import SubnetLinBiprop
from collections import Counter

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.model_utils import *

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


def evaluate(model, test_dataloader, criterion, args,device='cpu'):
    #device ='cpu'
    print(f'device {device}')
    model = model.to(device)
    criterion=criterion.to(device)

    model.eval()
    if args.dataset=='InsectWingbeat' or args.dataset=='JapaneseVowels':

        for batch in test_dataloader:
            data, label, padding, index = batch
            break
    elif args.forecast :
        for batch in test_dataloader:
            data, label = batch
            break

    else:
        for batch in test_dataloader:
            data, label, index = batch
            break
    data = data.to(device)

    print('data size:')
    print(data.size())

    model_input=torch.ones_like(data.float())*.5


    flops_dict, modules_not_found = flops(model, model_input, args)
    for k, v in flops_dict.items():
        print(f'{k}: {v}')

    print('\n\n\n Model Size: ')
    params_dict = model_size(model, args, quantized=False)
    #for k, v in params_dict.items():
        #print(f'{k}: {v}')

    #valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
    #print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')









'''
def evaluate_flops_memory_size(model, test_dataloader, criterion,train_dataloader, args):
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
    max_len=0
    for batch in train_dataloader:
        _, text = batch
        max_len=max(max_len,text.size(0))
    print(max_len)

    model.eval()
    valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
    print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    #flops_dict,modules_not_found=flops(model,torch.ones(max_len,1).int() )
    #for k,v in flops_dict.items():
        #print(f'{k}: {v}')

    #print(f'Modules not found for FLOP measurement: {modules_not_found}')

    #params_dict = model_size(model, args)
    #for k,v in params_dict.items():
        #print(f'{k}: {v}')
    #sys.exit()
    if args.model_type == 'Dense':
        print('\n\n Running Quantized model...')

        qconfig_dict = {
            torch.nn.Embedding: float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig,
        }
        quantize_dynamic(model, qconfig_dict, inplace=True,)


        model.eval()
        valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
        print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        flops_dict, modules_not_found = flops(model, torch.ones(max_len, 1).int())
        for k, v in flops_dict.items():
            print(f'{k}: {v}')
        print(f'Modules not found for FLOP measurement: {modules_not_found}')


        params_dict = model_size(model, args, quantized=True)
        for k, v in params_dict.items():
            print(f'{k}: {v}')

        valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
        print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    else:
        print('\n\n Running Quantized model...')

        qconfig_dict = {
            torch.nn.Embedding: float_qparams_weight_only_qconfig,
        }
        quantize_dynamic(model, qconfig_dict, inplace=True,)

        model.eval()
        valid_loss, valid_acc = test(model, test_dataloader, criterion, device)
        print(f'\t Quantized Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
'''



















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