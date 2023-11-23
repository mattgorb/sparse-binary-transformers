import torch
from models.base.dense_transformer_ts import TSTransformerModel
from models.base.sparse_binary_transformer_ts import TSSparseTransformerModel
from models.base.dense_transformer_ts_forecast import TSTransformerModelForecast
from models.layers.sparse_type import SubnetLinBiprop
from collections import Counter

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.model_utils import *

import time
from torch import optim
from args import args
import warnings
from utils.model_size import *
warnings.filterwarnings("ignore")
from torch.quantization import *
from utils.model_size import get_model_complexity_info
from metrics.flops import flops
from metrics.memory_size import memory, model_size
from utils.trainer import train,test_forecast,validation,test_anomaly_detection,train_forecast, train_lt_forecast, test_lt_forecast
from metrics.evaluate import evaluate
#from utils.trainer import train,test, validation,test_forecast
from data_factory.entity_loader import get_entity_dataset
from models.layers.sparse_type import SubnetLinBiprop



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_model(model, args):
    for n, m in model.named_modules():
        print(f'{n}, {m._get_name()}')
        #if hasattr(m, "weight") and m.weight is not None:
            #print(f'{n}: {m.weight.numel()}')

def main():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device=device
    if 'cuda' in str(device):
        root_dir='/s/lovelace/c/nobackup/iray/mgorb/luffy-data/'
        weight_file_base='/s/lovelace/c/nobackup/iray/mgorb/weights/'+args.weight_file
    else:
        root_dir='data/'
        weight_file_base = 'weights/' + args.weight_file

    ent=1

    weight_file = weight_file_base + f'_entity_{ent}_ds_{args.dataset}_forecast_{args.forecast}_ws_{args.window_size}.pt'

    train_dataloader=get_entity_dataset(root_dir, args.batch_size,mode='train',win_size=args.window_size,
                                        dataset=args.dataset, entity=ent, shuffle=True, forecast=args.forecast)
    val_dataloader=get_entity_dataset(root_dir, args.batch_size,mode='val',win_size=args.window_size,
                                      dataset=args.dataset, entity=ent, forecast=args.forecast)
    test_dataloader=get_entity_dataset(root_dir,args.batch_size, mode='test',
                                       win_size=args.window_size, dataset=args.dataset, entity=ent, forecast=args.forecast)

    input_dim=train_dataloader.dataset.train.shape[1]
    #input_dim=7

    if args.dmodel is None:
        dmodel = input_dim*2
    else:
        dmodel = args.dmodel

    if args.model_type=='Dense':
        model = TSTransformerModel(input_dim=input_dim, ninp=dmodel, nhead=2, nhid=256, nlayers=2, args=args).to(device)
    elif args.model_type=='Sparse':
        model=TSSparseTransformerModel(input_dim=input_dim, ninp=dmodel, nhead=2, nhid=256, nlayers=2, args=args).to(device)
    else:
        print("Invalid")
        sys.exit()
    print_model(model, args)

    freeze_model_weights(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    
    criterion = nn.MSELoss(reduction='none')
    best_loss = float('inf')


    if args.evaluate:
        evaluate(model, test_dataloader, criterion, args,device)
        sys.exit(0)


    print(f'number of training batches: {train_dataloader.dataset.__len__()/args.batch_size}')
    print(f'number of test batches: {test_dataloader.dataset.__len__()/args.batch_size}')
    print(f'number of val batches: {val_dataloader.dataset.__len__()/args.batch_size}')


    early_stopping_increment=0
    for epoch in range(args.epochs):

        train_loss = train_lt_forecast(model, train_dataloader, optimizer, criterion, device, args, epoch)
        test_loss_mse, test_loss_mae = test_lt_forecast(model, test_dataloader, train_dataloader, criterion, device, args, epoch,)
        if test_loss_mse<best_loss:
            best_loss=test_loss_mse
            torch.save(model.state_dict(), weight_file)
            print(f"New best, Saving model... ")
            print(f'\t Performance: Epoch: {epoch} | Train loss: {train_loss} |  Test loss: {test_loss_mse}')

        print(f'Epoch: {epoch} | Train loss: {train_loss} |  Test loss: {test_loss_mse}')



if __name__ == "__main__":
    print(args)
    for run in range(args.model_runs):
        print(f'Running experiment with weight seed {args.weight_seed}')
        main()
        args.weight_seed+=1
