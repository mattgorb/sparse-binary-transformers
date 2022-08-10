import torch
from torchtext.datasets import IMDB
from models.base.dense_transformer_ts import TSTransformerModel, TranAD_Basic
from models.base.sparse_binary_transformer_ts import TSSparseTransformerModel
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
import warnings
from utils.model_size import *
warnings.filterwarnings("ignore")
from torch.quantization import *
from utils.model_size import get_model_complexity_info
from metrics.flops import flops
from metrics.memory_size import memory, model_size

from metrics.evaluate import evaluate_flops_memory_size
#from utils.trainer import train,test, validation,test_forecast
from data_factory.entity_loader import get_entity_dataset




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device=device
    if 'cuda' in str(device):
        root_dir='/s/luffy/b/nobackup/mgorb/data/'
        weight_file_base='/s/luffy/b/nobackup/mgorb/weights/'+args.weight_file
    else:
        root_dir='data/'
        weight_file_base = 'weights/' + args.weight_file


    for ent in range(1):

        weight_file = weight_file_base + f'_entity_{ent}_ds_{args.dataset}_forecast_{args.forecast}_ws_{args.window_size}.pt'
        print(f'\n\n\nEntity {ent}')
        train_dataloader=get_entity_dataset(root_dir, args.batch_size,mode='train',win_size=args.window_size,
                                            dataset=args.dataset, entity=ent, shuffle=True, forecast=args.forecast)
        val_dataloader=get_entity_dataset(root_dir, args.batch_size,mode='val',win_size=args.window_size,
                                          dataset=args.dataset, entity=ent, forecast=args.forecast)
        test_dataloader=get_entity_dataset(root_dir,args.batch_size, mode='test',
                                           win_size=args.window_size, dataset=args.dataset, entity=ent, forecast=args.forecast)

        input_dim=train_dataloader.dataset.train.shape[1]

        dmodel = input_dim*4

        if args.model_type=='Dense':
            model = TSTransformerModel(input_dim=input_dim, ninp=dmodel, nhead=2, nhid=8, nlayers=2, args=args).to(device)
            #model=TranAD_Basic(feats=input_dim)
            from utils.trainer import train,test,test_forecast,validation

            #from models.base.dense_anomaly_ts import AnomalyTransformer
            #model = AnomalyTransformer(win_size=args.window_size, enc_in=input_dim, c_out=input_dim,e_layers=2, args=args).to(device)
            #from utils.trainer_anomaly import train, test, validation

        else:
            model=TSSparseTransformerModel(input_dim=input_dim, ninp=dmodel, nhead=2, nhid=16, nlayers=2, args=args).to(device)

        freeze_model_weights(model)
        print(f'The model has {count_parameters(model):,} trainable parameters')

        optimizer = optim.Adam(model.parameters(),lr=1e-4)
        criterion = nn.MSELoss(reduction='sum')
        best_val_loss = float('inf')

        if args.evaluate:
            evaluate_flops_memory_size(model, test_dataloader, criterion,train_dataloader, args)
            return


        print(f'number of training batches: {train_dataloader.dataset.__len__()/args.batch_size}')
        print(f'number of test batches: {test_dataloader.dataset.__len__()/args.batch_size}')


        test_loss=None
        for epoch in range(args.epochs):
            #print(f'\nEpoch {epoch}: ')
            start_time = time.time()

            train_loss = train(model, train_dataloader, optimizer, criterion, device,args,epoch)
            val_loss = validation(model, val_dataloader, optimizer, criterion, device,args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), weight_file)
                if epoch>5:
                    #if args.forecast:
                        #test_loss = test_forecast(model, test_dataloader,train_dataloader, criterion, device, args, ent)
                    #else:
                    test_loss = test(model, test_dataloader,val_dataloader, criterion, device, args, ent)
            else:
                val_loss=None
                test_loss=None

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Entity: {ent} | Epoch: {epoch} | Train loss: {train_loss} |  Val loss: {val_loss} |  Test loss: {test_loss}')





if __name__ == "__main__":
    print(args)
    main()
