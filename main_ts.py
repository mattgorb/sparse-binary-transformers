import torch
from torchtext.datasets import IMDB
from models.base.dense_transformer_ts import TSTransformerModel
#from models.base.sparse_binary_transformer_ts import SBTransformerModel
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
from utils.trainer import train,test
from data_factory.data_loader import get_dataset




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if str(device)=='cuda':
        root_dir='/s/luffy/b/nobackup/mgorb/data/'
        args.weight_file='/s/luffy/b/nobackup/mgorb/weights/'+args.weight_file
    else:
        root_dir='data/'
        args.weight_file = 'weights/' + args.weight_file

    train_dataloader=get_dataset(root_dir, args.batch_size,mode='train', dataset=args.dataset)
    test_dataloader=get_dataset(root_dir,args.batch_size, mode='test', dataset=args.dataset)

    input_dim=train_dataloader.dataset.train.shape[1]

    dmodel = 64

    if args.model_type=='Dense':
        model = TSTransformerModel(input_dim=input_dim, ninp=dmodel, nhead=2, nhid=16, nlayers=2).to(device)

    #else:
        #model=SBTransformerModel(ntoken=ntokens, ninp=EMBEDDING_DIM, nhead=2, nhid=16, nlayers=2, args=args).to(device)
    #print(model)

    freeze_model_weights(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')



    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    #criterion = nn.BCEWithLogitsLoss()


    criterion = nn.MSELoss()

    N_EPOCHS = 10

    best_valid_loss = float('inf')

    if args.evaluate:
        evaluate_flops_memory_size(model, test_dataloader, criterion,train_dataloader, args)
        return


    print(f'number of training batches: {train_dataloader.dataset.__len__()/args.batch_size}')
    print(f'number of test batches: {test_dataloader.dataset.__len__()/args.batch_size}')

    for epoch in range(args.epochs):

        start_time = time.time()

        #train_loss = train(model, train_dataloader, optimizer, criterion, device)

        valid_loss = test(model, test_dataloader, criterion, device, args, epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.weight_file)


        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} ')
        print(f'\t Val. Loss: {valid_loss:.3f} ')




if __name__ == "__main__":
    print(args)
    main()