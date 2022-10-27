import torch
from torchtext.datasets import IMDB
from models.base.dense_transformer_ts_classification import TSClassificationTransformer
from models.base.sparse_binary_transformer_ts_cls import TSClsSparseTransformerModel
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
from utils.trainer_ts_classification import train,test

from metrics.evaluate import evaluate
#from utils.trainer import train,test, validation,test_forecast
from data_factory.ts_classification_loader import get_classification_ds




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

    if args.dataset=='SMD':
        entities=28

    train_dataloader,val_dataloader, test_dataloader, classification_labels, input_dim =get_classification_ds(args.dataset,root_dir, args)


    weight_file = weight_file_base + f'classification_ds_{args.dataset}_ws_{args.window_size}.pt'


    if args.model_type=='Dense':
        model = TSClassificationTransformer(input_dim=input_dim, ninp=args.dmodel, nhead=args.n_head, nhid=args.nhid,
                                            nlayers=args.n_layers, args=args,classification_labels=classification_labels).to(device)
        model=model.float()
    else:
        model=TSClsSparseTransformerModel(input_dim=input_dim, ninp=args.dmodel, nhead=args.n_head, nhid=args.nhid,
                                            nlayers=args.n_layers,args=args,classification_labels=classification_labels).to(device)
        model=model.float()

        freeze_model_weights(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(),lr=float(args.lr))
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    if args.evaluate:
        evaluate(model, test_dataloader, criterion, args)
        return

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device,args.dataset)

        val_loss, val_acc=test(model, val_dataloader, criterion, device,args.dataset)
        test_loss, test_acc = test(model, test_dataloader, criterion, device, args.dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_file)
            #test_loss, test_acc = test(model, test_dataloader, criterion, device,args.dataset)
        #else:
            #val_loss=None
            #test_loss=None
            #val_acc=None
            #test_acc=None
        print(f'Train acc: {train_acc} | Val acc: {val_acc} | Test acc: {test_acc}')


if __name__ == "__main__":
    print(args)
    for run in range(args.model_runs):
        print(f'Running experiment with weight seed {args.weight_seed}')
        main()
        args.weight_seed+=1
