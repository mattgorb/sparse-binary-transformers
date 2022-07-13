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
import warnings
from utils.model_size import *
warnings.filterwarnings("ignore")
from torch.quantization import *
from utils.model_size import get_model_complexity_info
from metrics.flops import flops
from metrics.memory_size import memory, model_size
from metrics.accuracy import test, binary_accuracy
from metrics.evaluate import evaluate_flops_memory_size


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    i=0
    for batch in iterator:
        optimizer.zero_grad()
        label, text=batch
        label=label.to(device)
        text=text.to(device)

        i+=1

        predictions = model(text)#.squeeze(1)
        loss = criterion(predictions, label)


        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        acc = binary_accuracy(predictions, label)

        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def collate_batch(batch):
   label_list, text_list = [], []
   for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
   return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def main():
    SEED = 1234

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if str(device)=='cuda':
        root_dir='/s/luffy/b/nobackup/mgorb/data/imdb'
        args.weight_file='/s/luffy/b/nobackup/mgorb/weights/'+args.weight_file
    else:
        root_dir='data'
        args.weight_file = 'weights/' + args.weight_file

    print(root_dir)

    #train_data, test_data = IMDB(split=('train', 'test'), root=root_dir)
    train_iter = IMDB(split='train', root=root_dir)
    test_iter = IMDB(split='test', root=root_dir)

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    #print(list(train_iter))

    global text_transform
    global label_transform
    text_transform = lambda x:  [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
    label_transform = lambda x: 1 if x == 'pos' else 0
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for (label, line) in train_dataset:
        counter.update(tokenizer(line))
    #print(torchtext.__version__)
    vocab = torchtext.vocab.vocab(counter, min_freq=50, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    vocab.set_default_index(vocab['<unk>'])
    #torchtext.vocab.v
    ntokens=vocab.__len__()
    print(ntokens)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True,collate_fn=collate_batch)

    EMBEDDING_DIM = 50

    if args.model_type=='Dense':
        model = TransformerModel(ntoken=ntokens, ninp=EMBEDDING_DIM, nhead=2, nhid=16, nlayers=2).to(device)

    else:
        model=SBTransformerModel(ntoken=ntokens, ninp=EMBEDDING_DIM, nhead=2, nhid=16, nlayers=2, args=args).to(device)
    print(model)

    freeze_model_weights(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    #sys.exit()


    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    #criterion = nn.BCEWithLogitsLoss()


    criterion = nn.CrossEntropyLoss()

    N_EPOCHS = 10

    best_valid_loss = float('inf')

    if args.evaluate:
        evaluate_flops_memory_size(model, test_dataloader, criterion,train_dataloader)
        return


    for epoch in range(args.epochs):

        start_time = time.time()

        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        #model_int8 = quantize_fx.convert_fx(model)
        valid_loss, valid_acc = test(model, test_dataloader, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.weight_file)


        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')




if __name__ == "__main__":
    print(args)
    main()
