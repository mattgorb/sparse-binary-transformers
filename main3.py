import torch
import time
from torchtext.datasets import IMDB
import random
import torch.optim as optim
import torch.nn as nn
from model import TransformerModel
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
SEED = 1234

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device=='cuda':
    root_dir='/s/luffy/b/nobackup/mgorb/data/imdb'
else:
    root_dir='data'


#train_data, test_data = IMDB(split=('train', 'test'), root=root_dir)
train_iter = IMDB(split='train', root=root_dir)
test_iter = IMDB(split='test', root=root_dir)



tokenizer = get_tokenizer('basic_english')

counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))

text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
label_transform = lambda x: 1 if x == 'pos' else 0

def collate_batch(batch):
   label_list, text_list = [], []
   for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
   return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)


train_dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=True,
                              collate_fn=collate_batch)

for batch in train_dataloader:
    print(batch)
    sys.exit()

print(len(train_dataloader))
print(vars(train_data.examples[0]))



train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


MAX_VOCAB_SIZE = 50_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))
print(LABEL.vocab.stoi)


BATCH_SIZE = 64


train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)




def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    #rounded_preds = torch.round(torch.sigmoid(preds))
    #correct = (rounded_preds == y).float() #convert into float for division
    #acc = correct.sum() / len(correct)
    _, predicted = torch.max(preds, 1)

    acc = ((predicted == y).sum()/y.size(0))
    return acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



EMBEDDING_DIM = 50
#print(len(vocab.dictionary))
#sys.exit()
#ntoken, ninp, nhead, nhid, nlayers=6,
model = TransformerModel(ntoken=len(TEXT.vocab), ninp=EMBEDDING_DIM, nhead=5, nhid=16, nlayers=2).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

#for n,p in model.named_parameters():
    #print(n)
    #if p.requires_grad:
        #print(p.size())
#@print( model.parameters())


optimizer = optim.Adam(model.parameters(),lr=1e-4)
#criterion = nn.BCEWithLogitsLoss()


criterion = nn.CrossEntropyLoss()

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    i=0
    for batch in iterator:
        print(f'batch {i}')
        i+=1
        optimizer.zero_grad()

        predictions = model(batch.text.to(device))#.squeeze(1)

        label=batch.label.to(device)
        #print(label)
        #print(predictions)
        #print(label.dtype)
        label=label.type(torch.LongTensor)
        #print(predictions.size())
        #print(label.size())
        loss = criterion(predictions, label)

        acc = binary_accuracy(predictions, label)
        print(acc)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')