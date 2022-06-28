import argparse
import logging
import time

import torch
#from model import TextClassificationModel
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.datasets import DATASETS
#from torchtext.experimental.transforms import load_sp_model, PRETRAINED_SP_MODEL, SentencePieceTokenizer
from torchtext.utils import download_from_url
from torchtext.vocab import build_vocab_from_iterator

r"""
This file shows the training process of the text classification model.
"""


def yield_tokens(data_iter, ngrams):
    for _, text in data_iter:
        yield ngrams_iterator(tokenizer(text), ngrams)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classification model on text classification datasets.")
    parser.add_argument("--dataset", type=str, default="IMDB")
    parser.add_argument("--num-epochs", type=int, default=5, help="num epochs (default=5)")
    parser.add_argument("--embed-dim", type=int, default=30, help="embed dim. (default=32)")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size (default=16)")
    parser.add_argument("--split-ratio", type=float, default=0.95, help="train/valid split ratio (default=0.95)")
    parser.add_argument("--lr", type=float, default=4.0, help="learning rate (default=4.0)")
    parser.add_argument("--lr-gamma", type=float, default=0.8, help="gamma value for lr (default=0.8)")
    parser.add_argument("--ngrams", type=int, default=2, help="ngrams (default=2)")
    parser.add_argument("--num-workers", type=int, default=1, help="num of workers (default=1)")
    parser.add_argument("--device", default="cpu", help="device (default=cpu)")
    parser.add_argument("--data-dir", default="data", help="data directory (default=.data)")
    parser.add_argument(
        "--use-sp-tokenizer", type=bool, default=False, help="use sentencepiece tokenizer (default=False)"
    )
    parser.add_argument("--dictionary", help="path to save vocab")
    parser.add_argument("--save-model-path", help="path for saving model")
    parser.add_argument("--logging-level", default="WARNING", help="logging level (default=WARNING)")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data_dir = args.data_dir
    split_ratio = args.split_ratio
    ngrams = args.ngrams
    use_sp_tokenizer = args.use_sp_tokenizer

    logging.basicConfig(level=getattr(logging, args.logging_level))

    '''if use_sp_tokenizer:
        sp_model_path = download_from_url(PRETRAINED_SP_MODEL["text_unigram_15000"], root=data_dir)
        sp_model = load_sp_model(sp_model_path)
        tokenizer = SentencePieceTokenizer(sp_model)
    else:'''
    tokenizer = get_tokenizer("basic_english")

    train_iter = DATASETS[args.dataset](root=data_dir, split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, ngrams), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))

    def label_pipeline(x):
        if x=='neg':
            return 0
        elif x=='pos':
            return 1
        else:
            print('label failure')
            sys.exit()


    train_iter = DATASETS[args.dataset](root=data_dir, split="train")
    num_class = len(set([label for (label, _) in train_iter]))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    from model import TransformerModel
    print(len(vocab))
    #print(len(vocab.dictionary))
    #sys.exit()
    #ntoken, ninp, nhead, nhid, nlayers=6,
    model = TransformerModel(ntoken=len(vocab), ninp=embed_dim, nhead=6, nhid=50, nlayers=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    train_iter, test_iter = DATASETS[args.dataset]()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])




    train_dataloader = DataLoader(split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        for idx, batch in enumerate(iter(train_iter)):
            label, text=batch
            print(batch)
            #print(batch.label)
            sys.exit()

            predictions = model(text)

            print(predictions)
            labels = batch.label.to(device) - 1

            loss = criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)

            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_correct = 0
            test_epoch_count = 0

            for idx, batch in enumerate(iter(test_iter)):
                predictions = model(batch.text.to(device))
                labels = batch.label.to(device) - 1
                test_loss = criterion(predictions, labels)

                correct = predictions.argmax(axis=1) == labels
                acc = correct.sum().item() / correct.size(0)

                test_epoch_correct += correct.sum().item()
                test_epoch_count += correct.size(0)
                test_epoch_loss += loss.item()

        #print(f"{epoch_loss=}")
        print(f"epoch accuracy: {epoch_correct / epoch_count}")
        #print(f"{test_epoch_loss=}")
        print(f"test epoch accuracy: {test_epoch_correct / test_epoch_count}")



        train, test = torchtext.datasets.IMDB()
        loaded_vectors = vocab.GloVe(name='6B', dim=100)