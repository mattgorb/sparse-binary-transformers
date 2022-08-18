from metrics.accuracy import binary_accuracy
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from itertools import groupby
from operator import itemgetter
import pandas as pd
from metrics.pot.pot import pot_eval
from utils.train_util import adjust_learning_rate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from metrics.accuracy import binary_accuracy


def train(model, iterator, optimizer, criterion, device,dataset):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    i=0
    pad_mask=None
    for batch in iterator:
        optimizer.zero_grad()

        if dataset=='InsectWingbeat':
            data, label,pad_mask, index=batch
        else:
            data, label, index=batch

        label=label[:,0].long().to(device)
        data=data.float().to(device)

        predictions,_ = model(data.float(),pad_mask=pad_mask)

        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        acc = binary_accuracy(predictions, label)

        epoch_acc += acc.item()

        #print(i)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def test(model, iterator, criterion, device,dataset):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        pad_mask = None
        for batch in iterator:
            if dataset == 'InsectWingbeat':
                data, label, pad_mask, index = batch
            else:
                data, label, index = batch

            label = label[:, 0].long().to(device)
            data = data.float().to(device)
            predictions, _ = model(data, pad_mask=pad_mask)

            loss = criterion(predictions, label)
            print('here')
            print(predictions)
            print(label)

            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    sys.exit()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
