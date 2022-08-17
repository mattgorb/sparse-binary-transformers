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


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    i=0
    #print(len(iterator))
    for batch in iterator:
        optimizer.zero_grad()
        data, label, index=batch
        label=label[:,0].long().to(device)
        data=data.to(device)

        i+=1

        predictions,_ = model(data)

        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        acc = binary_accuracy(predictions, label)

        epoch_acc += acc.item()

        #print(i)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def test(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            data, label, index = batch
            label=label[:,0].long().to(device)
            data = data.to(device)
            predictions,_ = model(data)

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
