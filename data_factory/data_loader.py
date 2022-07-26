import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle



class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "SMD/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "SMD/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "SMD/SMD_test_label.npy")


        print(self.test.shape)
        print(self.test[:100,8])
        print(self.test[-100:,8])
        import matplotlib.pyplot as plt
        plt.plot([i for i in range(1000)], test_data[:1000,8])
        plt.show()
        plt.clf()
        plt.plot([i for i in range(1000)], self.test[:1000,8])
        plt.show()

        sys.exit()
        '''print(data[0])
        print(test_data[0])
        print(data.shape)
        print(test_data.shape)

        feat=self.train.shape[1]
        import matplotlib.pyplot as plt
        for f in range(feat):
            plt.clf()

            plt.plot([i for i in range(self.train[450000:500000,f].shape[0])],self.train[450000:500000,f])
            plt.savefig(f'output/{f}_train.png')

        feat=self.test.shape[1]
        import matplotlib.pyplot as plt
        for f in range(feat):
            plt.clf()

            plt.plot([i for i in range(self.test[450000:500000,f].shape[0])],self.test[450000:500000,f])
            plt.savefig(f'output/{f}_test.png')
        sys.exit()'''


    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size)
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size)
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size)
        else:
            return (self.test.shape[0] - self.win_size)

    def __getitem__(self, index):

        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
            #return np.float32(self.train[index:index + self.win_size]), None
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



def get_dataset(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    '''elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)'''

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader