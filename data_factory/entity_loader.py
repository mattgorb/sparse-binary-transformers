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
import ast
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle

class SMD(object):
    def __init__(self, data_path,entity, win_size, step, mode,forecast):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        #self.scaler=MinMaxScaler()

        self.data = np.genfromtxt(f'{data_path}SMD_raw/train/{entity}',
                             dtype=np.float64,
                             delimiter=',')
        self.scaler.fit(self.data)
        self.train = self.scaler.transform(self.data)
        self.test_data = np.genfromtxt(f'{data_path}SMD_raw/test/{entity}',
                             dtype=np.float64,
                             delimiter=',')
        self.test = self.scaler.transform(self.test_data)

        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.train = self.train[:(int)(data_len * 0.8)]


        self.test_labels = np.genfromtxt(f'{data_path}SMD_raw/labels/{entity}',
                             dtype=np.int,
                             delimiter=',')

        if forecast:
            filter_anomalies=np.argwhere(self.test_labels==0)
            self.test=self.test[filter_anomalies[:,0]]
            self.test_labels=self.test_labels[filter_anomalies[:,0]]


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
            return np.float32(self.train[index:index + self.win_size]), np.zeros_like(self.train[index:index + self.win_size])
            #return np.float32(self.train[index:index + self.win_size]), None
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros_like(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]), index
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])




class SMAP(object):
    def __init__(self, data_path,entity, win_size, step, mode,forecast):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        #self.scaler=MinMaxScaler()



        self.data = np.load(f'{data_path}SMAP_MSL/train/{entity}.npy',)


        self.scaler.fit(self.data)
        self.train = self.scaler.transform(self.data)
        self.test_data = np.load(f'{data_path}SMAP_MSL/test/{entity}.npy',)
        self.test = self.scaler.transform(self.test_data)

        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.train = self.train[:(int)(data_len * 0.8)]


        '''print(self.train.shape)
        print(self.test.shape)

        import matplotlib.pyplot as plt
        for feat in range(self.train.shape[1]):
            plt.clf()
            plt.plot([i for i in range(self.data.shape[0])],self.data[:,feat],label='traini')
            plt.plot([i for i in range(self.test_data.shape[0])],self.test_data[:,feat],label='t')
            plt.legend()
            plt.show()
        sys.exit()'''

        entities=pd.read_csv(f'{data_path}/SMAP_MSL/labeled_anomalies.csv')
        anomalies=entities[entities['chan_id']==entity]['anomaly_sequences'].values[0]
        self.anomalies = ast.literal_eval(anomalies)
        self.test_labels=[0 for i in range(self.test.shape[0])]
        self.test_labels = np.zeros([self.test.shape[0]], )
        for a in self.anomalies:
            self.test_labels[a[0]:a[1]+1]=1
        if forecast:
            filter_anomalies=np.argwhere(self.test_labels==0)
            self.test=self.test[filter_anomalies[:,0]]
            self.test_labels=self.test_labels[filter_anomalies[:,0]]


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
            return np.float32(self.train[index:index + self.win_size]), np.zeros_like(self.train[index:index + self.win_size])
            #return np.float32(self.train[index:index + self.win_size]), None
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros_like(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]), index
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])





def get_entity_dataset(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', entity=None, shuffle=False, forecast=None):
    if dataset == 'SMD':
        entities=os.listdir(f'{data_path}/SMD_raw/train')
        print(f'Dataset: {entities[entity]}')
        print(entities)

        dataset = SMD(data_path,entities[entity], win_size, step, mode, forecast)
    elif dataset == 'SMAP':
        entities=pd.read_csv(f'{data_path}/SMAP_MSL/labeled_anomalies.csv')
        entities=entities[entities['spacecraft']=='SMAP']
        entities=entities['chan_id'].values
        print(f'Dataset: {entities[entity]}')
        #print(entities)

        dataset = SMAP(data_path,entities[entity], win_size, step, mode, forecast)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader