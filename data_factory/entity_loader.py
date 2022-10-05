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
import sys
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle
from  sklearn.model_selection import train_test_split


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



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
        #self.train, self.val, _, _= train_test_split(self.train, np.zeros(self.train.shape[0]), test_size=0.15, random_state=1)


        self.test_labels = np.genfromtxt(f'{data_path}SMD_raw/labels/{entity}',
                             dtype=np.int,
                             delimiter=',')

        print(f'train size: {self.train.shape}')
        print(f'val size: {self.val.shape}')
        print(f'test size: {self.test.shape}')

        if forecast:
            filter_anomalies=np.argwhere(self.test_labels==0)
            self.test=self.test[filter_anomalies[:,0]]
            self.test_labels=self.test_labels[filter_anomalies[:,0]]

        '''print(self.train.shape)
        for i in range(self.train.shape[1]):
            print(f'{i}: {len(set(self.train[:,i]))}')

            train_set=set(self.train[:,i])
            if len(train_set)<50:
                print(train_set)
            #print(len([i for i in self.test[:,i] if i not in train_set]))
        sys.exit()'''
        '''import matplotlib.pyplot as plt
        for feat in range(self.train.shape[1]):
            plt.clf()
            plt.plot([i for i in range(self.train.shape[0])],self.train[:,feat],label='train')
            #plt.plot([i for i in range(self.test.shape[0])],self.test[:,feat],label='test')
            plt.plot([i for i in range(self.val.shape[0])],self.val[:,feat],label='test')
            plt.legend()
            print(feat)
            plt.show()
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
            return np.float32(self.train[index:index + self.win_size]), np.zeros_like(self.train[index:index + self.win_size])
            #return np.float32(self.train[index:index + self.win_size]), None
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros_like(self.val[index:index + self.win_size]), index
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]), index
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])




class SMAP_MSL(object):
    def __init__(self, data_path,entity, win_size, step, mode,forecast):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        self.data = np.load(f'{data_path}SMAP_MSL/train/{entity}.npy',)


        self.scaler.fit(self.data)
        self.train = self.scaler.transform(self.data)
        self.test_data = np.load(f'{data_path}SMAP_MSL/test/{entity}.npy',)
        self.test = self.scaler.transform(self.test_data)


        data_len = len(self.train)

        self.val = self.train[(int)(data_len * 0.8):]
        self.train = self.train[:(int)(data_len * 0.8)]
        #self.train, self.val, _, _ = train_test_split(self.train, np.zeros(self.train.shape[0]), test_size=0.15, random_state=1)



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


        self.anomaly_ratio=sum(self.test_labels)/len(self.test_labels)


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
            return np.float32(self.val[index:index + self.win_size]), np.zeros_like(self.val[index:index + self.win_size]), index
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]), index
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class electTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length,mode):
        if mode!='val':
            self.data = np.load(os.path.join(data_path, f'{mode}_data_{data_name}.npy'))
            self.v = np.load(os.path.join(data_path, f'{mode}_v_{data_name}.npy'))
            self.label = np.load(os.path.join(data_path, f'{mode}_label_{data_name}.npy'))
            self.test_len = self.data.shape[0]
            self.pred_length = predict_length
        else:
            self.data=None
        self.train=self.data
        self.test=self.data
        self.val=None

        print(self.label)
        print(self.label.shape)
        print(np.count_nonzero(self.label))
        print(np.mean(self.label))
        print(np.std(self.label))
        print(np.max(self.label))

        print(np.min(self.label))
        sys.exit()

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 2:]
        label = torch.from_numpy(self.label[index].copy())
        v = float(self.v[index][0])
        if v > 0:
            data = label / v
        else:
            data = label

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start+i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start+i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:]

        return all_data.squeeze(0), label#, v

class ForecastDS(object):
    def __init__(self, data_path, win_size, step, mode,dataset):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        if dataset=='electricity':
            train_start = '2011-01-01 00:00:00'
            train_end = '2014-09-01 00:00:00'
            valid_start = '2014-08-25 00:00:00'
            valid_end = '2014-09-08 00:00:00'
            test_start = '2014-09-01 00:00:00'  # need additional 7 days as given info
            test_end = '2014-12-31 23:00:00'

            #self.data = np.load()
            data_frame = pd.read_csv(f'{data_path}electricity/LD2011_2014.txt', sep=";", index_col=0, parse_dates=True, decimal=',')
            #print(data_frame.head())
            data_frame = data_frame.resample('1H', label='left', closed='right').sum()[train_start:test_end]
            #print(data_frame.head())
            #sys.exit()
            self.data = data_frame[train_start:train_end].values
            valid_data = data_frame[valid_start: valid_end].values
            test_data = data_frame[test_start:test_end].values
            print(self.data.shape)
            print(valid_data.shape)
            print(test_data.shape)

        else:
            print("dataset not found")
            sys.exit(1)


        self.scaler.fit(self.data)
        self.train = self.scaler.transform(self.data)
        self.test =self.scaler.transform(test_data)
        self.val = self.scaler.transform(valid_data)
        #self.train=self.data
        #self.test=test_data
        #self.val=valid_data
        self.test_raw=test_data
        print(self.test_raw.shape)
        print(np.count_nonzero(self.test_raw))
        #sys.exit()
    def inverse(self,x):
        return self.scaler.inverse_transform(x)

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
            return np.float32(self.val[index:index + self.win_size]), np.zeros_like(self.val[index:index + self.win_size]), index
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.zeros_like(
                self.test[index:index + self.win_size]), index
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_entity_dataset(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', entity=None, shuffle=False, forecast=None):
    if dataset == 'SMD':
        print(data_path)
        entities=os.listdir(f'{data_path}/SMD_raw/train')
        print(f'Dataset: {entities[entity]}')
        #print(entities)
        dataset=SMDSegLoader(f'{data_path}/SMD/', win_size, step, mode=mode)
        #dataset = SMD(data_path,entities[entity], win_size, step, mode, forecast)
    elif dataset == 'SMAP':
        entities=pd.read_csv(f'{data_path}/SMAP_MSL/labeled_anomalies.csv')
        entities=entities[entities['spacecraft']=='SMAP']
        entities=entities['chan_id'].values
        print(f'Dataset: {entities[entity]}')
        dataset = SMAP_MSL(data_path, entities[entity], win_size, step, mode, forecast)
        #print(entities)
    elif dataset == 'MSL':
        entities = pd.read_csv(f'{data_path}/SMAP_MSL/labeled_anomalies.csv')
        entities = entities[entities['spacecraft'] == 'MSL']
        entities = entities['chan_id'].values
        print(f'Dataset: {entities[entity]}')
        dataset = SMAP_MSL(data_path,entities[entity], win_size, step, mode, forecast)
    elif dataset == 'electricity' :
        #dataset = electTestDataset(data_path+'electricity/', dataset,1,mode )
        dataset = ForecastDS(data_path, win_size, step, mode, dataset)


    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader