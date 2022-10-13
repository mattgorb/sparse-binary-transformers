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
from data_factory.timefeatures import time_features

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
            return (self.train.shape[0] - self.win_size)
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size)
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size)
        else:
            return (self.test.shape[0] - self.win_size)

    def __getitem__(self, index):
        #index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]), index
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]), index
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


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1/ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, win_size=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.seq_len=win_size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]

        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        print(df_raw.shape)
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        print(df_data.shape)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        print(df_stamp)
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class ForecastDS(object):
    def __init__(self, data_path, win_size, step, mode,dataset):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        if dataset=='electricity':
            '''train_start = '2011-01-01 00:00:00'
            train_end = '2014-09-01 00:00:00'
            valid_start = '2014-08-25 00:00:00'
            valid_end = '2014-09-08 00:00:00'
            test_start = '2014-09-01 00:00:00'
            test_end = '2014-12-31 23:00:00'

            data_frame = pd.read_csv(f'{data_path}electricity/LD2011_2014.txt', sep=";", index_col=0, parse_dates=True, decimal=',')
            data_frame = data_frame.resample('1H', label='left', closed='right').sum()[train_start:test_end]'''

            #informer dates
            train_start = '2012-01-01 00:00:00'
            train_end = '2014-02-06 03:0:00'
            valid_start = '2014-02-04 04:00:00'
            valid_end = '2014-05-26 20:00:00'
            test_start = '2014-05-24 20:00:00'
            test_end = '2015-01-01 00:00:00'

            #pyraformer dates
            '''train_start = '2012-01-01 00:00:00'
            train_end = '2014-09-01 00:00:00'
            valid_start = '2014-08-25 00:00:00'
            valid_end = '2014-09-08 00:00:00'
            test_start = '2014-09-01 00:00:00'
            test_end = '2014-12-31 23:00:00'''
            data_frame = pd.read_csv(f'{data_path}electricity/ECL.csv', index_col=0, parse_dates=True,)

            self.data = data_frame[train_start:train_end].values
            valid_data = data_frame[valid_start: valid_end].values
            test_data = data_frame[test_start:test_end].values
            print(self.data.shape)
            print(valid_data.shape)
            print(test_data.shape)
        elif dataset=='ETTh1':
            #informer
            '''train_start = '2016-07-01 00:00:00'
            train_end = '2017-06-25 23:00:00'
            valid_start = '2017-06-24 00:00:00'
            valid_end = '2017-10-23 23:00:00'
            test_start = '2017-10-22 00:00:00'
            test_end = '2018-02-20 23:00:00'''''

            #pyraformer
            train_start = '2016-07-01 00:00:00'
            train_end = '2017-10-22 00:00:00'
            valid_start = '2017-06-24 00:00:00'
            valid_end = '2017-10-23 23:00:00'
            test_start = '2017-10-22 00:00:00'
            test_end = '2018-02-20 23:00:00'
            data_frame = pd.read_csv(f'{data_path}ETTh1/ETTh1.csv', index_col=0, parse_dates=True,)

            self.data = data_frame[train_start:train_end].values
            valid_data = data_frame[valid_start: valid_end].values
            test_data = data_frame[test_start:test_end].values

            print(self.data.shape)
            print(valid_data.shape)
            print(test_data.shape)
        elif dataset=='ETTm1':
            #informer dates
            '''train_start = '2016-07-01 00:00:00'
            train_end = '2017-06-25 23:45:00'
            valid_start = '2017-06-25 12:00:00'
            valid_end = '2017-10-23 23:45:00'
            test_start = '2017-10-23 12:00:00'
            test_end = '2018-02-20 23:45:00'''''

            #pyraformer dates
            train_start = '2016-07-01 00:00:00'
            train_end = '2017-10-20 00:00:00'
            valid_start = '2016-07-01 00:00:00'
            valid_end = '2017-10-20 23:45:00'
            test_start = '2017-10-20 00:00:00'
            test_end = '2018-02-20 23:45:00'
            data_frame = pd.read_csv(f'{data_path}ETTm1/ETTm1.csv', index_col=0, parse_dates=True,)

            self.data = data_frame[train_start:train_end].values
            valid_data = data_frame[valid_start: valid_end].values
            test_data = data_frame[test_start:test_end].values
            #print(self.data)
            #print(test_data)
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

        self.test_raw=test_data

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
        #entities=os.listdir(f'{data_path}/SMD_raw/train')
        #print(f'Dataset: {entities[entity]}')
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
    elif dataset == 'ETTh1':
        #dataset = ForecastDS(data_path, win_size, step, mode, dataset)
        dataset = Dataset_ETT_hour(data_path,flag=mode, win_size=win_size)
            #root_path, flag = 'train', size = None,
            #features = 'S', data_path = 'ETTm1.csv',
            #target = 'OT', scale = True, inverse = False, timeenc = 0, freq = 't', cols = None
    elif dataset == 'ETTm1':
        dataset = ForecastDS(data_path, win_size, step, mode, dataset)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader





