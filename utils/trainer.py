from metrics.accuracy import binary_accuracy
import torch
import matplotlib.pyplot as plt
import numpy as np

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    i=0
    for batch in iterator:
        optimizer.zero_grad()
        data, _=batch
        data=data.to(device)
        i+=1
        predictions = model(data)
        loss = criterion(predictions[:,-1,:], data[:,-1,:])

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #acc = binary_accuracy(predictions, label)
        #epoch_acc += acc.item()

        #if i%500==0:
            #print(i)

    return epoch_loss / len(iterator)



def test(model, iterator, criterion, device,args, epoch):

    def get_loss(data,name, indices=None):
        pred_data=data

        if indices is not None:
            pred_data=data[indices,:,:]
        predictions = model(pred_data)  # .squeeze(1)
        loss = criterion(predictions[:,-1,:], pred_data[:,-1,:])
        if f'{name}_loss' not in loss_dict:
            loss_dict[f'{name}_loss']=0
            loss_dict[f'{name}_count']=0
        loss_dict[f'{name}_loss']+=loss.item()
        loss_dict[f'{name}_count']+=pred_data.size(0)

    def get_graphs(data, name, indices=None):
        pred_data=data
        if indices is not None:
            pred_data=data[indices,:,:]
        predictions = model(pred_data)  # .squeeze(1)
        loss = criterion(predictions[:,-1,:], pred_data[:,-1,:])
        if f'{name}_pred' not in graph_dict:
            graph_dict[f'{name}_pred']=[]
            graph_dict[f'{name}_actual']=[]
        print('here')
        print(predictions[:,-1,:].cpu().detach().numpy().shape)
        graph_dict[f'{name}_pred'].extend(predictions[:,-1,:].cpu().detach().numpy())
        print(np.array(graph_dict[f'{name}_pred']).shape)
        graph_dict[f'{name}_actual'].extend(pred_data[:,-1,:].cpu().detach().numpy())


    loss_dict={}
    graph_dict={}
    model.eval()
    i=0

    with torch.no_grad():
        for batch in iterator:
            data, label = batch
            data = data.to(device)
            i += 1

            #full loss
            get_loss(data, 'epoch', indices=None)

            #first, specifically look at instances with no anomalies at all
            normal_data=[i for i in range(label.size(0)) if torch.sum(label[i,:])==0 ]
            if len(normal_data)>0:
                normal_data=torch.tensor(normal_data)
                get_loss(data, 'benign', indices=normal_data)
                get_graphs(data, 'benign', indices=normal_data)


            #examples with anomalies at forecast index
            anomaly_data=[i for i in range(label.size(0)) if label[i,-1]==1 ]
            if len(anomaly_data)>0:
                anomaly_data=torch.tensor(anomaly_data)
                get_loss(data, 'anomaly_all', indices=anomaly_data)
                get_graphs(data, 'anomaly_all', indices=anomaly_data)

            #anomaly is first in a benign set of time series data of  window size t
            anomaly_first=[i for i in range(label.size(0)) if (label[i,-1]==1 and label[i,-2]==0 and torch.sum(label[i,:])==1) ]
            if len(anomaly_first)>0:
                anomaly_first=torch.tensor(anomaly_first)
                get_loss(data, 'anomaly_first', indices=anomaly_first)
                get_graphs(data, 'anomaly_first', indices=anomaly_first)

    for item in ['benign','anomaly_all','anomaly_first']:
        pred=np.array(graph_dict[f'{item}_pred'])
        actual=np.array(graph_dict[f'{item}_actual'])
        print(item)
        print(pred.shape)
        print(actual.shape)
        continue
        for feat in range(pred.shape[1]):
            plt.clf()
            plt.plot([i for i in range(pred.shape[0])],pred[:,feat], label='pred')
            plt.plot([i for i in range(actual.shape[0])], actual[:, feat], label='actual')
            plt.savefig(f'output/{item}_epoch{epoch}_feat{feat}')
    sys.exit()
    print(f' Val. Losses: ')
    for key, val in loss_dict.items():
        print(f'{key}:  {val}')

    return loss_dict['epoch_loss'] / loss_dict['epoch_count']
    '''preds.extend(predictions[:, -1, :].cpu().detach().numpy())
    actual.extend(data[:,-1,:].cpu().detach().numpy())
    labels.extend(label[:,-1].detach().numpy())
    break
    preds=np.array(preds)
    actual=np.array(actual)
    labels=np.array(labels)
    print(preds.shape)
    print(actual.shape)
    print(labels.shape)

    features=preds.shape[1]
    import os
    print(os.listdir('.'))
    for feat in range(features):
        plt.clf()
        plt.plot([i for i in range(len(labels)) if labels[i]!=1], [preds[i,feat] for i in range(len(labels)) if labels[i]!=1], '.', color='blue')
        plt.plot([i for i in range(len(labels)) if labels[i]==1], [preds[i,feat] for i in range(len(labels)) if labels[i]==1], 'o', color='red')
        plt.savefig(f'output/{args.model_type}_epoch_{epoch}_feature_{feat}.png')
        sys.exit()'''
