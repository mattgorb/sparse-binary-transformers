from metrics.accuracy import binary_accuracy
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from itertools import groupby
from operator import itemgetter
import pandas as pd

def train(model, iterator, optimizer, criterion, device,args):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    i=0
    for batch in iterator:
        optimizer.zero_grad()
        data_base, _=batch
        data=torch.clone(data_base)
        if args.forecast:
            data[:,-1:,:]=0
        data=data.to(device)
        data_base=data_base.to(device)
        i+=1
        predictions = model(data)

        loss = criterion(predictions[:,-1,:], data_base[:,-1,:])

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i%500==0:
            print(i)

    return epoch_loss / iterator.dataset.__len__()




def test(model, iterator, criterion, device,args, epoch):

    sample_criterion=torch.nn.MSELoss(reduction='none')

    epoch_loss=0
    batch_num=0

    anomaly_ind=[]
    benign_ind=[]
    sample_loss_dict={}

    with torch.no_grad():
        for batch in iterator:
            data_base, label, index = batch
            data = torch.clone(data_base)
            if args.forecast:
                data[:, -1:, :] = 0

            data = data.to(device)
            data_base = data_base.to(device)
            batch_num += 1

            #full loss
            predictions = model(data)
            loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            epoch_loss+=loss

            sample_loss = sample_criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)

            for i,l in zip(index, sample_loss):
                sample_loss_dict[i.item()]=l.item()


            #first, specifically look at instances with no anomalies at all
            normal_data=[i for i in range(label.size(0)) if torch.sum(label[i,:])==0 ]
            if len(normal_data)>0:
                benign_ind.extend(index[normal_data].cpu().detach().numpy())

            #examples with anomalies at forecast index
            anomaly_data=[i for i in range(label.size(0)) if label[i,-1]==1 ]
            if len(anomaly_data)>0:
                anomaly_ind.extend(index[anomaly_data].cpu().detach().numpy())
            if batch_num%500==0:
                print(batch_num)


    anomaly_dict={}
    i=0
    for k, g in groupby(enumerate(anomaly_ind), lambda ix : ix[0] - ix[1]):
        anomaly_dict[i]=list(map(itemgetter(1), g))
        i+=1

    anomaly_final_vals=[]
    for key,val in anomaly_dict.items():
        sample_losses=[sample_loss_dict.get(key) for key in val]
        anomaly_final_vals.append(max(sample_losses))
        #print(key)
        #print(max(sample_losses))
        #print(sample_losses)
    #sys.exit()

    benign_final_vals = [sample_loss_dict.get(key) for key in benign_ind]


    #print(f' Val. Losses: ')
    #for item in ['epoch', 'benign', 'anomaly_all', 'anomaly_first']:
        #print(f"\t{item} avg. Loss {loss_dict[f'{item}_loss']/loss_dict[f'{item}_count']}, \n\tTotal: {loss_dict[f'{item}_loss']}, \n\tCount: {loss_dict[f'{item}_count']}\n")


    print(f'Binary classification scores ')
    #benign=list(sample_loss_dict['benign_sample_loss'])
    #anomaly=list(sample_loss_dict['anomaly_first_sample_loss'])
    labels=[0 for i in range(len(benign_final_vals))]+[1 for i in range(len(anomaly_final_vals))]
    scores=benign_final_vals+anomaly_final_vals

    if args.save_scores:
        df = pd.DataFrame({'scores': scores, 'labels':labels})
        df.to_csv('output/scores.csv')
    #sys.exit()
    
    print(f'ROC: {metrics.roc_auc_score(labels, scores)}')
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    print(f'PR Curve : {metrics.auc(recall, precision)}')
    #print(f'Recall : {recall}')
    #print(f'Precision : {precision}')
    #print(f'F1 : {metrics.f1_score(labels, scores)}')

    #sys.exit()

    return epoch_loss / iterator.dataset.__len__()


'''
def test(model, iterator, criterion, device,args, epoch):

    def get_loss(data,name, indices=None):
        pred_data=data

        if indices is not None:
            pred_data=data_base[indices,:,:]
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
            pred_data=data_base[indices,:,:]
        predictions = model(pred_data)  # .squeeze(1)

        if f'{name}_pred' not in graph_dict:
            graph_dict[f'{name}_pred']=[]
            graph_dict[f'{name}_actual']=[]
        graph_dict[f'{name}_pred'].extend(predictions[:,-1,:].cpu().detach().numpy())
        graph_dict[f'{name}_actual'].extend(pred_data[:,-1,:].cpu().detach().numpy())


    loss_dict={}
    graph_dict={}
    model.eval()
    i=0

    with torch.no_grad():
        for batch in iterator:
            data_base, label, index = batch
            data = torch.clone(data_base)
            data[:, -1:, :] = 0
            data = data.to(device)
            data_base = data_base.to(device)
            i += 1

            #full loss
            get_loss(data, 'epoch', indices=None)

            #first, specifically look at instances with no anomalies at all
            normal_data=[i for i in range(label.size(0)) if torch.sum(label[i,:])==0 ]
            if len(normal_data)>0:
                normal_data=torch.tensor(normal_data)
                get_loss(data, 'benign', indices=normal_data)
                if epoch%5==0: get_graphs(data, 'benign', indices=normal_data)


            #examples with anomalies at forecast index
            anomaly_data=[i for i in range(label.size(0)) if label[i,-1]==1 ]
            if len(anomaly_data)>0:
                anomaly_data=torch.tensor(anomaly_data)
                get_loss(data, 'anomaly_all', indices=anomaly_data)
                if epoch%5==0: get_graphs(data, 'anomaly_all', indices=anomaly_data)

            #anomaly is first in a benign set of time series data of  window size t
            anomaly_first=[i for i in range(label.size(0)) if (label[i,-1]==1 and label[i,-2]==0 and torch.sum(label[i,:])==1) ]
            if len(anomaly_first)>0:
                anomaly_first=torch.tensor(anomaly_first)
                get_loss(data, 'anomaly_first', indices=anomaly_first)
                if epoch%5==0: get_graphs(data, 'anomaly_first', indices=anomaly_first)
                
    if epoch % 5 == 0:
        for item in ['anomaly_first']:
            pred=np.array(graph_dict[f'{item}_pred'])
            actual=np.array(graph_dict[f'{item}_actual'])

            if item=='benign':
                pred=pred[:500,:]
                actual=actual[:500,:]
            for feat in range(pred.shape[1]):
                plt.clf()
                plt.plot([i for i in range(pred.shape[0])],pred[:,feat], label='pred')
                plt.plot([i for i in range(actual.shape[0])], actual[:, feat],':', label='actual')
                plt.legend()
                plt.savefig(f'output/{item}_feat{feat}')



    print(f' Val. Losses: ')
    for item in ['epoch', 'benign', 'anomaly_all', 'anomaly_first']:
        print(f"\t{item} avg. Loss {loss_dict[f'{item}_loss']/loss_dict[f'{item}_count']}, \n\tTotal: {loss_dict[f'{item}_loss']}, \n\tCount: {loss_dict[f'{item}_count']}\n")



    return loss_dict['epoch_loss'] / loss_dict['epoch_count']
    preds.extend(predictions[:, -1, :].cpu().detach().numpy())
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
