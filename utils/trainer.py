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

def attention_uniformity(attention_list,args):
    totals=None

    for att in attention_list:
        for head in range(att.size(1)):
            x=(torch.norm(att[:,head, -1, :], dim=1) *
             torch.sqrt(torch.tensor(att[-1,head, -1, :].numel())) - 1) / (
                        torch.sqrt(torch.tensor(att[-1,head, -1, :].numel())) - 1)
            if totals is None:
                totals=x
            else:
                totals+=x
    totals/=(len(attention_list)*att.size(1))
    return totals


def train(model, iterator, optimizer, criterion, device,args,epoch):
    epoch_loss = 0
    model.train()
    losses=[]

    for batch in iterator:
        optimizer.zero_grad()
        data_base, _=batch

        data=torch.clone(data_base)
        if args.forecast:
            data[:,-1:,:]=0
        data=data.to(device)
        data_base=data_base.to(device)

        predictions, _ = model(data )

        sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
        sample_loss = sample_loss.mean(dim=1)
        batch_loss=torch.sum(sample_loss)
        epoch_loss += sum(sample_loss.detach().cpu().numpy())

        batch_loss.backward()
        optimizer.step()

    return epoch_loss / iterator.dataset.__len__()



def validation(model, iterator, optimizer, criterion, device,args, epoch):
    epoch_loss = 0

    model.eval()


    losses=[]
    with torch.no_grad():
        for batch in iterator:

            data_base, _, _=batch

            data=torch.clone(data_base)
            if args.forecast:
                data[:,-1:,:]=0
            data=data.to(device)
            data_base=data_base.to(device)


            predictions, attention_list = model(data, )

            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)
            epoch_loss += sum(sample_loss.detach().cpu().numpy())




    return epoch_loss / iterator.dataset.__len__()


def test_anomaly_detection(model, iterator,val_iterator, criterion, device,args, entity, epoch):

    epoch_loss=0
    batch_num=0

    anomaly_ind=[]
    benign_ind=[]
    sample_loss_dict={}
    sample_attn_dict={}

    val_losses=[]

    preds=[]
    actual=[]
    labels=[]
    with torch.no_grad():
        for batch in val_iterator:
            data_base, label, index = batch
            data = torch.clone(data_base)
            if args.forecast:
                data[:, -1:, :] = 0

            data = data.to(device)
            data_base = data_base.to(device)
            predictions, _ = model(data, )

            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)

            val_losses.extend(sample_loss.cpu().detach().numpy())

        for batch in iterator:
            data_base, label, index = batch
            data = torch.clone(data_base)
            if args.forecast:
                data[:, -1:, :] = 0

            data = data.to(device)
            data_base = data_base.to(device)
            batch_num += 1

            #full loss
            predictions, attention_list = model(data, )


            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)
            epoch_loss += sum(sample_loss.detach().cpu().numpy())

            for i,l in zip(index, sample_loss,):
                sample_loss_dict[i.item()]=l.cpu().detach().numpy()

            #first, specifically look at instances with no anomalies at all
            normal_data=[i for i in range(label.size(0)) if torch.sum(label[i,:])==0 ]
            if len(normal_data)>0:
                benign_ind.extend(index[normal_data].cpu().detach().numpy())

            #examples with anomalies at forecast index
            anomaly_data=[i for i in range(label.size(0)) if label[i,-1]==1 ]
            if len(anomaly_data)>0:
                anomaly_ind.extend(index[anomaly_data].cpu().detach().numpy())

            preds.extend(predictions[:, -1, :].cpu().detach().numpy())
            actual.extend(data_base[:, -1, :].cpu().detach().numpy())
            labels.extend(label.cpu().detach().numpy())


    anomaly_dict={}
    i=0
    for k, g in groupby(enumerate(anomaly_ind), lambda ix : ix[0] - ix[1]):
        anomaly_dict[i]=list(map(itemgetter(1), g))
        i+=1

    anomaly_final_vals=[]
    for key,val in anomaly_dict.items():
        sample_losses=[sample_loss_dict.get(key) for key in val]
        #anomaly_final_vals.append(max(sample_losses))
        anomaly_final_vals.extend([max(sample_losses) for i in range(len(sample_losses))])


    benign_final_vals = [sample_loss_dict.get(key) for key in benign_ind]

    labels=[0 for i in range(len(benign_final_vals))]+[1 for i in range(len(anomaly_final_vals))]



    scores=benign_final_vals+anomaly_final_vals

    if args.save_scores:
        df = pd.DataFrame({'scores': scores, 'labels':labels})
        df.to_csv('output/scores.csv')
    
    '''print(f'ROC: {metrics.roc_auc_score(labels, scores)}')
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    #print(f'PR Curve : {metrics.auc(recall, precision)}')
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    print(f"max_f1_thresh: {max_f1_thresh}")
    print(f"max_f1: {max_f1}")
    print(f'TP: {len([i for i in anomaly_final_vals if i>=max_f1_thresh])} '
          f'TN: {len([i for i in benign_final_vals if i<max_f1_thresh])}, '
          f'FP: {len([i for i in benign_final_vals if i>=max_f1_thresh])}, '
          f'FN: {len([i for i in anomaly_final_vals if i<max_f1_thresh])}')'''


    #threshold=max(val_losses)
    #threshold = np.percentile(val_losses, 1 - iterator.dataset.anomaly_ratio)
    '''scores_with_threshold=(scores>threshold)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores_with_threshold)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    print(f1_scores)
    print(f'TP: {len([i for i in anomaly_final_vals if i>=threshold])} '
          f'TN: {len([i for i in benign_final_vals if i<threshold])}, '
          f'FP: {len([i for i in benign_final_vals if i>=threshold])}, '
          f'FN: {len([i for i in anomaly_final_vals if i<threshold])}')'''
    result, updated_preds = pot_eval(np.array(val_losses), np.array(scores), np.array(labels), args=args)
    print(result)

    '''combined_energy = np.concatenate([val_losses, benign_final_vals,anomaly_final_vals], axis=0)
    anomaly_ratio=len(anomaly_dict.keys())/combined_energy.shape[0]
    print(anomaly_ratio)
    thresh = np.percentile(val_losses, 100 - anomaly_ratio)
    print("Threshold :", thresh)
    pred = (scores > thresh)

    accuracy = accuracy_score(labels, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, pred,
                                                                          average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,recall, f_score))'''

    '''result, updated_preds = pot_eval(np.array(val_losses), np.array(scores), np.array(labels),args=args)

    #result={}
    result['base_roc']=metrics.roc_auc_score(labels, scores)
    result['base_pr']=metrics.auc(recall, precision)
    result['base_max_f1']=max_f1
    result['base_max_f1_threshold']=max_f1_thresh

    result['total_anomalies']=len(anomaly_final_vals)
    result['count_benign_gt_max_f1_th']=len([i for i in benign_final_vals if i>=max_f1_thresh])
    result['count_anomaly_gt_max_f1_th']=len([i for i in anomaly_final_vals if i>=max_f1_thresh])

    print(result)'''

    return epoch_loss / iterator.dataset.__len__()


def test_forecast(model, iterator, val_iterator, criterion, device, args, entity):
    epoch_loss = 0
    batch_num=1
    model.eval()


    preds=[]
    actual=[]

    with torch.no_grad():
        for batch in iterator:
            data_base, label, index = batch

            data = torch.clone(data_base)
            if args.forecast:
                data[:, -1:, :] = 0

            data = data.to(device)
            data_base = data_base.to(device)

            # full loss
            predictions, _ = model(data, )

            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)
            epoch_loss += sum(sample_loss.detach().cpu().numpy())
            batch_num+=1

            preds.extend(predictions[:, -1, :].cpu().detach().numpy())
            actual.extend(data_base[:, -1, :].cpu().detach().numpy())

    if args.save_graphs:
        preds = np.array(preds)
        actual = np.array(actual)
        s = preds.shape[1]

        for x in range(s):
            plt.clf()
            plt.plot([t for t in range(preds.shape[0])], preds[:, x], label='preds')
            plt.plot([t for t in range(actual.shape[0])], actual[:, x], ':', label='actual')
            plt.legend()
            plt.savefig(f'output/forecast_{x}.png')

    return epoch_loss / iterator.dataset.__len__()

