from metrics.accuracy import binary_accuracy
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from itertools import groupby
from operator import itemgetter
import pandas as pd
from metrics.pot.pot import pot_eval

def train(model, iterator, optimizer, criterion, device,args):
    epoch_loss = 0

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

        predictions = model(data, )

        loss = criterion(predictions[:,-1,:], data_base[:,-1,:])

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i%1000==0:
            print(i)
        #print(loss)
    return epoch_loss / iterator.dataset.__len__()



def validation(model, iterator, optimizer, criterion, device,args):
    epoch_loss = 0

    model.eval()
    i=0
    with torch.no_grad():
        for batch in iterator:

            data_base, _=batch

            data=torch.clone(data_base)
            if args.forecast:
                data[:,-1:,:]=0
            data=data.to(device)
            data_base=data_base.to(device)

            i+=1
            predictions = model(data, )

            loss = criterion(predictions[:,-1,:], data_base[:,-1,:])

            epoch_loss += loss.item()
            if i%1000==0:
                print(i)

    return epoch_loss / iterator.dataset.__len__()


def test(model, iterator,val_iterator, criterion, device,args, entity):

    sample_criterion=torch.nn.MSELoss(reduction='none')

    epoch_loss=0
    batch_num=0

    anomaly_ind=[]
    benign_ind=[]
    sample_loss_dict={}

    val_losses=[]

    preds=[]
    actual=[]
    labels=[]
    with torch.no_grad():
        for batch in val_iterator:
            data_base, label = batch
            data = torch.clone(data_base)
            if args.forecast:
                data[:, -1:, :] = 0

            data = data.to(device)
            data_base = data_base.to(device)
            predictions = model(data)
            sample_loss = sample_criterion(predictions[:, -1, :], data_base[:, -1, :])
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
            predictions = model(data)
            loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            epoch_loss+=loss

            sample_loss = sample_criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)

            for i,l in zip(index, sample_loss):
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

    if args.save_graphs:
        preds=np.array(preds)
        actual=np.array(actual)
        s=preds.shape[1]

        for x in range(s):
            plt.clf()
            plt.plot([t for t in range(preds.shape[0])], preds[:,x], label='preds')
            plt.plot([t for t in range(actual.shape[0])], actual[:,x],':', label='actual')
            for a in iterator.dataset.anomalies:
                plt.axvspan(a[0], a[1], facecolor='red', alpha=0.5)

            plt.legend()
            plt.savefig(f'output/{x}_bin.png')


    anomaly_dict={}
    i=0
    for k, g in groupby(enumerate(anomaly_ind), lambda ix : ix[0] - ix[1]):
        anomaly_dict[i]=list(map(itemgetter(1), g))
        i+=1

    anomaly_final_vals=[]
    for key,val in anomaly_dict.items():
        sample_losses=[sample_loss_dict.get(key) for key in val]
        anomaly_final_vals.append(max(sample_losses))

    benign_final_vals = [sample_loss_dict.get(key) for key in benign_ind]
    labels=[0 for i in range(len(benign_final_vals))]+[1 for i in range(len(anomaly_final_vals))]
    scores=benign_final_vals+anomaly_final_vals

    if args.save_scores:
        df = pd.DataFrame({'scores': scores, 'labels':labels})
        df.to_csv('output/scores.csv')
    
    #print(f'ROC: {metrics.roc_auc_score(labels, scores)}')
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    #print(f'PR Curve : {metrics.auc(recall, precision)}')
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    #print(f"max_f1_thresh: {max_f1_thresh}")
    #print(f"max_f1: {max_f1}")

    result, updated_preds = pot_eval(np.array(val_losses), np.array(scores), np.array(labels),args=args)


    '''result['base_roc']=metrics.roc_auc_score(labels, scores)
    result['base_pr']=metrics.auc(recall, precision)
    result['base_max_f1']=max_f1
    result['base_max_f1_threshold']=max_f1_thresh

    result['total_anomalies']=len(anomaly_final_vals)
    result['count_benign_gt_max_f1_th']=len([i for i in benign_final_vals if i>=max_f1_thresh])
    result['count_anomaly_gt_max_f1_th']=len([i for i in anomaly_final_vals if i>=max_f1_thresh])'''
    #result['min_anomaly_loss']=min(anomaly_final_vals)
    #result['max_val_loss']=max(val_losses)
    #result['max_benign_test_loss']=max(benign_final_vals)
    #result['count_benign_gt_max_val']=len([i for i in benign_final_vals if i>max(val_losses)])
    #result['count_anomaly_lt_max_val']=len([i for i in anomaly_final_vals if i<max(val_losses)])
    #result['count_anomaly_gt_max_val']=len([i for i in anomaly_final_vals if i>max(val_losses)])
    #result['benign_loss']=np.mean(np.array(benign_final_vals))
    print(result)

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
            predictions = model(data)

            loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            epoch_loss += loss
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
            plt.savefig(f'output/{x}.png')

    return epoch_loss / iterator.dataset.__len__()

