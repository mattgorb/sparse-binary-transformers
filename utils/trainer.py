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
import pandas as pd
from metrics.pot.pot import calc_point2point
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def train(model, iterator, optimizer, criterion, device,args,epoch):
    epoch_loss = 0
    model.train()
    losses=[]

    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        data_base, _=batch
        #if i % 500 == 0:
            #print(i)

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

def anomaly_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds,)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, preds, average='binary')
    #print( "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format( accuracy, precision, recall, f_score))
    tn, fp, fn, tp = confusion_matrix(labels, preds,).ravel()
    #print(f'tp: {tp} tn {tn}, fp {fp} fn {fn}')
    metrics_dict={
        'f1':f_score,
        'precision':precision,
        'recall':recall,
        'accuracy':accuracy,
        'TN':tn,
        'TP':tp,
        'FP':fp,
        'FN':fn
    }
    return metrics_dict


def update_anomaly_preds(labels,preds):
    anomaly_state=False
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if labels[j] == 0:
                    break
                else:
                    if preds[j] == 0:
                        #print('uppdating')
                        preds[j] = 1
            for j in range(i, len(labels)):
                if labels[j] == 0:
                    break
                else:
                    if preds[j] == 0:
                        preds[j] = 1
        elif labels[i] == 0:
            anomaly_state = False
    return preds

def test_anomaly_detection(model, iterator,val_iterator,train_iterator, criterion, device,args, entity, epoch,best_f1):

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
    train_labels = []
    with torch.no_grad():
        for batch in val_iterator:
            data_base, label, index = batch
            data = torch.clone(data_base)

            data = data.to(device)
            data_base = data_base.to(device)
            predictions, _ = model(data, )

            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)

            val_losses.extend(sample_loss.cpu().detach().numpy())
            train_labels.extend(label[:, -1].cpu().detach().numpy())

        for i,batch in enumerate(train_iterator):
            #if i%500==0:
                #print(i)
            data_base, label = batch
            data = torch.clone(data_base)

            data = data.to(device)
            data_base = data_base.to(device)
            predictions, _ = model(data, )

            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)

            val_losses.extend(sample_loss.cpu().detach().numpy())
            train_labels.extend(label[:, -1].cpu().detach().numpy())

        test_losses=[]
        for i,batch in enumerate(iterator):

            data_base, label, index = batch

            data = torch.clone(data_base)

            data = data.to(device)
            data_base = data_base.to(device)
            batch_num += 1

            #full loss
            predictions, attention_list = model(data, )


            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)
            test_losses.extend(sample_loss.cpu().detach().numpy())
            epoch_loss += sum(sample_loss.detach().cpu().numpy())

            for i,l in zip(index, sample_loss,):
                sample_loss_dict[i.item()]=l.cpu().detach().numpy()

            #first, specifically look at instances with no anomalies at all
            normal_data=[i for i in range(label.size(0)) if torch.sum(label[i,:])==0 ]
            if len(normal_data)>0:
                benign_ind.extend(index[normal_data].cpu().detach().numpy())

            #examples with anomalies at final index
            anomaly_data=[i for i in range(label.size(0)) if label[i,-1]==1 ]
            if len(anomaly_data)>0:
                anomaly_ind.extend(index[anomaly_data].cpu().detach().numpy())


            labels.extend(label[:, -1].cpu().detach().numpy())

    if args.dataset=='SMD':
        r=0.995
    else:
        r=0.99
    scores_threshold=np.quantile(np.concatenate([np.array(val_losses), np.array(test_losses)], axis=0), r)
    scores_manual_threshold=(test_losses>scores_threshold)
    scores_manual_threshold=update_anomaly_preds(labels, scores_manual_threshold)
    metrics_manual_threshold=anomaly_metrics(labels, scores_manual_threshold)
    metrics_manual_threshold[f'threshold_{r}']=scores_threshold
    print('Manual Threshold')
    print(metrics_manual_threshold)

    test_losses_cleaned = [test_losses[i] for i in benign_ind+anomaly_ind]
    labels_cleaned = [labels[i] for i in benign_ind+anomaly_ind]

    result, updated_preds = pot_eval(np.array(val_losses), np.array(test_losses_cleaned), np.array(labels_cleaned), args=args)
    print("POT method cleaned")
    print(result)


    scores_threshold=np.quantile(np.concatenate([np.array(val_losses), np.array(test_losses_cleaned)], axis=0), r)
    scores_manual_threshold_cleaned=(test_losses_cleaned>scores_threshold)
    scores_manual_threshold_cleaned=update_anomaly_preds(labels_cleaned, scores_manual_threshold_cleaned)
    metrics_manual_threshold_cleaned=anomaly_metrics(labels_cleaned, scores_manual_threshold_cleaned)
    metrics_manual_threshold_cleaned[f'threshold_{r}']=scores_threshold
    print('Manual Threshold Cleaned')
    print(metrics_manual_threshold_cleaned)




    df = pd.DataFrame({'scores': scores_manual_threshold, 'labels':labels})
    df.to_csv(f'/s/luffy/b/nobackup/mgorb/data/ad_results/scores_full_{args.dataset}_type_{args.model_type}_pr_{args.lin_prune_rate}.csv')
    df = pd.DataFrame({'scores_manual_threshold_cleaned': scores_manual_threshold_cleaned,'scores_POT':updated_preds, 'labels':labels_cleaned})
    df.to_csv(f'/s/luffy/b/nobackup/mgorb/data/ad_results/scores_{args.dataset}_type_{args.model_type}_pr_{args.lin_prune_rate}_epoch_{epoch}.csv')

    return metrics_manual_threshold_cleaned,epoch_loss / iterator.dataset.__len__()


def train_forecast(model, iterator, optimizer, criterion, device, args, epoch):
    epoch_loss = 0
    model.train()
    model.float()
    for i, batch in enumerate(iterator):

        optimizer.zero_grad()
        data_base, labels = batch

        data = torch.clone(data_base)
        if args.forecast:
            data[:, -args.forecasting_steps:, :] = 0
        data = data.to(device)
        data_base = data_base.to(device)
        predictions, _ = model(data)

        sample_loss = criterion(predictions[:, -args.forecasting_steps:, :], data_base[:, -args.forecasting_steps:, :])
        sample_loss = sample_loss.mean(dim=1)
        batch_loss = torch.sum(sample_loss)
        epoch_loss += torch.sum(sample_loss)

        batch_loss.backward()
        optimizer.step()

    return epoch_loss.item()/iterator.dataset.__len__()

def quantile_loss(labels, mu, quantile):
    I = (labels >= mu).float()
    diff = 2*(torch.sum(quantile*((labels-mu)*I)+ (1-quantile) *(mu-labels)*(1-I))).item()
    denom = torch.sum(torch.abs(labels)).item()
    q_loss = diff/denom
    print(q_loss)

def metrics(preds, actual,iterator):
    diffs = preds - actual

    se_loss = diffs * diffs

    mse=torch.mean(se_loss).item()
    print(f'Test set MSE: {mse}')

    print(f'Test set MAE: {torch.mean(torch.abs(diffs)).item()}')
    return mse


def test_forecast(model, iterator, val_iterator, criterion, device, args, epoch):
    epoch_loss = 0
    batch_num=1
    model.eval()


    preds=[]
    actual=[]

    with torch.no_grad():
        for i,batch in enumerate(iterator):
            data_base, label = batch

            data = torch.clone(data_base)
            if args.forecast:
                data[:, -1:, :] = 0

            data = data.to(device)
            data_base = data_base.to(device)

            # full loss
            predictions, _ = model(data, )


            sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
            sample_loss = sample_loss.mean(dim=1)

            batch_num+=1

            if i==0:
                preds=predictions[:, -1, :]
                actual=data_base[:, -1, :]
            else:
                preds=torch.cat([preds,predictions[:, -1, :]], dim=0)
                actual=torch.cat([actual,data_base[:, -1, :]], dim=0)

    #print('\nstandardized')
    loss1=metrics(preds,actual,iterator)




    preds=torch.tensor(iterator.dataset.inverse(np.array(preds.detach().cpu().numpy())))
    actual = torch.tensor(iterator.dataset.inverse(np.array(actual.detach().cpu().numpy())))
    df = pd.DataFrame(preds.detach().cpu().numpy(), columns = [i for i in range(preds.detach().cpu().numpy().shape[1])])
    df.to_csv(f'/s/lovelace/c/nobackup/iray/mgorb/data/forecast_output/{args.dataset}_epoch{epoch}_preds_model_type_{args.model_type}.csv')
    if epoch==0:
        df = pd.DataFrame(actual.detach().cpu().numpy(), columns = [i for i in range(actual.detach().cpu().numpy().shape[1])])
        df.to_csv(f'/s/lovelace/c/nobackup/iray/mgorb/data/forecast_output/{args.dataset}_actual_model_type_{args.model_type}.csv')
    #

    return loss1















def metrics_lt(preds, actual,iterator):
    diffs = preds - actual

    se_loss = diffs * diffs

    mse=torch.mean(se_loss).item()
    print(f'Test set MSE: {mse}')

    mae=torch.mean(torch.abs(diffs)).item()
    print(f'Test set MAE: {mae}')
    return mse, mae



def train_lt_forecast(model, iterator, optimizer, criterion, device, args, epoch):
    epoch_loss = 0
    model.train()
    model.float()
    for i, batch in enumerate(iterator):

        optimizer.zero_grad()
        data_base, labels = batch

        data = torch.clone(data_base)
        if args.forecast:
            data[:, -args.forecasting_steps:, :] = 0
        data = data.to(device)
        data_base = data_base.to(device)
        predictions, _ = model(data,has_src_mask=False)

        sample_loss = criterion(predictions[:, -args.forecasting_steps:, :], data_base[:, -args.forecasting_steps:, :])
        sample_loss = sample_loss.mean(dim=1)
        batch_loss = torch.sum(sample_loss)
        epoch_loss += torch.sum(sample_loss)

        batch_loss.backward()
        optimizer.step()

    return epoch_loss.item()/iterator.dataset.__len__()



def test_lt_forecast(model, iterator, val_iterator, criterion, device, args, epoch):
    epoch_loss = 0
    batch_num=1
    model.eval()


    preds=[]
    actual=[]

    with torch.no_grad():
        for i,batch in enumerate(iterator):
            data_base, label = batch

            data = torch.clone(data_base)
            if args.forecast:
                data[:, -args.forecasting_steps::, :] = 0
                

            data = data.to(device)
            data_base = data_base.to(device)

            # full loss
            predictions, _ = model(data,has_src_mask=False )

            sample_loss = criterion(predictions[:, -args.forecasting_steps:, :], data_base[:, -args.forecasting_steps:, :])
            sample_loss = sample_loss.mean(dim=1)

            batch_num+=1

            if i==0:
                preds=predictions[:, -args.forecasting_steps:, :]
                actual=data_base[:, -args.forecasting_steps:, :]
            else:
                preds=torch.cat([preds,predictions[:, -args.forecasting_steps:, :]], dim=0)
                actual=torch.cat([actual,data_base[:, -args.forecasting_steps:, :]], dim=0)

    #print('\nstandardized')
    mse, mae=metrics_lt(preds,actual,iterator)





    return mse, mae