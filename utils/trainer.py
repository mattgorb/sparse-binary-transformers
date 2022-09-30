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

    for i, batch in enumerate(iterator):
        if i%50==0:
            print(i)
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
        #if len(val_losses)<500:
        for batch in train_iterator:
            data_base, label = batch
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

    result, updated_preds = pot_eval(np.array(val_losses), np.array(scores), np.array(labels), args=args)


    if result['f1']>best_f1['f1']:
        print(result)
        df = pd.DataFrame({'scores': scores, 'labels':labels})
        df.to_csv(f'/s/luffy/b/nobackup/mgorb/data/ad_results/scores_{args.dataset}_{result["f1"]}_epoch_{epoch}_.csv')


    return result,epoch_loss / iterator.dataset.__len__()


def train_forecast(model, iterator, optimizer, criterion, device, args, epoch):
    epoch_loss = 0
    model.train()

    for i, batch in enumerate(iterator):
        if i % 50 == 0:
            print(i)
        optimizer.zero_grad()
        data_base, labels = batch

        data = torch.clone(data_base)
        if args.forecast:
            data[:, -1:, :] = 0
        data = data.to(device)
        data_base = data_base.to(device)

        predictions, _ = model(data)

        sample_loss = criterion(predictions[:, -1, :], data_base[:, -1, :])
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

def metrics(preds, actual):
    diffs = preds - actual

    se_loss = diffs * diffs

    print('mse')
    mse=torch.mean(se_loss).item()
    print(mse)

    print('mae')
    print(torch.mean(torch.abs(diffs)).item())

    nrmse = torch.sqrt(torch.sum(se_loss) / len(diffs)) / (torch.sum(actual) / len(diffs))
    print("nrmse")
    print(nrmse.item())

    print('quantiles')
    quantile_loss(torch.flatten(actual), torch.flatten(preds), 0.9)
    quantile_loss(torch.flatten(actual), torch.flatten(preds), 0.5)

    return mse


def test_forecast(model, iterator, val_iterator, criterion, device, args, epoch):
    epoch_loss = 0
    batch_num=1
    model.eval()


    preds=[]
    actual=[]

    with torch.no_grad():
        for i,batch in enumerate(iterator):
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
            #epoch_loss += sum(sample_loss.detach().cpu().numpy())
            batch_num+=1

            if i==0:
                preds=predictions[:, -1, :]
                actual=data_base[:, -1, :]
            else:
                preds=torch.cat([preds,predictions[:, -1, :]], dim=0)
                actual=torch.cat([actual,data_base[:, -1, :]], dim=0)

    print('\nstandardized')
    loss1=metrics(preds,actual)

    #print('\nnon standardized')
    #preds=torch.tensor(iterator.dataset.inverse(np.array(preds.detach().cpu().numpy())))
    #actual = torch.tensor(iterator.dataset.inverse(np.array(actual.detach().cpu().numpy())))
    #loss2=metrics(preds,actual)


    preds=torch.tensor(iterator.dataset.inverse(np.array(preds.detach().cpu().numpy())))
    actual = torch.tensor(iterator.dataset.inverse(np.array(actual.detach().cpu().numpy())))
    df = pd.DataFrame(preds.detach().cpu().numpy(), columns = [i for i in range(preds.detach().cpu().numpy().shape[1])])
    df.to_csv(f'/s/luffy/b/nobackup/mgorb/data/forecast_output/{args.dataset}_epoch{epoch}_preds.csv')
    if epoch==0:
        df = pd.DataFrame(actual.detach().cpu().numpy(), columns = [i for i in range(actual.detach().cpu().numpy().shape[1])])
        df.to_csv(f'/s/luffy/b/nobackup/mgorb/data/forecast_output/{args.dataset}_actual.csv')
    #
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

    return loss1

