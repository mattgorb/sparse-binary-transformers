from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.train_util import adjust_learning_rate, EarlyStopping
from sklearn import metrics
from metrics.pot.pot import pot_eval
from itertools import groupby
from operator import itemgetter

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)




def validation(model, vali_loader, optimizer, criterion, device,args, epoch):
    model.eval()

    loss_1 = []
    loss_2 = []
    for i, (input_data, _) in enumerate(vali_loader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           args.window_size)).detach())) + torch.mean(
                my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)).detach(),
                    series[u])))
            prior_loss += (torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.window_size)),
                           series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(),
                           (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.window_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)

        rec_loss = criterion(output[:,-1,:], input[:,-1,:])
        loss_1.append((rec_loss - args.k * series_loss).item())
        loss_2.append((rec_loss + args.k * prior_loss).item())

    return np.average(loss_1)#, np.average(loss_2)


def train(model, train_loader, optimizer, criterion, device,args,epoch):
    #print("======================TRAIN MODE======================")

    time_now = time.time()

    early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=args.dataset)
    train_steps = len(train_loader)

    model.train()
    iter_count = 0
    loss1_list = []
    for i, (input_data, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        iter_count += 1
        input = input_data.float().to(args.device)

        output, series, prior, _ = model(input)

        # calculate Association discrepancy
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           args.window_size)).detach())) + torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.window_size)).detach(),
                           series[u])))

            prior_loss += (torch.mean(my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        args.window_size)),
                series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)))))
        series_loss = series_loss / len(series)
        prior_loss = prior_loss / len(prior)

        rec_loss = criterion(output[:,-1,:], input[:,-1,:])

        loss1_list.append((rec_loss - args.k * series_loss).item())
        loss1 = rec_loss - args.k * series_loss
        loss2 = rec_loss + args.k * prior_loss

        if (i + 1) % 100 == 0:
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        # Minimax strategy
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()

    #print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(loss1_list)



    return train_loss


def test(model, test_dataloader,val_dataloader,train_dataloader, criterion, device, args, ent, epoch):
    '''self.model.load_state_dict(
        torch.load(
            os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))'''
    model.eval()
    temperature = 50

    print("======================TEST MODE======================")

    criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    attens_energy = []
    for i, (input_data, labels) in enumerate(train_dataloader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)
        loss = torch.mean(criterion(input[:,-1,:], output[:,-1,:]), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)),
                    series[u].detach()) * temperature

        metric =torch.squeeze( torch.softmax((-series_loss - prior_loss), dim=-1))
        cri = metric * loss

        cri = cri.detach().cpu().numpy()
        attens_energy.extend(cri)

    #attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # (2) find the threshold
    attens_energy = []
    for i, (input_data, labels) in enumerate(val_dataloader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)

        loss = torch.mean(criterion(input[:,-1,:], output[:,-1,:]), dim=-1)

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)),
                    series[u].detach()) * temperature
        # Metric
        metric =torch.squeeze( torch.softmax((-series_loss - prior_loss), dim=-1))
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.extend(cri)

    #attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    val_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, val_energy], axis=0)


    # (3) evaluation on the test set
    test_labels = []
    attens_energy = []

    anomaly_ind=[]
    benign_ind=[]
    sample_loss_dict={}
    for a, (input_data, labels,index) in enumerate(test_dataloader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)

        loss = torch.mean(criterion(input[:,-1,:], output[:,-1,:]), dim=-1)


        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.window_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.window_size)),
                    series[u].detach()) * temperature

        metric =torch.squeeze( torch.softmax((-series_loss - prior_loss), dim=-1))
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.extend(cri)
        test_labels.extend(labels[:,-1].cpu().numpy())

        for j, l in zip(index, cri):
            sample_loss_dict[j.item()] = l#.cpu().detach().numpy()

        # first, specifically look at instances with no anomalies at all
        normal_data = [i for i in range(labels.size(0)) if torch.sum(labels[i, :]) == 0]
        if len(normal_data) > 0:
            benign_ind.extend(index[normal_data].cpu().detach().numpy())

        # examples with anomalies at forecast index
        anomaly_data = [i for i in range(labels.size(0)) if labels[i, -1] == 1]
        if len(anomaly_data) > 0:
            anomaly_ind.extend(index[anomaly_data].cpu().detach().numpy())

    test_energy=np.array(attens_energy)

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


    combined_energy = np.concatenate([combined_energy, benign_final_vals,anomaly_final_vals], axis=0)
    anomaly_ratio=len(anomaly_dict.keys())/combined_energy.shape[0]
    print(anomaly_ratio)

    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
    print("Threshold :", thresh)

    labels=[0 for i in range(len(benign_final_vals))]+[1 for i in range(len(anomaly_final_vals))]
    scores=benign_final_vals+anomaly_final_vals

    pred = (scores > thresh)

    accuracy = accuracy_score(labels, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, pred,
                                                                          average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,recall, f_score))

    #sys.exit()


    '''anomaly_dict={}
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

    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    #print(f'PR Curve : {metrics.auc(recall, precision)}')
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    print(f"max_f1_thresh: {max_f1_thresh}")
    print(f"max_f1: {max_f1}")

    result, updated_preds = pot_eval(np.array(val_energy), np.array(scores), np.array(labels),args=args)
    print(result)'''

    print('energies train val test:')
    print(np.mean(train_energy))
    print(np.mean(val_energy))
    print(np.mean(test_energy))
    #result={}
    '''result['base_roc']=metrics.roc_auc_score(labels, scores)
    result['base_pr']=metrics.auc(recall, precision)
    result['base_max_f1']=max_f1
    result['base_max_f1_threshold']=max_f1_thresh

    result['total_anomalies']=len(anomaly_final_vals)
    result['count_benign_gt_max_f1_th']=len([i for i in benign_final_vals if i>=max_f1_thresh])
    result['count_anomaly_gt_max_f1_th']=len([i for i in anomaly_final_vals if i>=max_f1_thresh])

    print(result)'''
    #sys.exit()
    return None