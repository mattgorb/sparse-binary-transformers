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
    epoch_loss = 0
    normal_ts_loss=0

    model.eval()
    i=0

    with torch.no_grad():
        for batch in iterator:
            data, label = batch
            data = data.to(device)
            i += 1

            #full loss
            predictions = model(data)  # .squeeze(1)
            loss = criterion(predictions[:,-1,:], data[:,-1,:])
            epoch_loss += loss.item()

            #first, specifically look at instances with no anomalies at all
            normal_data=[i for i in range(label.size(0)) if torch.sum(label[i,:])==0 ]
            if len(normal_data)>0:
                normal_data=torch.tensor(normal_data)
                data_normal=data[normal_data, :,:]
                predictions_normal = model(data_normal)
                loss = criterion(predictions_normal[:, -1, :], data_normal[:, -1, :])
                normal_ts_loss+=loss.item()
                sys.exit()
            print('here')
            print(label)
            continue
            #
            x=torch.where(label[:,-1]==1,1,0)
            x=[i for i in range(label.size(0)) if (label[i,-1]==1 and label[i,-2]==0) ]

            if len(x)>0:
                print(torch.tensor(x))
                print(torch.tensor(x).dtype)
                print(label.size())
                print(label[torch.tensor(x),:])
                sys.exit()

    valid_loss=epoch_loss / len(iterator)

    print(f'\t Val. Loss: {valid_loss:.3f} ')


    return epoch_loss / len(iterator)
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
