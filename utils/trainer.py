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
    epoch_acc = 0

    model.eval()
    i=0
    preds=[]
    actual=[]
    labels=[]
    with torch.no_grad():
        for batch in iterator:
            data, label = batch
            data = data.to(device)
            i += 1
            predictions = model(data)  # .squeeze(1)
            loss = criterion(predictions[:,-1,:], data[:,-1,:])

            epoch_loss += loss.item()

            preds.extend(predictions[:, -1, :].cpu().detach().numpy())
            actual.extend(data[:,-1,:].cpu().detach().numpy())
            labels.extend(label.detach().numpy())

    preds=np.array(preds)
    print(preds.shape)

    plt.clf()
    plt.plot([x for x in range(len(preds))], preds, '.')

    plt.savefig(f'output/{args.model_type}_{epoch}.png')
    sys.exit()
    return epoch_loss / len(iterator)