from metrics.accuracy import binary_accuracy
import torch

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
        predictions = model(data)#.squeeze(1)
        loss = criterion(predictions, data)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #acc = binary_accuracy(predictions, label)
        #epoch_acc += acc.item()

        #if i%500==0:
            #print(i)

    return epoch_loss / len(iterator)



def test(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            data, _ = batch
            data = data.to(device)
            i += 1
            predictions = model(data)  # .squeeze(1)
            loss = criterion(predictions, data)

            epoch_loss += loss.item()


    return epoch_loss / len(iterator)