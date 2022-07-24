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

        if i%500==0:
            print(i)

    return epoch_loss / len(iterator)



def test(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            label, text = batch
            label = label.to(device)
            text = text.to(device)
            predictions = model(text).squeeze(1)

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator)