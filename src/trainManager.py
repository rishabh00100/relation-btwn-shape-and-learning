import torch
import torch.nn.functional as F
from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import numpy as np

def train(model, optimizer, scheduler, loss_fn, train_loader, val_loader, epochs=10, device="cpu"):
    liveloss = PlotLosses()
    lr_list = []
    
    metrics = {}
    metrics['valAccuracy'], metrics['valLogLoss'], metrics['accuracy'], metrics['logLoss'] = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)
    
    for epoch in range(epochs):
        logs = {}
        training_loss = 0.0
        model.train()
        num_train_correct = 0 
        num_train_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            # lr_list.append(optimizer.param_groups[0]["lr"])
            training_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_train_correct += torch.sum(correct).item()
            num_train_examples += correct.shape[0]
        training_loss /= len(train_loader.dataset)
        scheduler.step()
        
        valid_loss, num_correct, num_examples = evaluate_model(model, val_loader, device, loss_fn)
        train_acc = num_train_correct / num_train_examples
        val_acc = num_correct / num_examples
        print('Epoch: {}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train Acc: {:.2f}, Val Acc = {:.2f}'.format(epoch, training_loss, valid_loss, train_acc, val_acc))
        logs['log loss'] = training_loss
        logs['accuracy'] = train_acc
        logs['val_log loss'] = valid_loss
        logs['val_accuracy'] = val_acc
        
        metrics['logLoss'][epoch]       = training_loss
        metrics['accuracy'][epoch]      = train_acc
        metrics['valLogLoss'][epoch]    = valid_loss
        metrics['valAccuracy'][epoch]   = val_acc
        # print(optimizer.param_groups)
        # break
        logs['LR'] = optimizer.param_groups[0]["lr"]
        liveloss.update(logs)
        liveloss.send()
    # plt.plot(lr_list)
    return metrics

def evaluate_model(model, val_loader, device, loss_fn):
    model.eval()
    num_correct = 0 
    valid_loss = 0.0
    num_examples = 0
    for batch in val_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        loss = loss_fn(output,targets) 
        valid_loss += loss.data.item() * inputs.size(0)
        correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    valid_loss /= len(val_loader.dataset)
    return valid_loss, num_correct, num_examples

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat