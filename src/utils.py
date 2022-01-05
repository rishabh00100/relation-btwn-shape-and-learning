import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# prepare the dataset
def prepare_data(dataset, trainBatchSize, valBatchSize, testBatchSize):
    # load the dataset
    if dataset.lower() == 'helena':
        from src.helenaDataset import helenaDataset
        path = '/content/gdrive/MyDrive/Sem1/CMPSCI_682/main_project/inputs/HelenaData.csv'
        dataset = helenaDataset(path)
    elif dataset == 'helenaYug':
        from src.helenaDataset import helenaDataset
        path = '/content/drive/My Drive/NeuralNets HW1/main_project/inputs/HelenaData.csv'
        dataset = helenaDataset(path)
    elif dataset.lower() == 'cnae-9':
        from src.cnaeDataset import cnae9Dataset
        path = '/content/drive/My Drive/NeuralNets HW1/main_project/inputs/cnae.csv'
        dataset = cnae9Dataset(path)
    elif dataset.lower() == 'cnae-9-rishabh':
        from src.cnaeDataset import cnae9Dataset
        path = '/content/gdrive/MyDrive/Sem1/CMPSCI_682/main_project/inputs/cnae.csv'
        dataset = cnae9Dataset(path)
    elif dataset == 'minibooneYug':
        from src.minibooneDataset import minibooneDataset
        path = '/content/drive/My Drive/NeuralNets HW1/main_project/inputs/MiniBooNE.csv'
        dataset = minibooneDataset(path)
        print('correctly in miniboone')
    elif dataset == 'miniBooneRis':
        from src.minibooneDataset import minibooneDataset
        path = '/content/gdrive/MyDrive/Sem1/CMPSCI_682/main_project/inputs/MiniBooNE.csv'
        dataset = minibooneDataset(path)
        print('correctly in miniboone')
    # calculate split
    print(type(dataset), dataset)
    train, val, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=trainBatchSize, shuffle=True)
    val_dl = DataLoader(val, batch_size=valBatchSize, shuffle=True)
    test_dl = DataLoader(test, batch_size=testBatchSize, shuffle=False)
    return train_dl, val_dl, test_dl

def compute_saliency_maps(X, y, model, device):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    
    saliency = None
    out = model(X)  # Forward pass

    # Select scores belonging to correct class
    scores = out.gather(1, y.view(-1, 1)).squeeze()
    backward_tensor = torch.FloatTensor([1.0 for i in range(y.shape[0])])
    scores.backward(backward_tensor.to(device)) # Backward pass

    saliency = X.grad.data
    saliency = saliency.abs() # Absolute of vales
    return saliency

def get_final_dl(mode='test', data='helena'):
    train_dl, val_dl, test_dl = prepare_data(data, 512, 512, 512)
    # model.to(device)

    # Select dataset for operations
    if mode == 'train':
        final_dl = train_dl
    elif mode == 'val':
        final_dl = val_dl
    elif mode == 'test':
        final_dl = test_dl
    return final_dl


def compute_saliency_for_features(model, normalize=True, mode='test', data='helena', num_features=27, num_classes=100):
    model_outputs = {'saliency_list':[], 'target_list':[], 'output_list': []}
    device = get_default_device()
    final_dl = get_final_dl(mode=mode, data=data)
    # Calc saliency maps for each data point
    for batch in final_dl:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets).view(-1)
        # print(outputs, correct)
        # break
        model_outputs['target_list'].append(targets)
        model_outputs['output_list'].append(correct.cpu().detach().numpy())
        model_outputs['saliency_list'].append(compute_saliency_maps(inputs, targets, model, device))

    # Aggregate the saliency at class level for correct outputs
    agg_saliency = torch.zeros(num_classes, num_features).to(device)
    counter = 0
    class_counts = [0 for i in range(num_classes)]
    for saliency, outputs, targets in zip(model_outputs['saliency_list'], model_outputs['output_list'], model_outputs['target_list']):
        for s, o, t in zip(saliency, outputs, targets):
            # correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            # print(o)
            # break
            if o == True:
                agg_saliency[t.item(),:] += s
                class_counts[t.item()] += 1

    # print(class_counts)
    # Take the average by number of examples per class
    avg_saliency = []
    for idx, row in enumerate(agg_saliency):
        if class_counts[idx] != 0:
            avg_saliency.append(row.cpu().detach().numpy()/class_counts[idx])
        else:
            avg_saliency.append(row.cpu().detach().numpy())

    if normalize == True:
        # import numpy as np
        # Divide each row by its sum. Basically l1 norm
        avg_saliency = np.array(avg_saliency)
        row_sums = avg_saliency.sum(axis=1)
        avg_saliency = avg_saliency / row_sums[:, np.newaxis]
    return avg_saliency

def compute_model_activations(model, mode='test', normalize=True, num_classes=100, data='cnae-9-rishabh'):
    device = get_default_device()
    final_dl = get_final_dl(mode=mode, data=data)

    # Forward propogate through batches of final_dl
    # And collect weight activations of each neuron
    wb_dict = {}
    for name, param in model.named_parameters():
        wb_dict[str(name)] = param

    activation_list = []
    for inputs, targets in final_dl:
        inputs = inputs.to(device)
        targets = targets.to(device)
        Y = model(inputs)
        correct = torch.eq(torch.max(F.softmax(Y, dim=1), dim=1)[1], targets).view(-1)
        # print("correct", correct)
        # print(inputs[correct])
        # print(torch.sum(correct).item(), inputs[correct].shape, targets[correct].shape)
        # break
        inputs = inputs[correct]

        output = None
        for i in range(model.layers-1):
            weights = wb_dict['linears.{}.weight'.format(i)]
            bias = wb_dict['linears.{}.bias'.format(i)]
            # print(weights.T.shape, inputs.shape, bias.shape)
            out = torch.mm(inputs, weights.T) + bias
            out = F.relu(out)
            inputs = out
            if output is None:
                output = out
            else:
                output = torch.cat((output, out), 1)
        activation_list.append([output, targets[correct]])

    # Calc avg activation
    avg_activation = {}
    for activations, target_classes in activation_list:
      for activation, target_class in zip(activations, target_classes):
        target_class = target_class.item()
        if target_class in avg_activation.keys():
          # print(activation, target_class)
          avg_activation[target_class]['activation'] += activation.cpu().detach().numpy()
          avg_activation[target_class]['counts'] += 1
        else:
          avg_activation[target_class] = {'activation': activation.cpu().detach().numpy(), 'counts': 1}

    final_avg_activation = []
    for key in range(num_classes):
      if key in avg_activation.keys():
        avg_activation[key]['activation'] /= avg_activation[key]['counts']
        if normalize == True:
          row_sum = np.sum(avg_activation[key]['activation'])
          avg_activation[key]['activation'] /= row_sum
          final_avg_activation.append(avg_activation[key]['activation'])
      else:
        final_avg_activation.append(np.full(avg_activation[0]['activation'].shape, np.nan))


    return final_avg_activation