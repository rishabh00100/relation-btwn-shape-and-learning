import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Linear
from torch.nn import ReLU, Dropout
from torch.nn import Softmax
from torch.nn import BatchNorm1d
import torch.nn.init as init

def xavier(param):
    xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, Linear):
        xavier(m.weight.data)
        m.bias.data.zero_()

class ModularFC(Module):
    def __init__(self, in_dims, hiddenDims, num_classes):
        super(ModularFC, self).__init__()
        self.layers = len(hiddenDims) + 1
        self.linears = ModuleList()

        for i in range(self.layers):
            inDim = hiddenDims[i-1] if i>0 else in_dims
            outDim = hiddenDims[i] if i< (self.layers-1) else num_classes

            self.linears.append(Linear(inDim, outDim))
            # self.linears.append(nn.LayerNorm)
        self.linears.apply(weights_init)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            # x = self.linears[i // 2](x) + l(x)
            x = l(x)
            if i != self.layers-1:
                x = F.relu(x)
        return x

class FC_10(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs, dimList):
        super(FC_10, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, dimList[0])
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # self.bn1 = BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(dimList[0], dimList[1])
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # self.bn2 = BatchNorm1d(900, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = ReLU()
        # third hidden layer
        self.hidden3 = Linear(dimList[1], dimList[2])
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # self.bn3 = BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3 = ReLU()
        # fourth hidden layer
        self.hidden4 = Linear(dimList[2], dimList[3])
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # self.bn4 = BatchNorm1d(700, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act4 = ReLU()
        # fifth hidden layer and output
        self.hidden5 = Linear(dimList[3], dimList[4])
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        # self.bn5 = BatchNorm1d(600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act5 = ReLU()
        # sixth hidden layer and output
        self.hidden6 = Linear(dimList[4], dimList[5])
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        # self.bn6 = BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act6 = ReLU()
        # seventh hidden layer and output
        self.hidden7 = Linear(dimList[5], dimList[6])
        kaiming_uniform_(self.hidden7.weight, nonlinearity='relu')
        # self.bn7 = BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act7 = ReLU()
        # eigth hidden layer and output
        self.hidden8 = Linear(dimList[6], dimList[7])
        kaiming_uniform_(self.hidden8.weight, nonlinearity='relu')
        # self.bn8 = BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act8 = ReLU()
        # nineth hidden layer and output
        self.hidden9 = Linear(dimList[7], dimList[8])
        kaiming_uniform_(self.hidden9.weight, nonlinearity='relu')
        # self.bn9 = BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act9 = ReLU()
        # tenth hidden layer and output
        self.hidden10 = Linear(dimList[8], n_outputs)
        xavier_uniform_(self.hidden10.weight)
        self.act10 = Softmax(dim=1)
        self.dropout = Dropout(0.1, False)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        # X = self.bn1(X)
        X = self.act1(X)
        # X = self.dropout(X)
        # second hidden layer
        X = self.hidden2(X)
        # X = self.bn2(X)
        X = self.act2(X)
        # X = self.dropout(X)
        # third hidden layer
        X = self.hidden3(X)
        # X = self.bn3(X)
        X = self.act3(X)
        X = self.dropout(X)
        # fourth hidden layer
        X = self.hidden4(X)
        # X = self.bn4(X)
        X = self.act4(X)
        # X = self.dropout(X)
        # fifth hidden layer
        X = self.hidden5(X)
        # X = self.bn5(X)
        X = self.act5(X)
        # X = self.dropout(X)
        # sixth hidden layer
        X = self.hidden6(X)
        # X = self.bn6(X)
        X = self.act6(X)
        X = self.dropout(X)
        # seventh hidden layer
        X = self.hidden7(X)
        # X = self.bn7(X)
        X = self.act7(X)
        # X = self.dropout(X)
        # eigth hidden layer
        X = self.hidden8(X)
        # X = self.bn8(X)
        X = self.act8(X)
        X = self.dropout(X)
        # nineth hidden layer
        X = self.hidden9(X)
        # X = self.bn9(X)
        X = self.act9(X)
        X = self.dropout(X)
        # tenth output layer
        X = self.hidden10(X)
        # X = self.act5(X)
        return X
