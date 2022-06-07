## import the packages that might be used
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.data import NeighborSampler
from torch_geometric.datasets import Planetoid
import argparse

dataset = Planetoid(root='/tmp/Cora', name='Cora')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
n_features = dataset.num_node_features


parser = argparse.ArgumentParser(description='Training for PPI')
parser.add_argument('-n_l', '--n_layers', type=int, default=3,
                        help='number of convolutional layers (DEFAULT: 3)')
parser.add_argument('-lr', '--learning_rate', type = float, default=0.01,
                        help='learning rate (DEFAULT: 0.01)')
parser.add_argument('-wc', '--weight_decay', type = float, default=0,
                        help='weight decaying parameter (DEFAULT: 0)')
parser.add_argument('-hidden',  '--n_hidden', type = int, default=32,
                        help='number of hidden neurals in each layer (DEFAULT: 32)')
parser.add_argument('-model',  '--model', type = int, default=1,
                        help='1:GCN; 2:GraphSage')
parser.add_argument('-fc', '--fc', type=int, default=0,
                        help='0:fully connected layers; 1:convolutional layers')
parser.add_argument('-n_epoch', '--num_epoch', type=int, default=200,
                        help='number of epochs (DEFAULT: 200)')
parser.add_argument('-Gaus', '--Gaus', type=int, default=1,
                        help='0:|N(0,1)|; 1:N(1,1)')
args = parser.parse_args()

MODEL = {1:GCNConv,2:SAGEConv}
ConvModel = MODEL[args.model]

Gaussian = {0:torch.zeros,1:torch.ones}
GauMean = Gaussian[args.Gaus]

# Define the Net to get the dropout heteroscadestic variance
class Net(torch.nn.Module):
    def __init__(self,n_hidden):
        super(Net, self).__init__()
        self.num_hidden = n_hidden
        self.conv1 = ConvModel(n_features, self.num_hidden) # if you defined cache=True, the shape of batch must be same!
        self.conv2 = ConvModel(self.num_hidden, self.num_hidden)
        self.conv3 = ConvModel(self.num_hidden, self.num_hidden)
        self.fc1 = Linear(self.num_hidden, self.num_hidden)
        self.fc2 = Linear(self.num_hidden, self.num_hidden)
        self.fc3 = Linear(self.num_hidden, dataset.num_classes)
        
    def forward(self,TRAIN,dr=None):
        y,edge_index = data.x,data.edge_index
        y = F.relu(self.conv1(y, edge_index))
        if TRAIN == True:
            if args.fc == 1:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
            y = F.relu(self.conv2(y, edge_index))
            if args.fc == 1:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
            y = F.relu(self.conv3(y, edge_index))
            if args.fc == 1:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
            y = F.relu(self.fc1(y))
            if args.fc == 0:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
            y = F.relu(self.fc2(y))
            if args.fc == 0:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
            y = self.fc3(y)
            y = F.log_softmax(y, dim=1)
        else:
            y = F.relu(self.conv2(y, edge_index))
            y = F.relu(self.conv3(y, edge_index))
            y = F.relu(self.fc1(y))
            y = F.relu(self.fc2(y))
            y = self.fc3(y)
            y = F.log_softmax(y, dim=1)
        return y
    
def train(epoch,loader,TRAIN):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(TRAIN)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(data)

def test(loader,TRAIN):
    model.eval()
    correct = np.zeros(dataset.data.x.size(0))
    for i,data in enumerate(loader):
        TRAIN = True
        data = data.to(device)
        output = model(TRAIN)
        pred = output.max(dim=1)[1]
        TRAIN = False
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / sum(data.train_mask).item()
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return train_acc,val_acc,test_acc




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accs_mean = []
accs_var = []

train_accs = []
val_accs = []
test_accs = []
for j in range(2):
    model = Net(args.n_hidden).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(1, args.num_epoch):
        TRAIN = True
        train_loss = train(epoch, loader, TRAIN)
        
    TRAIN = False
    train_acc,val_acc,test_acc = test(loader,TRAIN)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    val_accs.append(val_acc)
    print(test_acc)
accs_mean.append(np.mean(test_accs))
accs_var.append(np.var(test_accs))
    
  
