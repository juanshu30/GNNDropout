## import the packages that might be used
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.data import NeighborSampler
from torch_geometric.datasets import Planetoid
import argparse

dataset = Planetoid(root='/tmp/Cora', name='Cora')
n_features = dataset.num_node_features
loader = DataLoader(dataset, batch_size=1, shuffle=True)
#import copy
#import random
#np.random.seed(1)
#index = random.sample(np.arange(0,2708).tolist(),2708)
#data_lst = []
#for i in range(10):
#    datasets = copy.deepcopy(dataset)
#    mask_train, mask_val, mask_test = torch.zeros(2708).type(torch.bool),torch.zeros(2708).type(torch.bool),torch.zeros(2708).type(torch.bool)
#    if i < 9:
#        mask_val[index[i*270:(i+1)*270]] = 1
#        mask_test[index[(i+1)*270:(i+2)*270]] = 1
#        mask_train[np.delete(index,np.arange(i*270,(i+2)*270,1))] = 1
#        datasets.data.train_mask = mask_train
#        datasets.data.val_mask = mask_val
#        datasets.data.test_mask = mask_test
#        data_lst.append(DataLoader(datasets, batch_size=1, shuffle=True))
#    else:
#        mask_val[index[9*270:]] = 1
#        mask_test[index[0:270]] = 1
#        mask_train[np.delete(index,np.arange(1*270,9*270,1))] = 1
#        datasets.data.train_mask = mask_train
#        datasets.data.val_mask = mask_val
#        datasets.data.test_mask = mask_test
#        data_lst.append(DataLoader(datasets, batch_size=1, shuffle=True))


parser = argparse.ArgumentParser(description='Training for PPI')
parser.add_argument('-n_l', '--n_layers', type=int, default=3,
                        help='number of convolutional layers (DEFAULT: 3)')
parser.add_argument('-lr', '--learning_rate', type = float, default=0.01,
                        help='learning rate (DEFAULT: 0.01)')
parser.add_argument('-wc', '--weight_decay', type = float, default=0,
                        help='weight decaying parameter (DEFAULT: 0)')
parser.add_argument('-hidden',  '--n_hidden', type = int, default=32,
                        help='number of hidden neurals in each layer (DEFAULT: 32)')
args = parser.parse_args()

# Define the Net to get the dropout heteroscadestic variance
class Net(torch.nn.Module):
    def __init__(self,num_layers,num_hidden,dr):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.droprate = dr
        self.conv1 = GCNConv(n_features, 64, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.conv2 = GCNConv(64, 64, cached=False)
        self.fc1 = Linear(64, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, dataset.num_classes)
        
    def forward(self,data,dr=None):
        y,edge_index = data.x,data.edge_index
        y = F.relu(self.conv1(y, edge_index))
        y = F.dropout(y, p=self.droprate, training=self.training)
        y = F.relu(self.conv2(y, edge_index))
        y = F.dropout(y, p=self.droprate, training=self.training)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        #y = F.dropout(y, p=0.5, training=self.training)
        #y = y.mul(torch.bernoulli(torch.ones(data.num_nodes,64).to(device)*0.2))
        y = self.fc3(y)
        y = F.log_softmax(y, dim=1)
        return y
    
def train(epoch,loader,dr):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data,dr)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(data)

def test(loader):
    model.eval()
    correct = np.zeros(dataset.data.x.size(0))
    for i,data in enumerate(loader):
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        TRAIN = False
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / sum(data.train_mask).item()
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return train_acc,val_acc,test_acc


train_accs = []
val_accs = []
test_accs = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in np.arange(0, 1, 0.1):

    model = Net(args.n_layers, args.n_hidden, i).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  
    for epoch in range(1, 201):
        TRAIN = True
        train_loss = train(epoch, loader,i)
        
    TRAIN = False
    train_acc,val_acc,test_acc = test(loader)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    val_accs.append(val_acc)
    print(test_acc)
print(test_accs)
print(val_accs)

np.savetxt("DropNode_test.txt",test_accs)
np.savetxt("DropNode_train.txt",train_accs)