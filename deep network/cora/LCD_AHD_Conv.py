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
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (DEFAULT: 1)')
parser.add_argument('-n_epoch', '--num_epoch', type=int, default=200,
                        help='number of epochs (DEFAULT: 200)')
parser.add_argument('-model',  '--model', type = int, default=1,
                        help='1:GCN; 2:GraphSage')
parser.add_argument('-Gaus', '--Gaus', type=int, default=1,
                        help='0:|N(0,1)|; 1:N(1,1)')
parser.add_argument('-nn',  '--nn', type = int, default=20,
                        help='Number of forward propogation to calculate the variance ')
args = parser.parse_args()

MODEL = {1:GCNConv,2:SAGEConv}
ConvModel = MODEL[args.model]

Gaussian = {0:torch.zeros,1:torch.ones}
GauMean = Gaussian[args.Gaus]

# Define the Net to get the dropout heteroscadestic variance
class Net(torch.nn.Module):
    def __init__(self,num_layers,num_hidden,dr):
        super(Net, self).__init__()
        self.hidden_layers = nn.ModuleList([
            ConvModel(n_features if i==0 else num_hidden, num_hidden)
            for i in range(num_layers)
        ])
        
        self.fc1 = Linear(num_hidden, num_hidden)
        self.fc2 = Linear(num_hidden, num_hidden)
        self.fc3 = Linear(num_hidden, dataset.num_classes)
        self.dr = dr
        
    def forward(self,TRAIN,nn = None):
        if TRAIN:
            yy = torch.ones([nn,dataset.data.num_nodes,int(dataset.num_classes)])
            for ii in range(nn):
                y,edge_index = data.x,data.edge_index
                for i, layer in enumerate(self.hidden_layers):
                    if i == 0:
                        y1 = F.relu(layer(y, edge_index))
                        y = F.dropout(y1, p=self.dr, training=self.training)
                    else:
                        y2 = F.relu(layer(y, edge_index) + y1)
                        y = F.dropout(y2, p=self.dr, training=self.training)
                y = F.relu(self.fc1(y))
                y = F.dropout(y, p=0.5, training=self.training)
                y = F.relu(self.fc2(y))
                y = F.dropout(y, p=0.5, training=self.training)
                y = self.fc3(y)
                y = F.log_softmax(y, dim=1)
                yy[ii] = y
            variance = torch.exp(torch.var(yy,0).mean(1))
        y,edge_index = data.x,data.edge_index
        if TRAIN:
            for i, layer in enumerate(self.hidden_layers):
                if i == 0:
                    y1 = F.relu(layer(y, edge_index))
                    y = F.dropout(y1, p=self.dr, training=self.training)
                    y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
                else:
                    y2 = F.relu(layer(y, edge_index) + y1)
                    y = F.dropout(y2, p=self.dr, training=self.training)
                    y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),torch.ones(y.shape[0],1))).to(device).mul(y)
        else:
            for i, layer in enumerate(self.hidden_layers):
                if i == 0:
                    y1 = F.relu(layer(y, edge_index))
                    y = F.dropout(y1, p=self.dr, training=self.training)
                else:
                    y2 = F.relu(layer(y, edge_index) + y1)
                    y = F.dropout(y2, p=self.dr, training=self.training)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.fc3(y)
        y = F.log_softmax(y, dim=1)
        return y
    
def train(epoch,loader,n):
    model.train()
    TRAIN = True
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(TRAIN,n)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(data)

def test(loader):
    model.eval()
    TRAIN = False
    correct = np.zeros(dataset.data.x.size(0))
    for i,data in enumerate(loader):
        data = data.to(device)
        output = model(TRAIN)
        loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        pred = output.max(dim=1)[1]
        TRAIN = False
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / sum(data.train_mask).item()
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return loss_val,test_acc

n = args.nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_accs = []
val_accs = []
test_accs = []

best_val_loss = 9999999
test_acc = 0
patience = 200
for i in np.array([8,16,32,64]):
    model = Net(i, args.n_hidden, args.dropout_rate).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.00001)
        
    for epoch in range(1, args.num_epoch):
        TRAIN = True
        loss_tra = train(epoch, loader, n)
        TRAIN = False
        loss_val,acc_test_tmp = test(loader)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            test_acc = acc_test_tmp
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter+=1
        if epoch%50 == 0: 
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
            print(log.format(epoch, loss_tra, loss_val, test_acc))
        if bad_counter == patience:
            break
    log = 'best Epoch: {:03d}, Val loss: {:.4f}, Test acc: {:.4f}'
    print(log.format(best_epoch, best_val_loss, test_acc))
    
#np.savetxt("/home/shu30/Research_GraphDropout/AAAI/result_deep/GS_cora_ADFF.txt",test_accs)
