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
from torch_geometric.datasets import PPI
from sklearn.metrics import f1_score
import argparse
import copy

parser = argparse.ArgumentParser(description='Training for PPI')
parser.add_argument('-n_l', '--n_layers', type=int, default=3,
                        help='number of convolutional layers (DEFAULT: 3)')
parser.add_argument('-lr', '--learning_rate', type = float, default=0.01,
                        help='learning rate (DEFAULT: 0.01)')
parser.add_argument('-wc', '--weight_decay', type = float, default=0,
                        help='weight decaying parameter (DEFAULT: 0)')
parser.add_argument('-hidden',  '--n_hidden', type = int, default=32,
                        help='number of hidden neurals in each layer (DEFAULT: 32)')
parser.add_argument('-dr',  '--Droprate', type = float, default=0.2,
                        help='probability of retaing at zero (DEFAULT: 0.2)')
parser.add_argument('-nn',  '--nn', type = int, default=20,
                        help='Number of forward propogation to calculate the variance ')
parser.add_argument('-model',  '--model', type = int, default=1,
                        help='1:GCN; 2:GraphSage')
parser.add_argument('-n_epoch', '--num_epoch', type=int, default=200,
                        help='number of epochs (DEFAULT: 200)')
parser.add_argument('-fc', '--fc', type=int, default=0,
                        help='0:fully connected layers; 1:convolutional layers')
parser.add_argument('-Gaus', '--Gaus', type=int, default=1,
                        help='0:|N(0,1)|; 1:N(1,1)')
args = parser.parse_args()

MODEL = {1:GCNConv,2:SAGEConv}
ConvModel = MODEL[args.model]

Gaussian = {0:torch.zeros,1:torch.ones}
GauMean = Gaussian[args.Gaus]
#Load the data

dataset_train = PPI(root='/home/shu30/CS590/PPI')
dataset_val = PPI(root='/home/shu30/CS590/PPI',split="val")
dataset_test = PPI(root='/home/shu30/CS590/PPI',split="test")
    
#Define Dataloader
train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset_val,  batch_size=1, shuffle=True)


# Define the Net to get the dropout heteroscadestic variance
class Net(torch.nn.Module):
    def __init__(self,num_hidden,drop_rate):
        super(Net, self).__init__()
        # if you defined cache=True, the shape of batch must be same!
        self.num_hidden = num_hidden
        self.droprate = drop_rate
        self.conv1 = ConvModel(n_features, self.num_hidden)# if you defined cache=True, the shape of batch must be same!
        self.conv2 = ConvModel(self.num_hidden, self.num_hidden)
        self.conv3 = ConvModel(self.num_hidden, self.num_hidden)
        self.fc1 = Linear(self.num_hidden, self.num_hidden)
        self.fc2 = Linear(self.num_hidden, self.num_hidden)
        self.fc3 = Linear(self.num_hidden, int(dataset_train.num_classes))
    
    def forward(self,data,TRAIN,nn=None):
        if TRAIN == True:
            yy = torch.ones([nn,data.num_nodes,int(dataset_train.num_classes)])
            for ii in range(nn):
                y,edge_index = data.x,data.edge_index
                y = F.relu(self.conv1(y, edge_index))
                y = torch.bernoulli(torch.ones(data.num_nodes,self.num_hidden).to(device)*(1-self.droprate)).mul(y)
                y = F.relu(self.conv2(y, edge_index))
                y = torch.bernoulli(torch.ones(data.num_nodes,self.num_hidden).to(device)*(1-self.droprate)).mul(y)
                y = F.relu(self.conv3(y, edge_index))
                y = torch.bernoulli(torch.ones(data.num_nodes,self.num_hidden).to(device)*(1-self.droprate)).mul(y)
                y = F.relu(self.fc1(y))
                y = F.relu(self.fc2(y))
                y = self.fc3(y)
                y = F.softmax(y, dim=1)
                yy[ii] = y
            variance = torch.exp(torch.var(yy,0).mean(1))
            y,edge_index = data.x,data.edge_index
            y = F.relu(self.conv1(y, edge_index))
            if args.fc == 1:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),variance.reshape(-1,1))).to(device).mul(y)
            y = F.relu(self.conv2(y, edge_index))
            if args.fc == 1:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),variance.reshape(-1,1))).to(device).mul(y)
            y = F.relu(self.conv3(y, edge_index))
            if args.fc == 1:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),variance.reshape(-1,1))).to(device).mul(y)
            y = F.relu(self.fc1(y))
            if args.fc == 0:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),variance.reshape(-1,1))).to(device).mul(y)
            y = F.relu(self.fc2(y))
            if args.fc == 0:
                y = torch.abs(torch.normal(GauMean(y.shape[0],y.shape[1]),variance.reshape(-1,1))).to(device).mul(y)
            y = self.fc3(y)
        else:
            y,edge_index = data.x,data.edge_index
            y = F.relu(self.conv1(y, edge_index))
            y = F.relu(self.conv2(y, edge_index))
            y = F.relu(self.conv3(y, edge_index))
            y = F.relu(self.fc1(y))
            y = F.relu(self.fc2(y))
            y = self.fc3(y)
        return y
    
def train(loader,TRAIN,nn):
    model.train()
    TRAIN = True
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data,TRAIN,nn)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader,TRAIN):
    model.eval()
    TRAIN = False
    ys, preds = [], []
    for data in loader:
        data = data.to(device)
        ys.append(data.y)
        out = model(data,TRAIN)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    return f1_score(y, pred, average='micro')


train_accs = []
val_accs = []
test_accs = []
losses = []


    
# Train the model
n_features = dataset_train.num_node_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args.n_hidden,args.Droprate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
nn = args.nn

for epoch in range(1, args.num_epoch):
    TRAIN = True
    loss = train(train_loader,TRAIN,nn)
    TRAIN = False
    val_f1 = test(val_loader,TRAIN)
    test_f1 = test(test_loader,TRAIN)
    if epoch % 50 == 0:
        print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, val_f1, test_f1))
        
TRAIN = False
train_accs.append(test(train_loader,TRAIN))
val_accs.append(test(val_loader,TRAIN))
test_accs.append(test(test_loader,TRAIN))
print(test(test_loader,TRAIN))
    
    
#np.savetxt("/home/shu30/Research_GraphDropout/AAAI/result/PPI_AHD.txt",test_accs)