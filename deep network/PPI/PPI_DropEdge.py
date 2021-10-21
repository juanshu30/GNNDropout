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
parser.add_argument('-lr', '--learning_rate', type = float, default=0.01,
                        help='learning rate (DEFAULT: 0.01)')
parser.add_argument('-hidden',  '--n_hidden', type = int, default=32,
                        help='number of hidden neurals in each layer (DEFAULT: 32)')
parser.add_argument('-dr',  '--dropout_rate', type = float, default=0.2,
                        help='dropout rate (DEFAULT: 0.2)')
parser.add_argument('-n_epoch', '--num_epoch', type=int, default=200,
                        help='number of epochs (DEFAULT: 200)')
parser.add_argument('-model',  '--model', type = int, default=1,
                        help='1:GCN; 2:GraphSage')
args = parser.parse_args()

MODEL = {1:GCNConv,2:SAGEConv}
ConvModel = MODEL[args.model]

#Load the data

# Define the Net to get the dropout heteroscadestic variance
class Net(torch.nn.Module):
    def __init__(self,num_layers,num_hidden,drop_rate):
        super(Net, self).__init__()
        # if you defined cache=True, the shape of batch must be same!
        
        self.hidden_layers = nn.ModuleList([
            ConvModel(n_features if i==0 else num_hidden, num_hidden)
            for i in range(num_layers)
        ])
        self.fc1 = Linear(num_hidden, num_hidden)
        self.fc2 = Linear(num_hidden, num_hidden)
        self.fc3 = Linear(num_hidden, int(dataset_train.num_classes))
        
        self.dr = drop_rate
        
    def forward(self,data,TRAIN):
        y,edge_index = data.x,data.edge_index
        if TRAIN == True:
            for i, layer in enumerate(self.hidden_layers):
                if i % 4 == 0:
                    edges = edge_index[:,torch.bernoulli(torch.ones([1,edge_index.shape[1]]).to(device) * (1-self.dr)).bool().reshape(-1)]
                    y = F.relu(layer(y, edges))
                else:
                    y = F.relu(layer(y, edges))
        else:
            for i, layer in enumerate(self.hidden_layers):
                y = F.relu(layer(y, edge_index))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y
    
def train(loader,TRAIN):
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data,TRAIN)
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

for i in np.array([8, 16, 32, 64]):
    
    dataset_train = PPI(root='/home/shu30/CS590/PPI')
    dataset_val = PPI(root='/home/shu30/CS590/PPI',split="val")
    dataset_test = PPI(root='/home/shu30/CS590/PPI',split="test")
    
    #Define Dataloader
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset_val,  batch_size=1, shuffle=True)
    
    # Train the model
    n_features = dataset_train.num_node_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(i,args.n_hidden,args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    TRAIN = True
    for epoch in range(1, args.num_epoch):
        TRAIN = True
        loss = train(train_loader, TRAIN)
        TRAIN = False
        val_f1 = test(val_loader, TRAIN)
        test_f1 = test(test_loader, TRAIN)
        if epoch % 50 == 0:
            print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
                epoch, loss, val_f1, test_f1))
        
    TRAIN = False
    train_accs.append(test(train_loader, TRAIN))
    val_accs.append(test(val_loader, TRAIN))
    test_accs.append(test(test_loader, TRAIN))
    print(test(test_loader, TRAIN))
    
    
np.savetxt("/home/shu30/Research_GraphDropout/AAAI/result/GS_PPI_DropEdge.txt",test_accs)