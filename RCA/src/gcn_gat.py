# We assume that PyTorch is already installed
import torch

# Numpy for matrices
import numpy as np
np.random.seed(0)

# Visualization
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_undirected
from metrics_pytorch import evaluate_metrics




import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv

class GraphCons(torch.nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(GraphCons, self).__init__()
        self.nnodes = nnodes

        
        self.emb1 = torch.nn.Embedding(nnodes, dim)
        self.emb2 = torch.nn.Embedding(nnodes, dim)
        self.lin1 = torch.nn.Linear(dim,dim)
        self.lin2 = torch.nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0))
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out, num_nodes, batch_size):
    super().__init__()
    self.num_nodes = num_nodes
    self.gc = GraphCons(
        num_nodes, 
        num_nodes, 
        dim=40,
        )
    
    self.batch_size = batch_size
    self.repeat_range = int((self.batch_size-1) * num_nodes)
    self.idx = torch.arange(num_nodes)

    self.gcn1 = GCNConv(dim_in, dim_h)
    #self.gcn1.lin.weight = self.gcn1.lin.weight.float()
    #self.gcn1.lin.bias = self.gcn1.lin.bias.float()

    self.gcn2 = GCNConv(dim_h, dim_out)

  def forward(self, x):
    # MTGNN graph construction
    x = x.float()
    edge_index = self.gc(self.idx)
    # Convert adjacency matrix to edge list of sparse tensor
    edge_index = edge_index.nonzero().t().contiguous()
    # Create the repeating increasing graph
    repeating_tensor = torch.arange(0,self.repeat_range+1,self.num_nodes) 
    
    # Add a dimension to the tensor
    #repeating_tensor = repeating_tensor.unsqueeze(1)
    repeating_tensor = repeating_tensor.repeat_interleave(edge_index.shape[1])
    repeating_tensor = repeating_tensor.reshape(1,-1)
    repeating_tensor = repeating_tensor.repeat(2,1)    
    edge_index = edge_index.repeat(1, self.batch_size)

    # Add the repeating tensor to the edge_index
    edge_index = edge_index + repeating_tensor  
    
    #h = F.dropout( x, p=0.5, training=self.training)
    h = self.gcn1(x, edge_index)
    #print(h.shape)
    #print(asd)
    h = torch.relu(h)
    #h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    
    h = F.softmax(h, dim=1)
    return h


class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, num_nodes, batch_size, heads=8):
    super().__init__()

    self.num_nodes = num_nodes
    self.gc = GraphCons(
        num_nodes, 
        num_nodes, 
        dim=40,
        )
    
    self.batch_size = batch_size
    self.repeat_range = int((self.batch_size-1) * num_nodes)
    self.idx = torch.arange(num_nodes)


    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, x):

    x = x.float()
    edge_index = self.gc(self.idx)
    # Convert adjacency matrix to edge list of sparse tensor
    edge_index = edge_index.nonzero().t().contiguous()
    # Create the repeating increasing graph
    repeating_tensor = torch.arange(0,self.repeat_range+1,self.num_nodes) 
    
    # Add a dimension to the tensor
    #repeating_tensor = repeating_tensor.unsqueeze(1)
    repeating_tensor = repeating_tensor.repeat_interleave(edge_index.shape[1])
    repeating_tensor = repeating_tensor.reshape(1,-1)
    repeating_tensor = repeating_tensor.repeat(2,1)    
    edge_index = edge_index.repeat(1, self.batch_size)

    # Add the repeating tensor to the edge_index
    edge_index = edge_index + repeating_tensor  


    #h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.relu(h)
    #h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    h = F.softmax(h, dim=1)
    #h = F.log_softmax(h, dim=1)
    return h

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200

    model.train()
    
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        print(data.x.shape)
        print(data.x)
        print(data.edge_index)
        print(data.edge_index.shape)
        _, out = model(data.x, data.edge_index)
        print(out)
        print(data.y)
        print(asd)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

    return model

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

import pandas as pd
NUM_NODES = 4

# Open a comma separated values file
df = pd.read_csv('../data/calibrated_multi.txt', sep=',') 

# Show the first 5 rows

# Set last NUM_NODES columns as labels and the rest as features
labels = df.iloc[:, -NUM_NODES:].values
features = df.iloc[:, :-NUM_NODES].values

# Split into train, validation and test
train_size = int(0.7 * len(df))
val_size = int(0.2 * len(df))
test_size = len(df) - train_size - val_size

train_labels = labels[:train_size]
val_labels = labels[train_size:train_size+val_size]
test_labels = labels[train_size+val_size:]

train_features = features[:train_size]
val_features = features[train_size:train_size+val_size]
test_features = features[train_size+val_size:]

# Normalize features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Calculate number of features per node
num_features = train_features.shape[1]/NUM_NODES

# Reshape features into a 3D matrix
train_features = train_features.reshape(train_size, NUM_NODES, int(num_features))
val_features = val_features.reshape(val_size, NUM_NODES, int(num_features))

# Reshape labels into a 3D matrix
train_labels = train_labels.reshape(train_size, NUM_NODES, 1)
val_labels = val_labels.reshape(val_size, NUM_NODES, 1)

# Iterate over train_features and create a Pytorch Geometric dataset
# Each element in the dataset is a graph represented by torch_geometric.data.Data
# Each row in train_features is a graph

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

# Create a Pytorch Geometric dataset from the list of Data objects
class CustomDataset(Dataset):
    def __init__(self, graph_data):
        super(CustomDataset, self).__init__()
        self.graphs = graph_data

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        
        return self.graphs[idx]

batch_size = 32  # Define your desired batch size

# Create empty lists
train_data_list = []
# Iterate over rows
for i in range(train_size):
    # Create a Data object for each graph
    data = Data(
        x=torch.tensor(train_features[i], dtype=torch.double), 
        y=torch.tensor(train_labels[i], dtype=torch.double),
    )
    # Append to list
    train_data_list.append(data)
from torch_geometric.data import Batch
batch = Batch.from_data_list(train_data_list)
train_dataset = CustomDataset(batch)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)


# Create empty lists
val_data_list = []
# Iterate over rows
for i in range(val_size):
    # Create a Data object for each graph
    data = Data(
        x=torch.tensor(val_features[i], dtype=torch.double), 
        y=torch.tensor(val_labels[i], dtype=torch.double),
    )
    # Append to list
    val_data_list.append(data)
val_dataset = CustomDataset(val_data_list)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)


NUM_CLASSES = 2
epochs = 500

model_name = 'GAT'
if model_name == 'GCN':
    model = GCN(7, 16, NUM_CLASSES, NUM_NODES, batch_size)
elif model_name == 'GAT':
    model = GAT(7, 16, NUM_CLASSES, NUM_NODES, batch_size)

optimizer = torch.optim.Adam(
                             model.parameters(),
                                      lr=0.0001)

criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor(
        [0.24,1-0.24],
        requires_grad=False),
        reduction='mean',
        )


for epoch in range(epochs+1):
    # Training
    model.train()
    optimizer.zero_grad()

    train_loss = 0
    for batch in train_dataloader:
        # Squeeze batch.y
        label = batch.y.squeeze()
        # One hot encode label
        label = F.one_hot(label.long(), num_classes=NUM_CLASSES)
        # Cast to float
        label = label.float()
    
        out = model(batch.x)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss

    print(f"done training")

    val_counter = 0
    val_loss = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0

    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            # Squeeze batch.y
            label = batch.y.squeeze().long()

            # One hot encode label
            label_encoded = F.one_hot(label, num_classes=NUM_CLASSES)
            # Cast to float
            label_encoded = label_encoded.float()

            out = model(batch.x)
            loss = criterion(out, label_encoded)

            # Take argmax of predictions
            out = out.argmax(dim=1)

            precision, recall, f1 = evaluate_metrics(out, label)

            val_loss += loss
            val_precision += precision
            val_recall += recall
            val_f1 += f1

            val_counter += 1

    # Print metrics 
    print(f"Epoch {epoch:>3} | Train Loss: {train_loss / len(train_dataloader):.3f}")
    print(f"val_loss: {val_loss / val_counter}")
    print(f"val_precision {val_precision / val_counter}")
    print(f"val_recall {val_recall / val_counter}")
    print(f"val_f1 {val_f1 / val_counter}")