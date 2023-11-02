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

# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="CiteSeer")

data = dataset[0]

# # Print information about the dataset
# print(f'Dataset: {dataset}')
# print('-------------------')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of nodes: {data.x.shape[0]}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')

# # Print information about the graph
# print(f'\nGraph:')
# print('------')
# print(f'Edges are directed: {data.is_directed()}')
# print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Graph has loops: {data.has_self_loops()}')

# print(data)

# from torch_geometric.utils import remove_isolated_nodes

# isolated = (remove_isolated_nodes(data['edge_index'])[2] == False).sum(dim=0).item()
# print(f'Number of isolated nodes = {isolated}')

# from torch_geometric.utils import to_networkx

# G = to_networkx(data, to_undirected=True)
# plt.figure(figsize=(18,18))
# plt.axis('off')
# nx.draw_networkx(G,
#                 pos=nx.spring_layout(G, seed=0),
#                 with_labels=False,
#                 node_size=50,
#                 node_color=data.y,
#                 width=2,
#                 edge_color="grey"
#                 )
# plt.show()

# print(asd)
# from torch_geometric.utils import degree
# from collections import Counter

# # Get list of degrees for each node
# degrees = degree(data.edge_index[0]).numpy()

# # Count the number of nodes for each degree
# numbers = Counter(degrees)

# # Bar plot
# fig, ax = plt.subplots(figsize=(18, 7))
# ax.set_xlabel('Node degree')
# ax.set_ylabel('Number of nodes')
# plt.bar(numbers.keys(),
#         numbers.values(),
#         color='#0A047A')


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
  def __init__(self, dim_in, dim_h, dim_out, num_nodes):
    super().__init__()

    self.gc = GraphCons(
        num_nodes, 
        num_nodes, 
        dim=40,
        )

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


    #h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(x, edge_index)
    print(h)
    print(asd)
    h = torch.relu(h)
    #h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)

    print(edge_index)
    print(adp)
    print(asd)
    return h, F.log_softmax(h, dim=1)


class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

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
print(df.head())

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

# Reshape labels into a 3D matrix
train_labels = train_labels.reshape(train_size, NUM_NODES, 1)

# Iterate over train_features and create a Pytorch Geometric dataset
# Each element in the dataset is a graph represented by torch_geometric.data.Data
# Each row in train_features is a graph

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Create empty lists
data_list = []

# Iterate over rows
for i in range(train_size):
    # Create a Data object for each graph
    data = Data(
        x=torch.tensor(train_features[i], dtype=torch.double), 
        y=torch.tensor(train_labels[i], dtype=torch.double),
    )
    # Append to list
    data_list.append(data)


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

dataset = CustomDataset(data_list)
batch_size = 1  # Define your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

NUM_CLASSES = 2
epochs = 10

model_name = 'GCN'
if model_name == 'GCN':
    model = GCN(7, 16, NUM_CLASSES, NUM_NODES)
elif model_name == 'GAT':
    model = GAT(dataset.num_features, 8, NUM_CLASSES)

optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.001,
                                      weight_decay=5e-4)

for epoch in range(epochs+1):
    # Training
    model.train()
    optimizer.zero_grad()

    for batch in dataloader:
        # print(batch)
        # print(batch.x)
        # print(batch.y)
        # print(asd)

        # cast to double
        #batch.x = batch.x.double()

        out = model(batch.x)
        print(asd)

print(train_features.shape)
print(train_features[0])
print(train_labels.shape)
print(asd)

# Create Pytorch Geometric dataset from features and labels
# Each row 

print(data)

print(asd)

# Create GCN model
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)

# Train
train(gcn, data)

# Test
acc = test(gcn, data)
print(f'\nGCN test accuracy: {acc*100:.2f}%\n')


# Create GAT model
gat = GAT(dataset.num_features, 8, dataset.num_classes)
print(gat)

# Train
train(gat, data)

# Test
acc = test(gat, data)
print(f'\nGAT test accuracy: {acc*100:.2f}%\n')



# Initialize new untrained model
untrained_gat = GAT(dataset.num_features, 8, dataset.num_classes)

# Get embeddings
h, _ = untrained_gat(data.x, data.edge_index)

# Train TSNE
tsne = TSNE(n_components=2, learning_rate='auto',
         init='pca').fit_transform(h.detach())

# Plot TSNE
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.show()


# Get embeddings
h, _ = gat(data.x, data.edge_index)

# Train TSNE
tsne = TSNE(n_components=2, learning_rate='auto',
         init='pca').fit_transform(h.detach())

# Plot TSNE
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.show()

from torch_geometric.utils import degree

# Get model's classifications
_, out = gat(data.x, data.edge_index)

# Calculate the degree of each node
degrees = degree(data.edge_index[0]).numpy()

# Store accuracy scores and sample sizes
accuracies = []
sizes = []

# Accuracy for degrees between 0 and 5
for i in range(0, 6):
  mask = np.where(degrees == i)[0]
  accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
  sizes.append(len(mask))

# Accuracy for degrees > 5
mask = np.where(degrees > 5)[0]
accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
sizes.append(len(mask))

# Bar plot
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlabel('Node degree')
ax.set_ylabel('Accuracy score')
plt.bar(['0','1','2','3','4','5','>5'],
        accuracies,
        color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%',
             ha='center', color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i]//2, sizes[i],
             ha='center', color='white')

if __name__ == "__main__":
    pass