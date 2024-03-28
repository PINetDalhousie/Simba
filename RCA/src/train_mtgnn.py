import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric_temporal.nn.attention import MTGNN
# import precision recall and f1scores from scikit learn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torcheval.metrics.functional import multiclass_f1_score

NUM_NODES = 5
NUM_FEATURES = 9
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
NUM_CLASSES = 4
BATCH_SIZE = 2048
ClASS_COUNTS = [18000, 120, 120, 120]
seq_len = 5

# Calculate the class weights
class_weights = [x / sum(ClASS_COUNTS) for x in ClASS_COUNTS]
class_weights = [1/x for x in class_weights]
#class_weights = [x / sum(class_weights) for x in class_weights]
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Load the mtgnn .txt file as a pandas dataframe
graph_data = pd.read_csv('../data/baseline_omni/20240308_153331_MTGNN.txt', sep=',', header=None)

# Print the number of columns
#print(graph_data.shape)
#print(asd)

# Separate the data into the features and the target
features = graph_data.iloc[:, :NUM_NODES * NUM_FEATURES]
target = graph_data.iloc[:, NUM_NODES * NUM_FEATURES:]

#print(graph_data.info())
#print(asd)
# print(target[30].value_counts())
# print(target[31].value_counts())
# print(target[32].value_counts())
# print(target[33].value_counts())
# print(target[34].value_counts())

# print(asd)

# Split the data into train, validation and test sets
train_size = int(TRAIN_SIZE * len(features))
val_size = int(VAL_SIZE * len(features))
test_size = len(features) - train_size - val_size

train_features = features.iloc[:train_size]
train_target = target.iloc[:train_size]

val_features = features.iloc[train_size:train_size+val_size]
val_target = target.iloc[train_size:train_size+val_size]

test_features = features.iloc[train_size+val_size:]
test_target = target.iloc[train_size+val_size:]

# Print total number of occurrences for each class
# print(val_target[30].value_counts())
# print(val_target[31].value_counts())
# print(val_target[32].value_counts())
# print(val_target[33].value_counts())
# print(val_target[34].value_counts())

# print(asd)

# Convert the data into float tensors
train_features = torch.tensor(train_features.values, dtype=torch.float)
train_target = torch.tensor(train_target.values, dtype=torch.float)

val_features = torch.tensor(val_features.values, dtype=torch.float)
val_target = torch.tensor(val_target.values, dtype=torch.float)



# Create a custom dataset
class GraphDataset(Dataset):
  def __init__(self, features, target, seq_len, num_nodes):
    self.features = features
    self.target = target
    self.seq_len = seq_len
    self.num_nodes = num_nodes

  def __len__(self):
    return len(self.features) - self.seq_len + 1

  def __getitem__(self, idx):
    # Get seq_len data points
    current_features = self.features[idx:idx+self.seq_len]
    # Reshape to seq_leng x num_nodes x num_features
    current_features = current_features.view(self.seq_len, self.num_nodes, -1)
    # Permute the tensor to num_features x num_nodes x seq_len
    current_features = current_features.permute(2, 1, 0)

    # Get the target
    current_target = self.target[idx+self.seq_len-1,:]
    # One hot encode the target
    current_target = torch.nn.functional.one_hot(current_target.to(torch.int64), num_classes=NUM_CLASSES)
    
    # Permute target to num_classes x num_nodes x 1
    current_target = current_target.permute(1, 0).unsqueeze(2)
    # Cast to float
    current_target = current_target.float()

    return current_features, current_target

## Calculate the mean and std of the dataset
mean = train_features.mean(dim=0)
std = train_features.std(dim=0)
# Normalize the dataset
train_features = (train_features - mean) / std
val_features = (val_features - mean) / std

# Create a dataloader
train_dataset = GraphDataset(train_features, train_target, seq_len=seq_len, num_nodes=NUM_NODES)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = GraphDataset(val_features, val_target, seq_len=seq_len, num_nodes=NUM_NODES)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get cpu, gpu or mps device for training.
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f"Using {device} device")


model = MTGNN(
  gcn_true=True,
  build_adj=True,
  gcn_depth=1,
  num_nodes=NUM_NODES,
  kernel_set=[3, 3],
  kernel_size=3,
  dropout=0.0,
  subgraph_size=5,
  node_dim=NUM_FEATURES,
  dilation_exponential=1,
  conv_channels=32,
  residual_channels=32,
  skip_channels=32,
  end_channels=32,
  seq_length=5,
  layers=2,
  propalpha=0.5,
  tanhalpha=0.5,
  in_dim=NUM_FEATURES,
  out_dim=NUM_CLASSES,
  layer_norm_affline=False,
).to(device)

print(model)
class_weights = class_weights.to(device)

from torch.nn import functional as F


class WeightedCELoss(torch.nn.Module):
  def __init__(self, weights):
    super(WeightedCELoss, self).__init__()
    self.weights = weights
    print(weights)

  def forward(self, output, target):
    # Apply weighted cross-entropy loss
    loss = F.cross_entropy(output, target, weight=self.weights, reduction='none')
    # Sum across 2nd dimension
    loss = loss.sum(dim=1)
    # Divide by number of base stations
    loss = loss / NUM_NODES
    # Reduce loss to batch mean
    return loss.mean()
  

loss_fn = WeightedCELoss(class_weights)
#loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

def train(dataloader, model, loss_fn, optimizer, summary_writer, epoch):
  size = len(dataloader.dataset)
  model.train()
  # Track number of samples for each class
  class_counts = np.zeros(NUM_CLASSES)

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    print(X.shape)
    print(asd)

    # Compute prediction error
    pred = model(X)
    
    # Add softmax layer to the output
    pred = torch.nn.functional.softmax(pred, dim=1)

    loss = loss_fn(pred, y)
    #print(loss)
    #print(asd)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #if batch % 100 == 0:
    loss, current = loss.item(), (batch + 1) * len(X)

    # Calculate F1 score
    pred = pred.argmax(dim=1)
    y = y.argmax(dim=1)
    pred = pred.reshape(-1)
    y = y.reshape(-1)
    # Convert to numpy
    pred = pred.cpu().detach()
    y = y.cpu().detach()
    # Convert to shape (num_samples)
    #print(y)
    # Calculate the class counts
    #class_counts += np.bincount(y, minlength=NUM_CLASSES)
    #print(class_counts)
    #print(asd)
    # Check both shapes are the same
    assert pred.shape == y.shape

    train_f1score = multiclass_f1_score(pred, y, num_classes=NUM_CLASSES, average=None).cpu().detach().numpy()

  # Log the loss and f1 score
  
  print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  print(f"Train f1 score: {train_f1score}")
  summary_writer.add_scalar('Loss', loss, epoch)
  for i in range(NUM_CLASSES):
    summary_writer.add_scalar(f'F1_Score_{i}', train_f1score[i], epoch)

    

def val(dataloader, model, loss_fn, summary_writer, epoch):
  model.eval()
  size = len(dataloader.dataset)
  val_loss, correct = 0, 0
  class_counts = np.zeros(NUM_CLASSES)
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      pred = torch.nn.functional.softmax(pred, dim=1)
      val_loss = loss_fn(pred, y).item()
      
      pred = pred.argmax(dim=1)
      y = y.argmax(dim=1)
      pred = pred.reshape(-1)
      y = y.reshape(-1)
      pred = pred.cpu().detach()
      y = y.cpu().detach()
      
      assert pred.shape == y.shape
      val_f1score = multiclass_f1_score(pred, y, num_classes=NUM_CLASSES, average=None)
  
  print(f"Avg val loss: {val_loss:>8f}")
  print(f"Val f1 score: {val_f1score}")
  summary_writer.add_scalar('Loss', val_loss, epoch)
  for i in range(NUM_CLASSES):
    summary_writer.add_scalar(f'F1_Score_{i}', val_f1score[i], epoch)

# Get current time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Create log path
log_path = f'../logs/{current_time}'
# Create the path
import os
os.makedirs(log_path)
# Create summary writers for train and val to log loss and f1score
train_writer = SummaryWriter(f'{log_path}/train')
val_writer = SummaryWriter(f'{log_path}/val')

epochs = 10000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, train_writer, t)
    val(val_dataloader, model, loss_fn, val_writer, t)
# Closer writers
train_writer.close()
val_writer.close()
