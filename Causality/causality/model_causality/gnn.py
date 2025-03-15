import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split

class GNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, data):
        # data contains node features (data.x) and edge index (data.edge_index)
        x, edge_index = data.x, data.edge_index
        # Pass through the first GCN layer and apply ReLU
        x = F.relu(self.conv1(x, edge_index))
        # Pass through the second GCN layer
        x = self.conv2(x, edge_index)
        return F.sigmoid(x)  # Sigmoid activation for binary classification (causal/non-causal)