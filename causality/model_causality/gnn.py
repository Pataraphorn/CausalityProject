import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Model: 
    def __init__(self, n_feature):
        super().__init__()
        self.model = GCN(n_feature=n_feature, hidden_channels=64).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train(self, data_loader):
        self.model.train()
        for data in data_loader:
            data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
        return self.model

    def test(self, data_loader):
        self.model.eval()
        correct = 0
        for data in data_loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(data_loader.dataset)  # Derive ratio of correct predictions.
    
    def predict(self, data_loader):
        self.model.eval()
        preds = []
        for data in data_loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            preds.extend(pred.cpu().numpy())
        return preds
    
    def run(self, train_loader, test_loader, n_epoch = 100):
        for epoch in range(1, n_epoch):
            train_acc = Model.test(train_loader)
            test_acc = Model.test(test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(386, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
    
    
