from .._utils import *

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Model: 
    def __init__(self, n_feature):
        super().__init__()
        self.model = GCN(n_feature=n_feature, hidden_channels=512).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def show_model(self):
        print(self.model)

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

class GCN(torch.nn.Module):

    def __init__(self, n_feature, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(n_feature, hidden_channels)
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

clf = Model(n_feature=386)

explainer = Explainer(
    model=clf.model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",  # node classification
        return_type="log_probs",
    ),
)


def fit(train_loader, test_loader, n_epoch=100):
    print("Start training model...")
    for epoch in range(1, n_epoch):
        clf.train(train_loader)
        train_acc = clf.test(train_loader)
        test_acc = clf.test(test_loader)
        # Check if the values are floats or extract the numeric value if necessary
        if isinstance(train_acc, (int, float)):
            print(
                f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )
        else:
            # Handle if the return value is not a numeric type (you might need to debug what it returns)
            print(f"Epoch: {epoch:03d}, Train Acc: {train_acc}, Test Acc: {test_acc}")


def predict(data_loader):
    return clf.predict(data_loader)


def get_explainer():
    return explainer


def explain(data_loader, node_index):
    print(f"Explanation for node index {node_index}")
    for data in data_loader:
        print(f"    Data: {data}")
        explanation = explainer(
            data.x, data.edge_index, batch=data.batch, index=node_index
        )
        print(f"    Generated explanations in {explanation.available_explanations}")

        # Get feature importance scores
        # top_k = 10
        feature_importance = explanation.node_mask
        if feature_importance is not None:
            feature_importance_np = feature_importance.detach().cpu().numpy()
            # print("   Feature Importance Scores:", feature_importance_np)
            # top_features = feature_importance_np.argsort()[-top_k:][::-1]  # Get indices of top 10 features
            # print(f"    Top {top_k} Most Important Features:")
            # for idx in top_features:
            #     print(f"    Feature {idx}: Importance {feature_importance_np[idx]}")
        else:
            print("Feature importance is not available.")

        # Save visualization
        # Define the directory path
        # output_dir = f'{current_dir}/result_explanier'
        # os.makedirs(output_dir, exist_ok=True)

        # node features are most important for the model’s decision => Higher values indicate more influential features
        path = f"result_feature_importance_node{node_index}.png"
        explanation.visualize_feature_importance(path, top_k=20)
        print(f"Feature importance plot has been saved to '{path}'")

        # visual representation of the important edges and nodes in the graph.
        # Highlights which part of the network contributes most to the model’s decision for the given node_index.
        path = f"result_subgraph_node{node_index}.pdf"
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")

    return explanation, feature_importance_np
