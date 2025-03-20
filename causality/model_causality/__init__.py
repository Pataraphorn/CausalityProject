from .._utils import *
from . import gcn, pc_algorithm, fci_algorithm
import networkx as nx
from IPython.display import SVG, display
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, f1_score

model_mapper = {
    "PC": pc_algorithm.PC(),
    "FCI": fci_algorithm.FCI(),
    "GCN": gcn.clf,
    # 'LGBM': lgbm.clf,
}

fit_map = {
    "GCN": gcn.fit,
    # 'LGBM': lgbm.fit
}

pred_map = {
    "GCN": gcn.predict,
    # 'LGBM': lgbm.pred
}

label_map = {"GCN": None}

explainer_map = {"GCN": gcn.get_explainer}

explain_map = {"GCN": gcn.explain}

class causalityModel:
    def __init__(self, model_type="PC"):
        super().__init__()
        self.model_type = model_type
        self.model = model_mapper[model_type]
        self.label = label_map[model_type]
        self.graph = None
        print("Processing with causality modeling name: ", self.model_type)

    def set_graph(self, graph):
        self.graph = graph
        return self.graph

    def training(self, data_matrix: str, columns: str):
        if self.model_type == "PC":
            self.graph = self.model.estimate_dag(data_matrix=data_matrix, columns=columns)
        else:
            print(f"This version not having causality modeling named {self.model_type} yet.")
            pass
        return self.graph

    def remove_edge(self):
        if self.model_type == "PC":
            graph = self.model.remove_no_edge()
        else:
            print(f"This version not having causality modeling named {self.model_type} yet.")
            pass
        return graph

    def show_graph(self, prog: str = "circo", format: str = "svg", graph=None):
        graph_temp = self.graph if graph is None else graph
        svg = SVG(
            nx.nx_agraph.to_agraph(graph_temp).draw(prog=prog, format=format)
        )  # prog='dot'
        print(display(svg))
        return svg

    def show_number_node_edge(self):
        print("Number of Edges :", self.graph.number_of_edges())
        print("After remove nodes that do not have edges")
        print("Number of Edges :", self.graph_remove_empty_edges.number_of_edges())

    def show_model(self):
        self.model.show_model()

    def set_model(self, model):
        self.model = model
        return self.model

    def fit(self, train_loader, test_loader, n_epoch: int = 100):
        self.model = fit_map[self.model_type](train_loader, test_loader, n_epoch)

    def predict(self, data_loader):
        return pred_map[self.model_type](data_loader)

    def set_label(self, label):
        self.label = label
        return self.label

    def confusion_matrix_plot(self, y, pred):
        ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=pred, display_labels=self.label)

    def eval_report(self, y, pred):
        print(
            classification_report(
                y_true=y, y_pred=pred, digits=3, target_names=self.label
            )
        )

    def confusion_report(self, y, pred):
        return confusion_matrix(
            y_true=y,
            y_pred=pred,
            labels=np.arange(len(self.label)) if self.label is not None else None,
        )

    def f1_score(self, X, y):
        return f1_score(X, y, average="weighted")

    def get_explainer(self):
        return explainer_map[self.model_type]

    def explain(self, data_loader, node_index):
        return explain_map[self.model_type](data_loader, node_index)
