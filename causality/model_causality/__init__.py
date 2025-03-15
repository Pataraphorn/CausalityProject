from .._utils import *
from ..model_causality import pc_algorithm, fci_algorithm, gnn
import networkx as nx
from IPython.display import SVG, display
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, f1_score

model_mapper = {
    'PC': pc_algorithm.PC(),
    'FCI': fci_algorithm.FCI(),
    # 'GNN': gnn.Model(n_feature = 25)
}

fit_map = {
    # 'GNN': gnn.Model().run
    # 'lgbm': lgbm.fit
}

pred_map = {
    # 'GNN': gnn.Model().predict
    # 'lgbm': lgbm.pred
}

class CausalityModel:
    def __init__(self, model_type="PC"):
        super().__init__()
        self.model_type = model_type
        self.model = model_mapper[model_type]
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
    
    def show_graph(self, prog: str = "circo" , format: str ="svg"):
        svg = SVG(nx.nx_agraph.to_agraph(self.graph).draw(prog=prog, format=format))  # prog='dot'
        print(display(svg))
        return svg

    def show_number_node_edge(self):
        print("Number of Edges :", self.graph.number_of_edges())
        print("After remove nodes that do not have edges")
        print("Number of Edges :", self.graph_remove_empty_edges.number_of_edges())

    def fit(self, X, y, val_X=None, val_y=None):
        self.model = fit_map[self.model_type](self.model,
                                              X=X, y=y,
                                              val_X=val_X, val_y=val_y)
    
    def predict(self, X):
        return pred_map[self.model_type](self.model, X=X)
    
    def confusion_matrix_plot(self, X, y):
        pred = self.predict(X)
        ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=pred, display_labels=self.label)
     
    def eval_report(self, X, y):
        pred = self.predict(X)
        return classification_report(y_true=y, y_pred=pred, digits=3, target_names=self.label)
    
    def confusion_report(self, X, y):
        pred = self.predict(X)
        return confusion_matrix(y_true=y, y_pred=pred, labels=self.label)
    
    def f1_score(self, X, y):
        return f1_score(X, y, average='weighted')   