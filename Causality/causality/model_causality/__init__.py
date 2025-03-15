from .._utils import *
from ..model_causality import pc_algorithm, fci_algorithm
import networkx as nx
from IPython.display import SVG, display

model_mapper = {
    'PC': pc_algorithm.PC(),
    'FCI': fci_algorithm.FCI()
}

class CausalityModel:
    def __init__(self, model_type="PC"):
        super().__init__()
        self.model_type = model_type
        self.graph = None
        print("Processing with causality modeling name: ", self.model_type)

    def set_graph(self, graph):
        self.graph = graph
        return self.graph
    
    def training(self, data_matrix: str, columns: str):
        model = model_mapper[self.model_type]
        if self.model_type == "PC":
            self.graph = model.estimate_dag(data_matrix=data_matrix, columns=columns)
        else:
            print(f"This version not having causality modeling named {self.model_type} yet.")
            pass
        return self.graph
    
    def remove_edge(self):
        if self.model_type == "PC":
            graph = pc_algorithm.PC().remove_no_edge()
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
