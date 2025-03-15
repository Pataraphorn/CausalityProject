from .._utils import *
import networkx as nx
from IPython.display import SVG, display
import pcalg
from gsq.ci_tests import ci_test_bin, ci_test_dis

class PC:
    def __init__(self):
        self.indep_test_func = ci_test_bin
        self.alpha = 0.01
        self.method = "stable"
        self.graph = None
        self.graph_remove_empty_edges = None
        
    def set_indep_test_func(self, func):
        self.indep_test_func = func

    def estimate_dag(self, data_matrix, columns):
        args = {
            "indep_test_func": self.indep_test_func,
            "data_matrix": data_matrix,
            "alpha": self.alpha,
            "method": self.method,
        }
        self.graph, sep_set = pcalg.estimate_skeleton(**args)
        self.graph = pcalg.estimate_cpdag(skel_graph=self.graph, sep_set=sep_set)
        self.graph = nx.relabel_nodes(self.graph, mapping={node: columns[node] for node in self.graph.nodes()})
        self.graph.draw_pydot_graph()
        return self.graph
    
    def remove_no_edge(self):
        self.graph_remove_empty_edges = self.graph.copy()
        remove = [node for node, degree in dict(self.graph_remove_empty_edges.degree()).items() if degree < 1]
        self.graph_remove_empty_edges.remove_nodes_from(remove)
        return self.graph_remove_empty_edges

    # def show_graph(self, prog: str = "circo" , format: str ="svg"):
    #     svg = SVG(nx.nx_agraph.to_agraph(self.graph).draw(prog=prog, format=format))  # prog='dot'
    #     print(display(svg))
    #     return svg
    
    # def show_number_node_edge(self):
    #     print("Number of Edges :", self.graph.number_of_edges())
    #     print("After remove nodes that do not have edges")
    #     print("Number of Edges :", self.graph_remove_empty_edges.number_of_edges())
