from .._utils import *
from collections import Counter
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

class2bin = {'Normal':0,
             'Anomaly':1}
bin2class = ['Normal', 'Anomaly']

def class_trans(data, mode=class2bin):
    return mode[data]

def get_embeddings(data, embeddings):
    a = np.zeros((len(data), embeddings.shape[1]))
    for indice in range(len(data)):
        if indice == -1 :
            continue
        else :
            a[indice] = embeddings[indice]
    return a