import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import social_data_reader
from monkeyids import Monkey


def create_digraph_with_edge_weights(edge_weights):
    G = nx.from_pandas_edgelist(edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                    create_using=nx.DiGraph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    normalized_matrix = adj_matrix / np.sum(adj_matrix)
    print(adj_matrix)
    adj_df = pd.DataFrame(normalized_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    return G, adj_matrix, weights

def create_graph_with_edge_weights(edge_weights):
    pass

def create_digraph_for_zombies():
    G = nx.DiGraph()
    zombies = [Monkey.Z_M1.value, Monkey.Z_F1.value, Monkey.Z_F2.value,
               Monkey.Z_F3.value, Monkey.Z_F4.value, Monkey.Z_F5.value,
               Monkey.Z_F6.value, Monkey.Z_F7.value, Monkey.Z_J1.value, Monkey.Z_J2.value]
    G.add_nodes_from(zombies)
    return G