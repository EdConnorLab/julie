import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import social_data_reader

def plot_directed_social_network(edge_weights):
    G = nx.from_pandas_edgelist(edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                    create_using=nx.DiGraph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    normalized_matrix = adj_matrix / np.sum(adj_matrix)
    print(adj_matrix)
    adj_df = pd.DataFrame(normalized_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]

    # Visualize the graph with a different layout and edge color
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes, seed for reproducibility
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, arrowsize=15, width=weights,
            edge_color='darkblue')
    plt.show()

def plot_undirected_social_network(edge_weights):
    pass


def main():
    pass