import networkx as nx
import pandas as pd
from monkeyids import Monkey


def create_digraph_with_edge_weights(edge_weights):
    normalized_edge_weights = normalize_weights_min_max(edge_weights)
    print(normalized_edge_weights)
    G = nx.from_pandas_edgelist(normalized_edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                create_using=nx.DiGraph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_df = pd.DataFrame(adj_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    # print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(weights)
    return G, adj_matrix, weights


def normalize_weights_min_max(edgelist):
    normalized_edgelist = edgelist[['Focal Name', 'Social Modifier']]
    normalized_edgelist['weight'] = (edgelist['weight'] - edgelist['weight'].min()) / (edgelist['weight'].max() - edgelist['weight'].min())
    normalized_edgelist['weight'] = normalized_edgelist['weight'] * 8
    return normalized_edgelist


def create_graph_with_edge_weights(edge_weights):
    normalized_edge_weights = normalize_weights_min_max(edge_weights)
    G = nx.from_pandas_edgelist(normalized_edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                create_using=nx.Graph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_df = pd.DataFrame(adj_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    # print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(weights)
    return G, adj_matrix, weights



def create_social_graph_for(graph_type, monkey_group):

    if graph_type == 'digraph':
        G = nx.DiGraph()
    elif graph_type == 'graph':
        G = nx.Graph()
    else:
        raise ValueError("Invalid graph type. Please provide 'digraph' or 'graph' as the type.")

    if monkey_group.lower() == 'zombies':
        zombies = [Monkey.Z_M1.value, Monkey.Z_F1.value, Monkey.Z_F2.value,
                   Monkey.Z_F3.value, Monkey.Z_F4.value, Monkey.Z_F5.value,
                   Monkey.Z_F6.value, Monkey.Z_F7.value, Monkey.Z_J1.value, Monkey.Z_J2.value]
        G.add_nodes_from(zombies)
    elif monkey_group.lower() == 'bestfrans':
        bestfrans = [Monkey.B_M1.value, Monkey.B_F1.value, Monkey.B_F2.value, Monkey.B_F3.value, Monkey.B_J1.value]
        G.add_nodes_from(bestfrans)
    elif monkey_group.lower() == 'instigators':
        instigators = [Monkey.I_M1.value, Monkey.I_F1.value, Monkey.I_F2.value, Monkey.I_F3.value, Monkey.I_F4.value,
                       Monkey.I_F5.value, Monkey.I_F6.value, Monkey.I_F7.value, Monkey.I_F8.value, Monkey.I_F9.value,
                       Monkey.I_J1.value, Monkey.I_J2.value, Monkey.I_J3.value, Monkey.I_J4.value]
        G.add_nodes_from(instigators)
    elif monkey_group.lower() == 'strangerthings':
        strangerthings = [Monkey.S_M1.value, Monkey.S_F1.value, Monkey.S_F2.value, Monkey.S_F3.value, Monkey.S_F4.value,
                          Monkey.S_J1.value, Monkey.S_J2.value, Monkey.S_J3.value, Monkey.S_J4.value, Monkey.S_J5.value,
                          Monkey.S_J6.value, Monkey]
        G.add_nodes_from(strangerthings)
    else:
        raise ValueError("Invalid monkey group name. Please provide 'zombies', 'bestfrans', 'instigators', "
                         "or 'strangerthings'.")

    return G

