import os
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path

import pandas as pd

from julie.social_network_anlaysis.network import create_digraph_with_edge_weights, create_graph_with_edge_weights
from julie.social_network_anlaysis.social_data_reader import read_raw_social_data, extract_pairwise_interactions, \
    generate_edgelist_from_pairwise_interactions, clean_raw_social_data, combine_edgelists, \
    read_social_data_and_validate
from julie.spike_rate_analysis import read_sorted_data, compute_spike_rates_per_channel_per_monkey_for_raw_data, \
    set_node_attributes_with_default
from social_data_reader import extract_grooming_interactions


def main():
    social_data = read_social_data_and_validate()
    extract_grooming_interactions(social_data)
    current_dir = os.getcwd()
    raw_data_file_name = 'ZombiesFinalRawData.xlsx'
    file_path = Path(current_dir).parent.parent / 'resources' / raw_data_file_name
    raw_social_data = read_raw_social_data(file_path)
    social_data = clean_raw_social_data(raw_social_data)
    agonistic = extract_pairwise_interactions(social_data, 'agonistic')
    submissive = extract_pairwise_interactions(social_data, 'submissive')
    edgelist_agonistic = generate_edgelist_from_pairwise_interactions(agonistic)
    edgelist_submissive = generate_edgelist_from_pairwise_interactions(submissive)
    combined_edgelist = combine_edgelists(edgelist_agonistic, edgelist_submissive)

    affiliative= extract_pairwise_interactions(social_data, 'affiliative')
    edgelist_affiliative = generate_edgelist_from_pairwise_interactions(affiliative)

    date = "2023-10-30"
    round = "1698699440778381_231030_165721"
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = os.path.join(cortana_path, date, round)
    raw_trial_data = read_sorted_data(round_path)
    avg_spike_rate = compute_spike_rates_per_channel_per_monkey_for_raw_data(raw_trial_data)
    random_row = avg_spike_rate.loc["Channel.C_017_Unit 1"]
    norm_values = ((random_row - random_row.min()) / (random_row.max() - random_row.min())).to_dict()

    #G, adj_matrix, weights = create_digraph_with_edge_weights(combined_edgelist)
    G, adj_matrix, weights = create_graph_with_edge_weights(edgelist_affiliative)
    set_node_attributes_with_default(G, norm_values, 'spike_rate', default_value=0)

    colormap = plt.cm.get_cmap('YlOrBr')
    attribute = nx.get_node_attributes(G, 'spike_rate')
    node_colors = [colormap(attribute[node]) for node in G.nodes()]
    print(f'degree centrality {nx.degree_centrality(G)}')
    print(f'betweenness centrality {nx.betweenness_centrality(G)}')
    print(f'eigenvector centrality {nx.eigenvector_centrality(G)}')

    # Visualize the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    nx.draw(G, pos, with_labels=True, width=weights, font_size=7, node_color=node_colors, node_size=1500)
    plt.show()

if __name__ == '__main__':
    main()
