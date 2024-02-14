import os
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from julie.social_network_anlaysis.network import create_digraph_with_edge_weights
from julie.social_network_anlaysis.social_data_reader import read_raw_social_data, extract_pairwise_interactions, \
    generate_weights_from_pairwise_interactions, clean_raw_social_data
from julie.spike_rate_analysis import read_raw_trial_data, compute_spike_rates_per_channel_per_monkey, \
    set_node_attributes_with_default



current_dir = os.getcwd()
raw_data_file_name = 'ZombiesFinalRawData.xlsx'
file_path = Path(current_dir).parent.parent / 'resources' / raw_data_file_name
raw_social_data = read_raw_social_data(file_path)
social_data = clean_raw_social_data(raw_social_data)
agonistic = extract_pairwise_interactions(social_data, 'agonistic')
edge_weights = generate_weights_from_pairwise_interactions(agonistic)

date = "2023-10-30"
round = "1698699440778381_231030_165721"
cortana_path = "/home/connorlab/Documents/IntanData"
round_path = os.path.join(cortana_path, date, round)
raw_trial_data = read_raw_trial_data(round_path)
avg_spike_rate = compute_spike_rates_per_channel_per_monkey(raw_trial_data)
random_row = avg_spike_rate.loc["Channel.C_017_Unit 1"]
norm_values = ((random_row - random_row.min()) / (random_row.max() - random_row.min())).to_dict()

G, adj_matrix, weights = create_digraph_with_edge_weights(edge_weights)
set_node_attributes_with_default(G, norm_values, 'spike_rate', default_value=0)

colormap = plt.cm.get_cmap('YlOrBr')
attribute = nx.get_node_attributes(G, 'spike_rate')
node_colors = [colormap(attribute[node]) for node in G.nodes()]

# Visualize the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed =42)  # Position nodes using the spring layout
nx.draw(G, pos, with_labels=True, width=weights, font_size=7, node_color=node_colors, node_size=1500)
plt.show()