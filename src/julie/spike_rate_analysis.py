import os
import pandas as pd
import networkx as nx
from clat.intan.rhd import load_intan_rhd_format
from matplotlib import pyplot as plt

from julie.single_channel_analysis import read_pickle, calculate_spike_rate
from julie.single_unit_analysis import calculate_spike_timestamps
from julie.social_network_anlaysis.monkeyids import Monkey


def main():
    date = "2023-10-30"
    round = "1698699440778381_231030_165721"
    sorted_spikes_filename = "sorted_spikes.pkl"

    cortana_path = "/home/connorlab/Documents/IntanData"
    round_path = os.path.join(cortana_path, date, round)
    compiled_trials_filepath = os.path.join(round_path, "compiled.pk1")
    raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)

    # TODO: specify which sorting pickle to use and which units to plot, then add them to dataframe
    rhd_file_path = os.path.join(round_path, "info.rhd")
    sorted_spikes_filepath = os.path.join(cortana_path, date, round, sorted_spikes_filename)
    sorted_spikes = read_pickle(sorted_spikes_filepath)
    sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
    sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)

    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    unique_channels = set()
    for index, row in raw_trial_data.iterrows():
        unique_channels.update(row['SpikeTimes'].keys())

    avg_spike_rate_by_unit = pd.DataFrame(index=[])

    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}

        for channel in unique_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if channel in row['SpikeTimes']:
                    data = row['SpikeTimes'][channel]
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            monkey_specific_spike_rate[channel] = avg_spike_rate

        # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)

    print(avg_spike_rate_by_unit)

    G = nx.DiGraph()
    monkeys = [Monkey.Z_M1.value, Monkey.Z_F1.value, Monkey.Z_F2.value,
               Monkey.Z_F3.value, Monkey.Z_F4.value, Monkey.Z_F5.value,
               Monkey.Z_F6.value, Monkey.Z_F7.value, Monkey.Z_J1.value, Monkey.Z_J2.value]
    G.add_nodes_from(monkeys)

    second_row = avg_spike_rate_by_unit.iloc[1]
    norm_values = (second_row - second_row.min()) / (second_row.max() - second_row.min())
    print(f'norm values: {norm_values}')
    norm_values_dict = norm_values.to_dict()
    print(f'norm values dict: {norm_values_dict}')
    set_node_attributes_with_default(G, norm_values_dict, 'spike_rate', default_value=0)
    print(G.nodes(data=True))
    # Define a colormap and normalize the values
    colormap = plt.cm.get_cmap('YlOrBr')
    attribute = nx.get_node_attributes(G, 'spike_rate')

    # Normalize the spike_rate values to range [0, 1]
    norm = plt.Normalize(min(attribute.values()), max(attribute.values()))
    # Map the normalized values to colors using the colormap
    node_colors = [colormap(norm(attribute[node])) for node in G.nodes()]
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    nx.draw(G, pos, with_labels=True, node_color=node_colors)
    plt.show()

    # spike rate for each picture
    # for index, row in raw_trial_data.iterrows():
    #     for unit, data in row['SpikeTimes'].items():
    #         start, stop = row['EpochStartStop']
    #         duration = stop - start
    #         rate = len(data)/duration
    #         print(f'spike rate manually calculated: {rate}')
    #         spike_rate = calculate_spike_rate(data, row['EpochStartStop'])
    #         print(f'For unit {unit}, the spike rate is {spike_rate}')

def set_node_attributes_with_default(graph, values_dict, attribute_name, default_value=0):
    for node in graph.nodes():
        value = values_dict.get(node, default_value)
        nx.set_node_attributes(graph, {node: value}, attribute_name)


if __name__ == '__main__':
    main()