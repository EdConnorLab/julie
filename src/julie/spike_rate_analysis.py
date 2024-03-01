import os
from pathlib import Path

import pandas as pd
import networkx as nx
from clat.intan.rhd import load_intan_rhd_format
from clat.intan.channels import Channel
from matplotlib import pyplot as plt

from julie.single_channel_analysis import read_pickle, calculate_spike_rate, extract_target_channel_data
from julie.single_unit_analysis import calculate_spike_timestamps

def compute_average_spike_rates(date, round):
    cortana_path = "/home/connorlab/Documents/IntanData"
    round_path = Path(os.path.join(cortana_path, date, round))
    script_dir = Path(__file__).parent
    compiled_dir = (script_dir / '..' / '..' / 'compiled').resolve()
    matching_files = list(compiled_dir.glob(f"*{round}*"))
    raw_trial_data = read_pickle(matching_files[0])
    print("raw trial data printed here: ")
    print(raw_trial_data)

    # Check if the experimental round is sorted
    sorted = round_path / 'sorted_spikes.pkl'
    if sorted.exists():
        print("This round contains sorted spikes")
        sorted_data = read_sorted_data(round_path)
        channels_with_units = set()
        for index, row in sorted_data.iterrows():
            channels_with_units.update(row['SpikeTimes'].keys())
        sorted_channels = {channel.split('_Unit')[0] for channel in channels_with_units}

    valid_channels = {Channel.C_000, Channel.C_003, Channel.C_010} - sorted_channels

    sorted_data_spike_rates = compute_spike_rates_per_channel_per_monkey_for_all_channels(sorted_data)
    raw_data_spike_rates = compute_spike_rates_per_channel_per_monkey(raw_trial_data, valid_channels)
    print(sorted_data_spike_rates)
    print(raw_data_spike_rates)
    average_spike_rates = pd.concat([sorted_data_spike_rates, raw_data_spike_rates])
    print('-----------------------combined--------------------')
    print(average_spike_rates)

    return average_spike_rates

def read_sorted_data(round_path, sorted_spikes_filename ='sorted_spikes.pkl', compiled_trials_filename ='compiled.pk1'):
    compiled_trials_filepath = os.path.join(round_path, compiled_trials_filename)
    raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)
    rhd_file_path = os.path.join(round_path, 'info.rhd')
    sorted_spikes_filepath = os.path.join(round_path, sorted_spikes_filename)
    sorted_spikes = read_pickle(sorted_spikes_filepath)
    sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
    sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)
    return sorted_data
def compute_spike_rates_per_channel_per_monkey_for_all_channels(raw_trial_data):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    for index, row in raw_trial_data.iterrows():
        unique_channels.update(row['SpikeTimes'].keys())

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

    print('---------------- Average spike rate by unit ---------------')
    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit

def compute_spike_rates_per_channel_per_monkey(raw_trial_data, valid_channels):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()

    avg_spike_rate_by_unit = pd.DataFrame(index=[])

    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}

        for channel in valid_channels:
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

    print('---------------- Average spike rate by unit ---------------')
    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit

def set_node_attributes_with_default(graph, values_dict, attribute_name, default_value=0):
    for node in graph.nodes():
        value = values_dict.get(node, default_value)
        nx.set_node_attributes(graph, {node: value}, attribute_name)


if __name__ == '__main__':
    date = "2023-10-30"
    # round = "1698696277254800_231030_160438"
    round = "1698699440778381_231030_165721"
    avg_spike_rates = compute_average_spike_rates(date, round)



# spike rate for each picture
# for index, row in raw_trial_data.iterrows():
#     for unit, data in row['SpikeTimes'].items():
#         start, stop = row['EpochStartStop']
#         duration = stop - start
#         rate = len(data)/duration
#         print(f'spike rate manually calculated: {rate}')
#         spike_rate = calculate_spike_rate(data, row['EpochStartStop'])