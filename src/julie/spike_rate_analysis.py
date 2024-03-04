import os
from pathlib import Path

import pandas as pd
import networkx as nx
from clat.intan.channels import Channel
from clat.intan.rhd import load_intan_rhd_format

from julie.single_channel_analysis import read_pickle, calculate_spike_rate
from julie.single_unit_analysis import calculate_spike_timestamps
from metadata_reader import RecordingMetadataReader


def compute_average_spike_rates(date, round_number):
    metadata_reader = RecordingMetadataReader()

    pickle_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_number) + ".pk1"
    compiled_dir = (Path(__file__).parent.parent.parent / 'compiled').resolve()
    pickle_filepath = os.path.join(compiled_dir, pickle_filename)
    raw_trial_data = read_pickle(pickle_filepath)

    intan_dir = metadata_reader.get_intan_folder_name_for_specific_round(date, round_number)
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = Path(os.path.join(cortana_path, date, intan_dir))

    # Check if the experimental round is sorted
    sorted = round_path / 'sorted_spikes.pkl'
    if sorted.exists():
        print("Sorted round...")
        sorted_data = read_sorted_data(round_path)
        channels_with_units = (sorted_data['SpikeTimes'][0].keys())
        sorted_channels = [int(s.split('_')[1][1:]) for s in channels_with_units]
        sorted_enum_channels = [Channel(f'C-{channel:03}') for channel in sorted_channels]
        channels = metadata_reader.get_valid_channels(date, round_number)
        print(f"channels: {channels}")
        valid_channels = set(channels) - set(sorted_enum_channels)
        print("Computing average spike rate for sorted data ...")
        sorted_data_spike_rates = compute_spike_rates_per_channel_per_monkey_for_sorted_data(sorted_data)
        print("Computing average spike rate for the rest of the data...")
        raw_data_spike_rates = compute_spike_rates_per_channel_per_monkey_for_raw_data(raw_trial_data, valid_channels)
        average_spike_rates = pd.concat([sorted_data_spike_rates, raw_data_spike_rates])
    else:
        valid_channels = set(metadata_reader.get_valid_channels(date, round_number))
        print(f"channels: {valid_channels}")
        print("Computing average spike rate for unsorted data ...")
        raw_data_spike_rates = compute_spike_rates_per_channel_per_monkey_for_raw_data(raw_trial_data, valid_channels)
        average_spike_rates = raw_data_spike_rates

    print('--------------------average spike rate dataframes combined--------------------')
    print(average_spike_rates)

    return average_spike_rates


def read_sorted_data(round_path, sorted_spikes_filename='sorted_spikes.pkl', compiled_trials_filename='compiled.pk1'):
    compiled_trials_filepath = os.path.join(round_path, compiled_trials_filename)
    raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)
    rhd_file_path = os.path.join(round_path, 'info.rhd')
    sorted_spikes_filepath = os.path.join(round_path, sorted_spikes_filename)
    sorted_spikes = read_pickle(sorted_spikes_filepath)
    sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
    sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)
    return sorted_data


def compute_spike_rates_per_channel_per_monkey_for_sorted_data(raw_trial_data):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(raw_trial_data['SpikeTimes'][0].keys())

    print("-------------------unique channels-----------------")
    print(unique_channels)
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

    print('---------------- Average spike rate per channel per monkey for all channels ---------------')
    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit


def compute_spike_rates_per_channel_per_monkey_for_raw_data(raw_trial_data, valid_channels):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    print(f'valid channels {valid_channels}')
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
                    pass
                    # print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            monkey_specific_spike_rate[channel] = avg_spike_rate

        # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)

    print('------------- Average spike rate per channel per monkey for valid channels ------------')
    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit


def set_node_attributes_with_default(graph, values_dict, attribute_name, default_value=0):
    for node in graph.nodes():
        value = values_dict.get(node, default_value)
        nx.set_node_attributes(graph, {node: value}, attribute_name)


if __name__ == '__main__':
    # avg_spike_rates = compute_average_spike_rates("2023-10-24", 2)
    avg_spike_rates = compute_average_spike_rates("2023-09-29", 2) # one with the problem

# spike rate for each picture
# for index, row in raw_trial_data.iterrows():
#     for unit, data in row['SpikeTimes'].items():
#         start, stop = row['EpochStartStop']
#         duration = stop - start
#         rate = len(data)/duration
#         print(f'spike rate manually calculated: {rate}')
#         spike_rate = calculate_spike_rate(data, row['EpochStartStop'])
