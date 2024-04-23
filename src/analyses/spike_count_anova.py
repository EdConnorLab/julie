import os
from pathlib import Path
import numpy as np
import pandas as pd
from clat.intan.channels import Channel

from analyses.single_channel_analysis import read_pickle, get_spike_count
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis
from recording_metadata_reader import RecordingMetadataReader
from spike_rate_analysis import read_sorted_data
from scipy.stats import f_oneway

def get_spike_count_for_each_trial(date, round_number):
    """
    Return: number of spikes for each trial for a given experimental round
    """
    metadata_reader = RecordingMetadataReader()
    pickle_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_number) + ".pk1"
    compiled_dir = (Path(__file__).parent.parent.parent / 'compiled').resolve()
    pickle_filepath = os.path.join(compiled_dir, pickle_filename)
    raw_trial_data = read_pickle(pickle_filepath)
    intan_dir = metadata_reader.get_intan_folder_name_for_specific_round(date, round_number)
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = Path(os.path.join(cortana_path, date, intan_dir))
    valid_channels = set(metadata_reader.get_valid_channels(date, round_number))

    raw_data_spike_counts = count_spikes_from_raw_trial_data(raw_trial_data, valid_channels)

    # Check if the experimental round is sorted
    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        print(f"Reading sorted data: {round_path}")
        sorted_data = read_sorted_data(round_path)
        channels_with_units = (sorted_data['SpikeTimes'][0].keys())
        sorted_channels = [int(s.split('_')[1][1:]) for s in channels_with_units]
        sorted_enum_channels = list(set([Channel(f'C-{channel:03}') for channel in sorted_channels]))
        # remove channel only if it exists as index
        for channel in sorted_enum_channels:
            if channel in raw_data_spike_counts.index:
                raw_data_spike_counts = raw_data_spike_counts.drop(channel)
        sorted_data_spike_counts = count_spikes_from_sorted_data(sorted_data)
        spike_counts = pd.concat([sorted_data_spike_counts, raw_data_spike_counts])
    else:
        spike_counts = raw_data_spike_counts

    return spike_counts


def count_spikes_from_sorted_data(sorted_data):
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    spike_count_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())
    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_counts = {}
        for channel in unique_channels:
            spike_count = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_count.append(get_spike_count(data, row['EpochStartStop']))
                else:
                   print(f"No data for {channel} in row {index}")
            monkey_specific_spike_counts[channel] = spike_count
        spike_count_by_unit[monkey] = pd.Series(monkey_specific_spike_counts)
    return spike_count_by_unit

def count_spikes_from_raw_trial_data(raw_trial_data, valid_channels):
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    spike_count_per_channel = pd.DataFrame()
    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_spike_counts = {}
        for channel in valid_channels:
            spike_counts = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_counts.append(get_spike_count(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")
            monkey_spike_counts[channel] = spike_counts
        spike_count_per_channel[monkey] = pd.Series(monkey_spike_counts)
    return spike_count_per_channel

def count_spikes_from_sorted_data(sorted_data):
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())
    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}
        for channel in unique_channels:
            spike_count = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_count.append(get_spike_count(data, row['EpochStartStop']))
                else:
                   print(f"No data for {channel} in row {index}")
            monkey_specific_spike_rate[channel] = spike_count
        spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)
    return spike_rate_by_unit


def get_value_from_dict_with_channel(channel, dictionary):
    if isinstance(channel, str):
        return dictionary[channel]
    else:
        for key, value in dictionary.items():
            if key.value == channel.value:
                return value


def is_channel_in_dict(channel, diction):
    if isinstance(channel, str):
        if channel in list(diction.keys()):
            return True
    else:
        for key in diction:
            if channel.value == key.value:
                return True


def perform_anova_on_rows(df):
    """
    Perform one-way ANOVA on rows of a DataFrame
    """
    results = []
    significant_results = []
    for index, row in df.iterrows():
        # Extract groups as lists
        groups = [group for group in row if isinstance(group, list)]
        # Perform OneWay ANOVA
        f_val, p_val = f_oneway(*groups)
        results.append((f_val, p_val))
        if p_val < 0.05:
            significant_results.append((date, round_no, index, p_val))
    return results, significant_results


def anova_permutation_test(groups, num_permutations=1000):
    data = np.concatenate(groups)
    original_group_sizes = [len(group) for group in groups]
    observed_f_stat, _ = f_oneway(*groups)

    permutation_f_stats = []
    for _ in range(num_permutations):
        np.random.shuffle(data)
        new_groups = np.split(data, np.cumsum(original_group_sizes)[:-1])
        f_stat, _ = f_oneway(*new_groups)
        permutation_f_stats.append(f_stat)

    p_value = np.mean([f_stat >= observed_f_stat for f_stat in permutation_f_stats])
    return observed_f_stat, p_value


def perform_anova_permutation_test_on_rows(df, num_permutations=1000):
    results = []
    total_sig = 0
    for index, row in df.iterrows():
        groups = [np.array(cell) for cell in row if isinstance(cell, list)]
        f_stat, p_value = anova_permutation_test(groups, num_permutations=num_permutations)
        results.append((f_stat, p_value))
        if p_value < 0.05:
            print(f"Row {index}: F-statistic = {f_stat}, p-value = {p_value}")
            total_sig += 1
    return results, total_sig



if __name__ == '__main__':
    metadata_for_analysis = get_metadata_for_preliminary_analysis()
    total_sig_cells = 0
    all_significant_results = []
    for _, row in metadata_for_analysis.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        spike_count_dataframe = get_spike_count_for_each_trial(date, round_no)
        # anova_results, sig_results = perform_anova_on_rows(spike_count_dataframe)
        results, sig = perform_anova_permutation_test_on_rows(spike_count_dataframe, num_permutations=1000)
        total_sig_cells = total_sig_cells + sig
        # print(f'For {date} Round No. {round_no}')
        # all_significant_results.extend(sig_results)
        # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'P-Value'])
        # results_file_path = 'significant_anova_results.xlsx'
        # results_df.to_excel(results_file_path, index=False)
    print(total_sig_cells)