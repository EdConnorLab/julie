import pandas as pd

import spike_rate_computation
from channel_enum_resolvers import drop_duplicate_channels, is_channel_in_dict, get_value_from_dict_with_channel
from single_channel_analysis import read_pickle, get_spike_count
from data_readers.recording_metadata_reader import RecordingMetadataReader
from spike_rate_computation import read_sorted_data


def add_metadata_to_spike_counts(spike_count_df, date, round_number, time_window):
    """
    Adds metadata to spike counts DataFrame.
    """
    spike_count_df['Date'] = date
    spike_count_df['Round No.'] = round_number
    spike_count_df['Time Window'] = [time_window]  * len(spike_count_df)
    return spike_count_df


def get_spike_count_for_unsorted_cell_with_time_window(date, round_number, unsorted_cell, time_window):
    reader = RecordingMetadataReader()
    pickle_filepath, _, _ = reader.get_metadata_for_spike_analysis(date, round_number)
    raw_trial_data = read_pickle(pickle_filepath)
    raw_data_spike_counts = count_spikes_for_specific_cell_time_windowed(raw_trial_data, unsorted_cell, time_window)
    raw_data_spike_counts = add_metadata_to_spike_counts(raw_data_spike_counts, date, round_number, time_window)
    return raw_data_spike_counts


def get_spike_count_for_sorted_cell_with_time_window(date, round_number, sorted_cell, time_window):
    reader = RecordingMetadataReader()
    _, _, sorted_data_path = reader.get_metadata_for_spike_analysis(date, round_number)
    sorted_data = spike_rate_computation.read_sorted_data(sorted_data_path)
    sorted_data_spike_counts = count_spikes_for_specific_cell_time_windowed(sorted_data, sorted_cell, time_window)
    sorted_data_spike_counts = add_metadata_to_spike_counts(sorted_data_spike_counts, date, round_number, time_window)
    return sorted_data_spike_counts



def get_spike_count_for_each_trial(date, round_number):
    """
    Return: number of spikes for each trial for a given experimental round
    """
    reader = RecordingMetadataReader()
    pickle_filepath, valid_channels, round_path = reader.get_metadata_for_spike_analysis(date, round_number)

    raw_trial_data = read_pickle(pickle_filepath)
    raw_data_spike_counts = count_spikes_from_raw_trial_data(raw_trial_data, valid_channels)

    # Check if the experimental round is sorted
    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():

        sorted_data = read_sorted_data(round_path)
        sorted_data_spike_counts = count_spikes_from_sorted_data(sorted_data)

        raw_data_spike_counts = drop_duplicate_channels(raw_data_spike_counts, sorted_data)
        spike_counts = pd.concat([sorted_data_spike_counts, raw_data_spike_counts])
    else:
        spike_counts = raw_data_spike_counts

    return spike_counts


def generate_time_windows_for_given_window_size(window_size):
    """
    Generate a list of time windows (tuples) representing ranges with a specified window size.

    Parameters:
    window_size (int): Must be a positive integer and should not exceed 2000 ms.

    Returns:
    time_windows (list of tuples): Each tuple represents a range.

    Raises:
    ValueError:
        If the window_size is not a positive integer or exceeds 2000.
    """
    if not isinstance(window_size, int) or window_size <= 0 or window_size > 2000:
        raise ValueError("Window size must be a positive integer and not exceed 2000 ms.")

    time_windows = [(i, i + window_size) for i in range(0, 2000, window_size)]

    return time_windows


def count_spikes_for_specific_cell_time_windowed(raw_data, cell, time_window):
    unique_monkeys = raw_data['MonkeyName'].dropna().unique().tolist()
    spike_count_per_channel = pd.DataFrame()
    for monkey in unique_monkeys:
        monkey_data = raw_data[raw_data['MonkeyName'] == monkey]
        monkey_spike_counts = {}
        spike_counts = []
        for index, row in monkey_data.iterrows():
            if is_channel_in_dict(cell, row['SpikeTimes']):
                data = get_value_from_dict_with_channel(cell, row['SpikeTimes'])
                if time_window is not None:
                    window_start_micro, window_end_micro = time_window
                    window_start_sec = window_start_micro * 0.001
                    window_end_sec = window_end_micro * 0.001
                    start_time, end_time = row['EpochStartStop']
                    spike_counts.append(
                        get_spike_count(data, (start_time + window_start_sec, start_time + window_end_sec)))
                else:
                    spike_counts.append(get_spike_count(data, row['EpochStartStop']))
            else:
                print(f"No data for {cell} in row {index}")
        monkey_spike_counts[cell] = spike_counts
        spike_count_per_channel[monkey] = pd.Series(monkey_spike_counts)
    return spike_count_per_channel


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

