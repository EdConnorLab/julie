import numpy as np
import pandas as pd
from clat.intan.channels import Channel
from matplotlib import pyplot as plt

from anova_scan import generate_time_windows_for_given_window_size
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis
from monkey_names import Zombies
from spike_count import get_spike_counts_for_given_time_window, add_metadata_to_spike_counts, \
    get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files

def sum_lists(row):
    # Using zip to pair up corresponding elements and sum them
    return [sum(elements) for elements in zip(*row)]

def fractional_change(lst):
    return [(lst[i+1] - lst[i]) / lst[i] if lst[i] != 0 else 0 for i in range(len(lst)-1)]

def calculate_fractional_change(data, monkeys, channels, chunk_size):
    spike_counts = get_spike_counts_for_time_chunks(monkeys, data, channels, chunk_size)
    spike_counts['total_sum'] = spike_counts.apply(lambda row: sum_lists(row), axis=1)
    spike_counts['fractional_change'] = spike_counts['total_sum'].apply(fractional_change)
    return  spike_counts[['total_sum', 'fractional_change']]


def plot_fractional_change_data(date, round_no, data, time_chunk_size):
    max_length = data['fractional_change'].apply(len).max()
    time_in_sec = np.arange(time_chunk_size, time_chunk_size * (max_length+1), time_chunk_size)
    for index, row in data.iterrows():
        plt.plot(time_in_sec, row['fractional_change'], label=index)
    plt.title(f'{date} Round {round_no} Fractional Change by Channel')
    plt.legend()
    plt.yticks(np.arange(min(plt.ylim()), max(plt.ylim()), 0.5))
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.axhline(y=1.8, color='black') # threshold
    plt.show()

def find_pairs(crossings):
    pairs = []
    start_index = None
    for i in range(len(crossings)):
        if crossings[i] == 1:
            start_index = i
        elif crossings[i] == -1 and start_index is not None:
            pairs.append((start_index, i))
            start_index = None
        elif crossings[i] == -1 and start_index is None:
            pairs.append((0, i))
            start_index = None
    if start_index is not None:
        pairs.append((start_index, len(crossings)-1))
    return pairs
#
def convert_to_time_windows(index_pairs, time_increments):
    response_windows = []
    for start, end in index_pairs:
        if start == 0 and start == end:
            response_windows.append((0, round(time_increments[start], 1)))
        elif start == end:
            response_windows.append((round(time_increments[start], 1), round(time_increments[-1], 1)))
        else:
            response_windows.append((round(time_increments[start], 1), round(time_increments[end], 1)))
    return response_windows
if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    date = "2023-11-27"
    round_no = 3
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

    time_chunk_size = 0.2  # in sec
    unsorted_df = calculate_fractional_change(raw_unsorted_data, zombies, valid_channels, time_chunk_size)
    # plot_fractional_change_data(date, round_no, unsorted_df, time_chunk_size)

    if sorted_data is not None:
        unique_channels = set()
        unique_channels.update(sorted_data['SpikeTimes'][0].keys())
        sorted_df = calculate_fractional_change(sorted_data, zombies, unique_channels, time_chunk_size)
        # plot_fractional_change_data(date, round_no, sorted_df, time_chunk_size)

    max_length = unsorted_df['fractional_change'].apply(len).max()
    time_in_sec = np.arange(time_chunk_size, time_chunk_size * (max_length+1), time_chunk_size)
    time_starting_with_zero = np.arange(0, time_chunk_size * (max_length+2), time_chunk_size)
    print('time in sec')
    print(time_in_sec)
    threshold = 1.8


    for index, row in unsorted_df.iterrows():
        row_array = np.array(row['fractional_change'])
        above_threshold = row_array > threshold
        crossings = np.diff(above_threshold.astype(int))
        pairs = find_pairs(crossings)
        print(pairs)
        time_windows = convert_to_time_windows(pairs, time_in_sec)
        print(time_windows)

        plt.plot(time_in_sec[:len(row_array)], row_array)
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        # TODO: try plotting!
        for time_window in time_windows:
            start, end = time_window
            print(start)
            print(end)
            start_index = np.where(np.isclose(time_starting_with_zero, start, atol=1e-5))[0][0]
            end_index = np.where(np.isclose(time_starting_with_zero, end, atol=1e-5))[0][0]
            print(f"indices: {start_index}-{end_index}")
            plt.plot(start, row_array[start_index],'go')
            plt.plot(end, row_array[end_index], 'bo')
        plt.show()
    #     for up_idx in crossing_up_indices:
    #         # Find the first down index that is greater than the current up index
    #         down_idx = crossing_down_indices[crossing_down_indices > up_idx]
    #         if down_idx.size > 0:
    #             start_time = time_in_sec[up_idx]
    #             end_time = time_in_sec[down_idx[0]]
    #             plt.plot(start_time, row_array[up_idx], 'go', label='Start of Window')  # Green dot
    #             plt.plot(end_time, row_array[down_idx[0]], 'mo', label='End of Window')  # Magenta dot
    #             plt.title(f'{index} - Window: Start at {start_time}s, End at {end_time}s')
    #             print(f'Window starts at {start_time}s and ends at {end_time}s')
    #
    #     plt.yticks(np.arange(min(plt.ylim()), max(plt.ylim()), 0.5))
    #     plt.grid(axis='y', linestyle='--', linewidth=0.5)
    #     plt.legend()


    # results_df_final = drop_duplicate_channels_with_matching_time_window(all_anova_sig_results)

    # set threshold to 1.8?

