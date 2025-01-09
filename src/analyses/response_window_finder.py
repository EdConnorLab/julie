from datetime import datetime

import numpy as np
import pandas as pd
from clat.intan.channels import Channel
from matplotlib import pyplot as plt

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from anova_scan import generate_time_windows_for_given_window_size
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis, \
    get_spike_count_for_single_neuron_with_time_window
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

def find_boundary_pairs(fractional_changes, up_threshold, down_threshold):
    pairs = []
    i = 0
    while i < len(fractional_changes):
        if fractional_changes[i] > up_threshold:  # Check for an up boundary
            up_boundary = i
            # Look for the next down boundary after the up boundary
            for j in range(up_boundary + 1, len(fractional_changes)):
                if fractional_changes[j] < down_threshold:
                    down_boundary = j
                    pairs.append((up_boundary, down_boundary))
                    i = down_boundary  # Start next search after the found down boundary
                    break
            else:
                # If no down boundary is found, break the loop to avoid infinite looping
                break
        i += 1
    return pairs

def round_tuple_values(t):
    return tuple(round(x, 2) for x in t)

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

    date = "2023-10-31"
    round_no = 2
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

    time_chunk_size = 0.05  # in sec
    up_threshold = 0.5
    down_threshold = -0.5

    unsorted_df = calculate_fractional_change(raw_unsorted_data, zombies, valid_channels, time_chunk_size)
    # plot_fractional_change_data(date, round_no, unsorted_df, time_chunk_size)

    if sorted_data is not None:
        unique_channels = set()
        unique_channels.update(sorted_data['SpikeTimes'][0].keys())
        sorted_df = calculate_fractional_change(sorted_data, zombies, unique_channels, time_chunk_size)
        # plot_fractional_change_data(date, round_no, sorted_df, time_chunk_size)

    max_length = unsorted_df['fractional_change'].apply(len).max()
    time_in_sec = np.arange(0, time_chunk_size * (max_length+2), time_chunk_size)
    unsorted_df['response_windows'] = None
    for index, row in unsorted_df.iterrows():
        # print(index)
        fractional_change_row = np.array(row['fractional_change'])
        spike_total_row = np.array(row['total_sum'])
        boundary_pairs_found = find_boundary_pairs(fractional_change_row, up_threshold, down_threshold)
        # print(boundary_pairs_found)
        time_window_tuple_list = []
        for pair in boundary_pairs_found:
            time_window_index = tuple(x + 1 for x in pair)
            time_window_tuple = tuple(time_in_sec[i] for i in time_window_index)
            time_window_tuple = round_tuple_values(time_window_tuple)
            time_window_tuple = tuple(x*1000 for x in time_window_tuple) # convert to microseconds
            time_window_tuple_list.append(time_window_tuple)
        # print(time_window_tuple_list)
        if time_window_tuple_list:  # Only update if there is something to update
            unsorted_df.at[index, 'response_windows'] = time_window_tuple_list

        # print(unsorted_df)
        # plt.plot(time_in_sec[1:len(fractional_change_row) + 1], fractional_change_row)
        # plt.plot(time_in_sec[:len(spike_total_row)], spike_total_row)
        #
        # plt.show()


    # Perform ANOVA on response windows
    # Initializing the expanded DataFrame
    expanded = []
    for index, row in unsorted_df.iterrows():
        if row['response_windows'] is not None:
            for window in row['response_windows']:
                expanded.append({
                    'Date': datetime.strptime(date, "%Y-%m-%d"),
                    'Round No.': round_no,
                    'Cell': str(index),
                    'Time Window': window,
                    'Time Window Start': window[0],
                    'Time Window End': window[1]
                })
    expanded_df = pd.DataFrame(expanded)
    print(expanded_df)
    # calculate spike count
    spike_count = get_spike_count_for_single_neuron_with_time_window(expanded_df)
    # print(spike_count)
    required_columns = ['Date', 'Round No.', 'Time Window']
    zombies_columns = [col for col in zombies + required_columns if col in spike_count.columns]
    zombies_spike_count = spike_count[zombies_columns]

    print(zombies_spike_count)
    if all(all(item == 0 for item in sublist) for sublist in zombies_spike_count.apply(pd.Series).stack()):
        print("All entries are zero. Skipping analysis.")
    else:
        anova_results, sig_results = perform_anova_on_dataframe_rows_for_time_windowed(zombies_spike_count)
        print('-------------------------- ANOVA passed ----------------------------')
        print(sig_results)

    # Questions:
    # What to do with very small windows (e.g., 50-100 ms)
    # Changing increment size (50 ms) since it seems it's too small?






    # results_df_final = drop_duplicate_channels_with_matching_time_window(all_anova_sig_results)
