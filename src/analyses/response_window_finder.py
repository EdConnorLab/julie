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



if __name__ == '__main__':

    time_windows = generate_time_windows_for_given_window_size(50)

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    date = "2023-10-27"
    round_no = 2
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
    for index, row in unsorted_df.iterrows():
        threshold = 1.8
        row_array = np.array(row['fractional_change'])
        above_threshold = row_array > threshold
        crossings = np.diff(above_threshold.astype(int))
        crossing_up_indices = np.where(crossings == 1)[0]
        crossing_down_indices = np.where(crossings == -1)[0] + 1
        plt.plot(time_in_sec, row_array)

        # Ensure the index does not exceed the array bounds
        if crossing_down_indices[-1] >= len(row_array):
            crossing_down_indices = crossing_down_indices[:-1]

        if crossing_up_indices.size > 0 and crossing_down_indices.size > 0:
            if crossing_up_indices[0] > crossing_down_indices[0]:
                crossing_down_indices = crossing_down_indices[1:]
            start_index = crossing_up_indices[0]
            end_index = crossing_down_indices[0] if crossing_down_indices.size > 0 else -1

            start_point = (time_in_sec[start_index], row_array[start_index])
            end_point = (time_in_sec[end_index], row_array[end_index])
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.plot(time_in_sec[start_index], row_array[start_index], 'go', label='Start of Window')  # Green dot
            plt.plot(time_in_sec[end_index], row_array[end_index], 'mo', label='End of Window')  # Magenta dot
            plt.yticks(np.arange(min(plt.ylim()), max(plt.ylim()), 0.5))
            plt.grid(axis='y', linestyle='--', linewidth=0.5)
            plt.title(f'{index}')
        plt.show()

    # results_df_final = drop_duplicate_channels_with_matching_time_window(all_anova_sig_results)

    # set threshold to 1.8?

