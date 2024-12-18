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



if __name__ == '__main__':


    time_windows = generate_time_windows_for_given_window_size(50)

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    date = "2023-10-04"
    round_no = 1
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

    chunk_size = 0.05
    spike_counts_for_unsorted = get_spike_counts_for_time_chunks(zombies, raw_unsorted_data, valid_channels, chunk_size)
    spike_counts_for_unsorted['total_sum'] = spike_counts_for_unsorted.apply(lambda row: sum_lists(row), axis=1)
    spike_counts_for_unsorted['fractional_change'] = spike_counts_for_unsorted['total_sum'].apply(fractional_change)
    print(spike_counts_for_unsorted[['total_sum', 'fractional_change']])
    subset_df = spike_counts_for_unsorted[['total_sum', 'fractional_change']]
    for index, row in subset_df.iterrows():
        plt.plot(row['fractional_change'], label=index)
    plt.title('Fractional Change by Channel')
    plt.legend()
    plt.show()

    if sorted_data is not None:
        unique_channels = set()
        unique_channels.update(sorted_data['SpikeTimes'][0].keys())
        spike_counts_for_sorted = get_spike_counts_for_time_chunks(zombies, sorted_data, unique_channels, chunk_size)
        spike_counts_for_sorted['total_sum'] = spike_counts_for_sorted.apply(lambda row: sum_lists(row), axis=1)
        spike_counts_for_sorted['fractional_change'] = spike_counts_for_sorted['total_sum'].apply(fractional_change)
        print(spike_counts_for_sorted[['total_sum', 'fractional_change']])
        subset_sorted = spike_counts_for_sorted[['total_sum', 'fractional_change']]
        for index, row in subset_sorted.iterrows():
            plt.plot(row['fractional_change'], label = index)
        plt.title('Fractional Change by Channel')
        plt.legend()
        plt.show()
    # results_df_final = drop_duplicate_channels_with_matching_time_window(all_anova_sig_results)
    # print(results_df_final)
    # results_df_final.to_excel('/home/connorlab/Documents/GitHub/Julie/anova_scan/ER_ANOVA_scan_size_300ms.xlsx')

