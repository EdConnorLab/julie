import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from clat.intan.channels import Channel

import spike_count
import spike_rate_analysis
from analyses.single_channel_analysis import read_pickle, get_spike_count
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis
from monkey_names import Monkey
from recording_metadata_reader import RecordingMetadataReader
from spike_rate_analysis import read_sorted_data
from scipy.stats import f_oneway


def perform_anova_on_dataframe_rows(df):
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
        # if p_val < 0.05:
        #     significant_results.append((date, round_no, index, p_val))
    return results, significant_results


def perform_anova_on_dataframe_rows_for_time_windowed(df):
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
        results.append({'Date': row['Date'], 'Round No.': row['Round No.'],
                        'Time Window': row['Time Window'],
                        'Cell': index, 'F Value': f_val, 'P Value': p_val})
        if p_val < 0.05:

            # Collect significant result data
            significant_results.append({
                'Date': row['Date'],
                'Round No.': row['Round No.'],
                'Time Window': row['Time Window'],
                'Cell': index,
                'P Value': p_val
            })
    results_df = pd.DataFrame(results)
    significant_results_df = pd.DataFrame(significant_results)
    return results_df, significant_results_df

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
        # results.append((f_stat, p_value))
        if p_value < 0.05 and not math.isnan(f_stat):
            results.append((index, f_stat, p_value))
            print(f"Row {index}: F-statistic = {f_stat}, p-value = {p_value}")
            total_sig += 1
    return results, total_sig


def convert_to_enum(channel_str):
    enum_name = channel_str.split('.')[1]
    return getattr(Channel, enum_name)

if __name__ == '__main__':
    '''
    Date: 2024-04-29
    ANOVA for selected cells from Ed (time windowed)
    '''
    zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]

    metadata_reader = RecordingMetadataReader()
    raw_metadata = metadata_reader.get_raw_data()
    metadata_for_prelim_analysis = raw_metadata.parse('Cells_fromEd')
    metadata_subset = metadata_for_prelim_analysis[['Date', 'Round No.', 'Cell',
                                                    'Time Window Start', 'Time Window End', 'Location']]
    metadata_cleaned = metadata_subset.dropna()

    # subset sorted and unsorted
    mask = metadata_cleaned['Cell'].str.contains('Unit')
    sorted_cells = metadata_cleaned[mask]
    unsorted_cells = metadata_cleaned[~mask]

    # unsorted
    unsorted_cells['Cell'] = unsorted_cells['Cell'].apply(convert_to_enum)
    rows_for_unsorted = []
    for index, row in unsorted_cells.iterrows():
        time_window = (row['Time Window Start'], row['Time Window End'])
        spike_count_unsorted = spike_count.get_spike_count_for_unsorted_cell_with_time_window(row['Date'].strftime('%Y-%m-%d'),
                                                                                  row['Round No.'], row['Cell'],
                                                                                  time_window)
        rows_for_unsorted.append(spike_count_unsorted)
    spike_count_for_unsorted = pd.concat(rows_for_unsorted)
    zombies_columns = [col for col in zombies if col in spike_count_for_unsorted.columns]
    required_columns = zombies_columns + [col for col in ['Date', 'Round No.', 'Time Window'] if
                                          col in spike_count_for_unsorted.columns]
    unsorted_spike_count_zombies = spike_count_for_unsorted[required_columns]
    print(unsorted_spike_count_zombies)

    # Perform ANOVA on unsorted
    anova_results, sig_results = perform_anova_on_dataframe_rows_for_time_windowed(unsorted_spike_count_zombies)
    anova_results.to_csv('Unsorted_windowed_ANOVA_results.csv')
    sig_results.to_csv('Unsorted_windowed_ANOVA_significant_results.csv')

    # sorted
    rows_for_sorted = []
    for index, row in sorted_cells.iterrows():
        time_window = (row['Time Window Start'], row['Time Window End'])
        spike_count_sorted = spike_count.get_spike_count_for_sorted_cell_with_time_window(row['Date'].strftime('%Y-%m-%d'),
                                                                              row['Round No.'], row['Cell'],
                                                                              time_window)
        rows_for_sorted.append(spike_count_sorted)
    spike_count_for_sorted = pd.concat(rows_for_sorted)
    zombies_columns = [col for col in zombies if col in spike_count_for_sorted.columns]
    required_columns = zombies_columns + [col for col in ['Date', 'Round No.', 'Time Window'] if
                                          col in spike_count_for_sorted.columns]
    sorted_spike_count_zombies = spike_count_for_sorted[required_columns]
    print(sorted_spike_count_zombies)

    # Perform ANOVA on sorted
    anova_results, sig_results = perform_anova_on_dataframe_rows_for_time_windowed(sorted_spike_count_zombies)
    anova_results.to_csv('Sorted_windowed_ANOVA_results.csv')
    sig_results.to_csv('Sorted_windowed_ANOVA_significant_results.csv')

    '''
    ANOVA or PermANOVA on all rounds from metadata
    '''
    # metadata_for_analysis = get_metadata_for_preliminary_analysis()
    # total_sig_cells = 0
    # all_significant_results = []
    # for _, row in metadata_for_analysis.iterrows():
    #     date = row['Date'].strftime('%Y-%m-%d')
    #     round_no = row['Round No.']
    #     spike_count_dataframe = get_spike_count_for_each_trial(date, round_no)
    #     # anova_results, sig_results = perform_anova_on_rows(spike_count_dataframe)
    #     results, sig = perform_anova_permutation_test_on_rows(spike_count_dataframe, num_permutations=1000)
    #     for result in results:
    #         index, f_stat, p_value = result
    #         to_be_saved = [date, round_no, index, f_stat, p_value]
    #         all_significant_results.append(to_be_saved)
    #     total_sig_cells = total_sig_cells + sig
    #     # print(f'For {date} Round No. {round_no}')
    #     # all_significant_results.extend(sig_results)
    #     # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'P-Value'])
    #     # results_file_path = 'significant_anova_results.xlsx'
    #     # results_df.to_excel(results_file_path, index=False)
    # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'F-statistics', 'P-Value'])
    # results_file_path = 'significant_anova_results.xlsx'
    # results_df.to_excel(results_file_path, index=False)
    # print(total_sig_cells)