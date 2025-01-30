from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from intan.channels import Channel

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from excel_data_reader import ExcelDataReader
from initial_4feature_lin_reg import get_spike_count_for_single_neuron_with_time_window
from monkey_names import Zombies
from response_window_finder import compute_fractional_and_rate_of_change
from spike_count import get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files


def cusum(data, max_count, k, h):
    """
    Perform CUSUM change detection.

    Parameters:
        data (np.array): Input data.
        target_mean (float): Target or reference mean.
        k (float): Reference value or allowance.
        h (float): Threshold for detecting change.

    Returns:
        cusum_pos (np.array): Positive CUSUM statistic.
        cusum_neg (np.array): Negative CUSUM statistic.
        change_points (list): Indices where change is detected.
    """
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    change_points = []

    for t in range(1, len(data)):
        cusum_pos[t] = max(0, cusum_pos[t - 1] + data[t] - k)
        cusum_neg[t] = max(0, cusum_neg[t - 1] - data[t] - k)

        if cusum_pos[t] > h or cusum_neg[t] > h:
            change_points.append(t)
    if max_count < 3:
        change_points = []

    return cusum_pos, cusum_neg, change_points


def extract_consecutive_ranges(numbers):
    """
    Extracts ranges of consecutive numbers from a list of numbers.

    Parameters:
    numbers (list): A list of integers.

    Returns:
    list: A list of tuples, where each tuple contains the first and last consecutive integers from the input list,
          excluding ranges where the start and end are the same.
    """
    result = []
    if len(numbers) > 0:
        numbers = sorted(numbers)
        start = numbers[0]
        end = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] == end + 1:
                end = numbers[i]
            else:
                if start != end:  # Only append if start and end are not the same
                    result.append((start, end))
                start = end = numbers[i]
        if start != end:  # Check again for the last range
            result.append((start, end))
    return result


def extract_values_from_ranges(ranges, values):
    """
    Extracts elements from a list based on the start and end indices provided in ranges.

    Parameters:
    ranges (list): A list of tuples with start and end indices.
    values (list): A list from which elements are extracted based on the indices.

    Returns:
    list: A list of tuples with elements from the values list based on the given ranges.
    """
    extracted_values = []
    for start, end in ranges:
        if start <= len(values) and end < len(values):
            extracted_values.append((values[start], values[end]))
    return extracted_values

def fill_missing_ones(numbers):
    if not numbers:
        return []

    filled = []
    i = 0
    while i < len(numbers) - 1:
        filled.append(numbers[i])
        if numbers[i + 1] == numbers[i] + 2:
            filled.append(numbers[i] + 1)
        i += 1
    filled.append(numbers[len(numbers) - 1])
    return filled

if __name__ == '__main__':
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    excel = pd.ExcelFile("/home/connorlab/Documents/GitHub/Julie/window_finder_results/Cortana_Windows.xlsx")
    all_rounds = excel.parse('AllRounds')
    # print(all_rounds)
    # date = "2023-09-26"
    # round_no = 3
    results = []
    for index, row in all_rounds.iterrows():
        date = str(row['Date'])
        round_no = row['Round']
        date_only = row['Date'].strftime('%Y-%m-%d')
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

        time_chunk_size = 0.05  # in sec


        spike_counts = get_spike_counts_for_time_chunks(zombies, raw_unsorted_data, valid_channels, time_chunk_size)
        fractional_and_rate_of_change = compute_fractional_and_rate_of_change(raw_unsorted_data, zombies,
                                                                              valid_channels, time_chunk_size)
        spike_counts['total_sum'] = fractional_and_rate_of_change['total_sum']
        time = np.arange(0.05, 3.50, time_chunk_size)

        rounded_time = np.round(time, 2)
        spike_counts['response_windows'] = None
        for index, row in spike_counts.iterrows():
            data = row['total_sum']

            std_dev = np.std(data)
            if std_dev > 0:
                mean = np.mean(data) # baseline mean
                normalized_data = (data - mean)/std_dev
                max_data = max(data)
                # Parameters
                k = 0.9  # sensitivity parameter
                h = 0.3  # threshold

                cusum_pos, cusum_neg, change_points = cusum(normalized_data, max_data, k, h)
                # final_change_points = fill_missing_ones(change_points)
                final_change_points = change_points
                windows = extract_consecutive_ranges(final_change_points)

                time_windows = extract_values_from_ranges(windows, rounded_time)
                print(f"---------------- {date} round no. {round_no} {index}----------------")
                print(windows)
                print(time_windows)
                print(final_change_points)

                # Extract y-values from data at these indices
                y_values_at_change_points = [data[i] for i in final_change_points]
                t_values_at_change_points = [rounded_time[i] for i in final_change_points]

                # # Plotting
                # plt.figure(figsize=(12, 6))
                # plt.plot(time[:len(data)], data, label='Total Spike Count')
                # plt.plot(time[:len(data)], cusum_pos, label='CUSUM+', linestyle='--')
                # plt.plot(time[:len(data)], cusum_neg, label='CUSUM-', linestyle='--')
                # plt.scatter(t_values_at_change_points, y_values_at_change_points, color='red', zorder=5)
                # plt.axhline(y=k, color='green', linestyle='--', label='Threshold')
                # # Using your function to get consecutive ranges
                # consecutive_ranges = extract_consecutive_ranges(change_points)
                # print(f"consecutive ranges: {consecutive_ranges}")
                # overall_max = np.maximum.reduce([data, cusum_pos, cusum_neg])
                # # Shading consecutive ranges
                # for start, end in consecutive_ranges:
                #     plt.fill_betweenx([0, max(overall_max)], rounded_time[start], rounded_time[end], color='red', alpha=0.4)
                #
                # plt.title(f'{date_only} Round {round_no} {index} CUSUM Test for Change Detection')
                # plt.xlabel('Time')
                # plt.ylabel('Value')
                # plt.legend()
                # plt.show()

                if len(time_windows) > 0:
                    results.append({
                        'Date': date_only,
                        'Round No.': round_no,
                        'Cell': str(index),
                        'Time Window': time_windows
                    })

    results_df = pd.DataFrame(results)
    results_sorted = results_df.sort_values(by=['Date', 'Round No.', 'Cell'])
    results_sorted.to_excel('cusum_window_before_explode.xlsx')

    print(results_sorted.head())
    results_expanded = results_sorted.explode('Time Window')
    results_expanded.to_excel('cusum_window_after_explode.xlsx')
