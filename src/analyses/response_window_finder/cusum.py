import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from monkey_names import Zombies
from recording_metadata_reader import RecordingMetadataReader
from spike_count import get_spike_counts_for_time_chunks, get_spike_count_for_single_neuron_with_time_window
from spike_rate_computation import get_raw_data_and_channels_from_files


def cusum(data, max_count, k, h):
    """
    Perform CUSUM change detection.

    Parameters:
        data (np.array): Input data.
        max_count (float): Target or reference mean.
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


def compute_total_sum_of_spikes(raw_data, monkeys, channels, chunk_size):

    spike_counts = get_spike_counts_for_time_chunks(monkeys, raw_data, channels, chunk_size)
    spike_counts['total_sum'] = spike_counts.apply(lambda row: [sum(elements) for elements in zip(*row)], axis=1)

    return spike_counts


def plot_spike_count_with_response_windows(time, data, cusum_pos, cusum_neg, change_points):
    # Plotting
    y_values_at_change_points = [data[i] for i in final_change_points]
    t_values_at_change_points = [time[i] for i in final_change_points]

    plt.figure(figsize=(12, 6))
    plt.plot(time[:len(data)], data, label='Total Spike Count')
    plt.plot(time[:len(data)], cusum_pos, label='CUSUM+', linestyle='--')
    plt.plot(time[:len(data)], cusum_neg, label='CUSUM-', linestyle='--')
    plt.scatter(t_values_at_change_points, y_values_at_change_points, color='red', zorder=5)
    plt.axhline(y=k, color='green', linestyle='--', label='Threshold')
    # Using your function to get consecutive ranges
    consecutive_ranges = extract_consecutive_ranges(change_points)
    print(f"consecutive ranges: {consecutive_ranges}")
    overall_max = np.maximum.reduce([data, cusum_pos, cusum_neg])
    # Shading windows
    for start, end in consecutive_ranges:
        plt.fill_betweenx([0, max(overall_max)], time[start], time[end], color='red', alpha=0.4)

    plt.title(f'{date_only} Round {round_no} {index} --- windows based on CUSUM algorithm')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def save_cusum_results(response_windows_found):
    results_df = pd.DataFrame(response_windows_found)
    results_sorted = results_df.sort_values(by=['Date', 'Round No.', 'Cell'])
    results_sorted.to_excel('cusum_window_before_explode.xlsx')

    print(results_sorted.head())
    results_expanded = results_sorted.explode('Time Window')
    results_expanded.to_excel('cusum_window_after_explode.xlsx')
    print('cusum results saved!')

if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    recording_metadata = RecordingMetadataReader().get_raw_data()
    all_rounds = recording_metadata.parse("AllRounds")

    time_chunk_size = 0.05  # in sec
    time = np.arange(time_chunk_size, 3.50, time_chunk_size)
    rounded_time = np.round(time, 2)

    results = []
    for _, row in all_rounds.iterrows():
        date = str(row['Date'])
        round_no = row['Round']
        date_only = row['Date'].strftime('%Y-%m-%d')
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
        spike_counts = compute_total_sum_of_spikes(raw_unsorted_data, zombies, valid_channels, time_chunk_size)

        spike_counts['response_windows'] = None
        for index, r in spike_counts.iterrows():
            data = r['total_sum']
            std_dev = np.std(data)
            if std_dev > 0:
                mean = np.mean(data) # baseline mean
                normalized_data = (data - mean)/std_dev
                max_data = max(data)
                # Parameters
                k = 0.9  # sensitivity parameter
                h = 0.3  # threshold

                cusum_pos, cusum_neg, change_points = cusum(normalized_data, max_data, k, h)
                windows = extract_consecutive_ranges(change_points)
                time_windows = extract_values_from_ranges(windows, rounded_time)
                print(f"---------------- {date} round no. {round_no} {index}----------------")
                print(windows)
                print(time_windows)

                # Plotting
                # plot_spike_count_with_response_windows(rounded_time, data, cusum_pos, cusum_neg, change_points)

                if len(time_windows) > 0:
                    results.append({
                        'Date': date_only,
                        'Round No.': round_no,
                        'Cell': str(index),
                        'Time Window': time_windows
                    })

    save_cusum_results(results)

    '''
        Date Created: 2025-01-29
        ANOVA for windows found from cusum algorithm
    '''
    results['Time Window'] = results['Time Window'].apply(
        lambda s: tuple(int(float(num) * 1000) for num in s.strip('()').split(',')))

    print(
        "--------------------------------------------- cusum windows ----------------------------------------------------------")
    print(results)

    cusum_spike_count = get_spike_count_for_single_neuron_with_time_window(results)
    print(cusum_spike_count)

    zombies_columns = [col for col in zombies if col in cusum_spike_count.columns]
    additional_columns = ['Date', 'Round No.', 'Time Window']
    zombies_cusum_spike_count = cusum_spike_count[zombies_columns + additional_columns]
    cusum_anova_results, cusum_sig_results = perform_anova_on_dataframe_rows_for_time_windowed(
        zombies_cusum_spike_count)

    print('------------------------------------ cusum window results -----------------------------------')
    # print(cusum_anova_results)
    print(cusum_sig_results)
    # cusum_anova_results.to_excel('cusum_anova_results.xlsx')
    # cusum_sig_results.to_excel('cusum_sig_results.xlsx')

