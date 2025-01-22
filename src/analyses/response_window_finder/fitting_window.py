import numpy as np
import pandas as pd

from cusum import extract_consecutive_ranges, extract_values_from_ranges
from monkey_names import Zombies
from response_window_finder import compute_fractional_and_rate_of_change
from spike_count import get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files

def expand_window_from_max_threshold(spike_count, threshold = 0.5):
    print(spike_count)
    max_value = max(spike_count)
    all_windows = []
    window_indices = []
    if max_value != 0:
        indices_of_max = [index for index, value in enumerate(spike_count) if value == max_value]
        # print(f"max values are in {indices_of_max}")
        for i in indices_of_max:
            window_indices = [i]
            after = i + 1
            before = i - 1
            while after <= len(spike_count) - 1:
                if spike_count[after] > max_value * threshold:
                    window_indices.append(after)
                    after += 1
                else:
                    break
            while before >= 0:
                if spike_count[before] > max_value * threshold:
                    window_indices.append(before)
                    before -= 1
                else:
                    break
            if len(window_indices) > 1:
                window_indices.sort()
                spike_values_in_window = [spike_count[i] for i in window_indices]
                print(f"spike values in window {spike_values_in_window}")
                all_windows.append(window_indices)
    else:
        print("No spikes detected")
    # remove duplicates
    unique_tuples = set(tuple(x) for x in all_windows)
    unique_windows = [list(x) for x in unique_tuples]
    return unique_windows


def unique_elements(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist] # Flatten the list of lists into a single list
    unique = list(set(flat_list)) # Remove duplicates

    return sorted(unique)


def expand_window_from_dynamic_threshold(spike_count, threshold = 0.5):
    max_value = max(spike_count)
    start_threshold = 0.7
    high_value = max_value * start_threshold
    window_indices = []
    all_windows = []
    if max_value != 0:
        starting_indices = [index for ind, value in enumerate(spike_count) if value >= high_value]
        # print(f"starting indices are {starting_indices}")
        for i in starting_indices:
            initial_window_start = i
            window_indices.append(i)
            while i < len(spike_count) - 1:
                if spike_count[i + 1] > spike_count[i] * threshold:
                    window_indices.append(i + 1)
                    i += 1
                else:
                    break
            i = initial_window_start
            while i > 0:
                if spike_count[i - 1] > spike_count[i] * threshold:
                    window_indices.append(i - 1)
                    i -= 1
                else:
                    break
            if len(window_indices) > 1:
                window_indices.sort()
                all_windows.append(window_indices)
    else:
        pass
        # print(f"no spikes detected")
    # remove duplicates
    # print(f"spike count {spike_count}")
    unique_windows = unique_elements(all_windows)
    # print(f"unique windows {unique_windows}")
    final_windows = extract_consecutive_ranges(unique_windows)
    # print(f"final windows {final_windows}")
    spike_values_in_window = [spike_count[i] for i in unique_windows]
    # print(f"all spike values: {spike_values_in_window}")
    return final_windows, spike_values_in_window

if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]
    excel = pd.ExcelFile("/home/connorlab/Documents/GitHub/Julie/window_finder_results/Cortana_Windows.xlsx")
    all_rounds = excel.parse('AllRounds')
    results = []
    time_chunk_size = 0.05  # in sec
    time = np.arange(0.05, 4.00, time_chunk_size)
    rounded_time = np.round(time, 2)
    # date = "2023-09-26"
    # round_no = 1
    for index, row in all_rounds.iterrows():
        date = str(row['Date'])
        round_no = row['Round']
        date_only = row['Date'].strftime('%Y-%m-%d')
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

        time_chunk_size = 0.05  # in sec

        spike_counts = get_spike_counts_for_time_chunks(zombies, raw_unsorted_data, valid_channels, time_chunk_size)
        fractional_and_rate_of_change = compute_fractional_and_rate_of_change(raw_unsorted_data, zombies, valid_channels, time_chunk_size)
        total_sum = fractional_and_rate_of_change['total_sum']
        # print(total_sum)
        for ind, spike_total in total_sum.items():
            windows1, spike_values = expand_window_from_dynamic_threshold(spike_total, threshold=0.5)
            time_windows = extract_values_from_ranges(windows1, rounded_time)
            if len(windows1) > 0:
                print(f"{date_only} Round No. {round_no} ")
                # print(f"---------------For {ind}: {windows1}")
                # print(f"---------------For {ind}: {spike_values}")
                print(f"---------------For {ind}: {time_windows}")

                results.append({
                    'Date': date_only,
                    'Round No': round_no,
                    'Cell': str(ind),
                    'Time Windows': time_windows
                })

    results_df = pd.DataFrame(results)
    results_sorted = results_df.sort_values(by = ['Date', 'Round No', 'Cell'])
    results_sorted.to_excel('fit_window_before_explode.xlsx')

    print(results_sorted.head())
    results_expanded = results_sorted.explode('Time Windows')
    results_expanded.to_excel('fit_window_after_explode.xlsx')
