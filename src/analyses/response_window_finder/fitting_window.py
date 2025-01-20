import numpy as np

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
            threshold = 0.5
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


def expand_window_from_dynamic_threshold(spike_count, threshold = 0.5):
    max_value = max(spike_count)
    window_indices = []
    all_windows = []
    if max_value != 0:
        indices_of_max = [index for index, value in enumerate(spike_count) if value == max_value]
        #    print(f"max values are in {indices_of_max}")
        for i in indices_of_max:
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
                # print(spike_count)
                # print(f"window detected-- indices at {window_indices}")
                spike_values_in_window = [spike_count[i] for i in window_indices]
                # print(f"window detected-- {spike_values_in_window}")
                # time = np.array(window_indices) * 0.05 + 0.05
                # window_time_in_sec = (format(time[0], '.2f'), format(time[-1], '.2f'))
                # print(window_time_in_sec)
                all_windows.append(window_indices)
    else:
        print(f"no spikes detected")
    # remove duplicates
    unique_tuples = set(tuple(x) for x in all_windows)
    unique_windows = [list(x) for x in unique_tuples]
    return unique_windows

if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    date = "2023-09-26"
    round_no = 1
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

    time_chunk_size = 0.05  # in sec

    spike_counts = get_spike_counts_for_time_chunks(zombies, raw_unsorted_data, valid_channels, time_chunk_size)
    fractional_and_rate_of_change = compute_fractional_and_rate_of_change(raw_unsorted_data, zombies, valid_channels, time_chunk_size)
    total_sum = fractional_and_rate_of_change['total_sum']
    print(total_sum)
    for index, spike_total in total_sum.items():
        windows1 = expand_window_from_dynamic_threshold(spike_total, threshold=0.5)
        if len(windows1) > 0:
            print(f"-----------------------------For {index}: {windows1}")

'''
    for monkey in zombies:
        monkey_specific_spike_counts = spike_counts[monkey]
        for index, spike_count in monkey_specific_spike_counts.items():
            print(f"{monkey} {index}")
            windows1 = expand_window_from_max_threshold(spike_count, threshold = 0.5)
            if len(windows1) > 0:
                print(f"-----------------------------For {monkey} {index}: {windows1}")
            # windows2 = expand_window_from_dynamic_threshold(spike_count, threshold = 0.5)
            # print(f"For {monkey} {index}: {windows2}")
'''