import numpy as np

from monkey_names import Zombies
from response_window_finder import compute_fractional_and_rate_of_change
from spike_count import get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files

if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    date = "2023-09-26"
    round_no = 1
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

    time_chunk_size = 0.05  # in sec

    spike_counts = get_spike_counts_for_time_chunks(zombies, raw_unsorted_data, valid_channels, time_chunk_size)
    # print(spike_counts)

    for monkey in zombies:
        monkey_specific_spike_counts = spike_counts[monkey]
        print(monkey)
        for index, spike_count in monkey_specific_spike_counts.items():
            max_value = max(spike_count)
            max_index = spike_count.index(max_value)

            print(index, spike_count)
            print(max_index, spike_count[max_index])

