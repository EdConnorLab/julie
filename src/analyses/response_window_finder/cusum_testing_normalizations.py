import numpy as np
import pandas as pd

from baseline_spike_rate import get_inter_trial_intervals, get_all_spikes_within_interval_by_channel, \
    compute_spontaneous_firing_rate_by_channel, compute_standard_deviation_of_spontaneous_spike_count_for_time_chunk
from channel_enum_resolvers import convert_to_enum
from cusum import compute_total_sum_of_spikes, cusum, extract_consecutive_ranges, \
    find_corresponding_values_for_index_ranges, z_score, min_max_scale, plot_spike_count_with_response_windows
from monkey_names import Zombies
from spike_rate_computation import get_raw_data_and_channels_from_files, get_raw_spike_tstamp_data

if __name__ == "__main__":
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    time_chunk_size = 0.05  # in sec
    rounded_time = np.round(np.arange(time_chunk_size, 3.50, time_chunk_size), 2)


    response_window_test = pd.read_excel('response_window_algorithm_validation_test.xlsx')
    for _, row in response_window_test.iterrows():
        date = str(row['Date'])
        round_no = row['Round No.']
        date_only = row['Date'].strftime('%Y-%m-%d')
        channels = row['Cell']
        channels_to_read = convert_to_enum(channels)
        raw_unsorted_data, valid_channels, _ = get_raw_data_and_channels_from_files(date_only, round_no)
        spike_counts = compute_total_sum_of_spikes(raw_unsorted_data, zombies, [channels_to_read], time_chunk_size)
        trial_intervals = raw_unsorted_data['EpochStartStop']
        raw_spike_timestamp_data, sample_rate = get_raw_spike_tstamp_data(date_only, round_no)
        inter_trial_intervals, total_duration = get_inter_trial_intervals(trial_intervals.tolist())
        spontaneous_spikes_by_channel = get_all_spikes_within_interval_by_channel(raw_spike_timestamp_data,
                                                                                  inter_trial_intervals)
        spon_firing_rate_by_channel = compute_spontaneous_firing_rate_by_channel(
            spontaneous_spikes_by_channel, total_duration,valid_channels)
        spon_std_dev = compute_standard_deviation_of_spontaneous_spike_count_for_time_chunk(inter_trial_intervals,valid_channels,
                                                                             raw_spike_timestamp_data, time_chunk_size)

        spike_counts['response_windows'] = None
        for index, r in spike_counts.iterrows():
            total_spike_count_data = r['total_sum']
            # 1. no normalization - subtracting mean
            raw_mean = np.mean(total_spike_count_data)
            raw_std = np.std(total_spike_count_data)
            k_1 = raw_std*0.5
            h_1 = 1
            cusum_pos, cusum_neg, change_points = cusum(total_spike_count_data, raw_mean, k_1, h_1)
            windows = extract_consecutive_ranges(change_points)
            time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            print("# 1. no normalization - subtracting mean")
            print(f"k {k_1:.2f}, h {h_1:.2f}")
            if len(time_windows) > 0:
                print(f"---------------- {date_only} round no. {round_no} {index}----------------")
                print(time_windows)
            plot_spike_count_with_response_windows(rounded_time, None, total_spike_count_data, cusum_pos, cusum_neg,
                                                  change_points, h_1)
            # 2. no normalization - subtracting spon mean
            spon_mean = spon_firing_rate_by_channel.get(index)
            spon_std_dev = spon_std_dev.get(index)
            k_2 = spon_std_dev*0.5
            h_2 = 1
            spon_mean_spike_count = spon_mean * time_chunk_size
            cusum_pos, cusum_neg, change_points = cusum(total_spike_count_data, spon_mean_spike_count, k_2, h_2)
            windows = extract_consecutive_ranges(change_points)
            time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            print("# 2. no normalization - subtracting spon mean")
            print(f"k {k_2:.2f}, h {h_2:.2f}")
            if len(time_windows) > 0:
                print(f"---------------- {date_only} round no. {round_no} {index}----------------")
                print(time_windows)
            plot_spike_count_with_response_windows(rounded_time, None, total_spike_count_data, cusum_pos, cusum_neg,
                                                  change_points, h_2)

            # 3. normalization - z-score
            normalized_data = z_score(total_spike_count_data)
            h_3 = 1
            k_3 = 0.5
            cusum_pos, cusum_neg, change_points = cusum(normalized_data, 0, k_3, h_3)
            windows = extract_consecutive_ranges(change_points)
            time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            print("# 3. normalization - z-score")
            print(f"k {k_3:.2f}, h {h_3:.2f}")
            if len(time_windows) > 0:
                print(f"---------------- {date_only} round no. {round_no} {index}----------------")
                print(time_windows)
            plot_spike_count_with_response_windows(rounded_time, normalized_data, total_spike_count_data, cusum_pos, cusum_neg,
                                                  change_points, h_2)
            # 4. normalization - min-max scaling
            normalized_data = min_max_scale(total_spike_count_data)
            mean = np.mean(normalized_data)
            std = np.std(normalized_data)
            k_4 = 0.5*std
            h_4 = 1
            cusum_pos, cusum_neg, change_points = cusum(normalized_data, mean, k_4, h_4)
            windows = extract_consecutive_ranges(change_points)
            time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            print("# 4. normalization - min-max scaling")
            print(f"k {k_4:.2f}, h {h_4:.2f}")
            if len(time_windows) > 0:
                print(f"---------------- {date_only} round no. {round_no} {index}----------------")
                print(time_windows)
            plot_spike_count_with_response_windows(rounded_time, normalized_data, total_spike_count_data, cusum_pos, cusum_neg,
                                                  change_points, h_4)
