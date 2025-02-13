import numpy as np
import pandas as pd

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from baseline_spike_rate import get_average_spontaneous_firing_rate, get_inter_trial_intervals, \
    get_all_spikes_within_interval_by_channel, compute_spontaneous_firing_rate_by_channel, \
    compute_standard_deviation_of_spontaneous_spike_count_for_time_chunk
from cusum import compute_total_sum_of_spikes, min_max_scale, cusum, extract_consecutive_ranges, \
    find_corresponding_values_for_index_ranges, z_score
from inter_rater_reliability import total_entries
from monkey_names import Zombies
from recording_metadata_reader import RecordingMetadataReader
from spike_count import get_spike_count_for_single_neuron_with_time_window
from spike_rate_computation import get_raw_data_and_channels_from_files, get_raw_spike_tstamp_data


def cusum_operation(h, k, normalization_methods = None):

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
        date_only = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        raw_unsorted_data, valid_channels, _ = get_raw_data_and_channels_from_files(date, round_no)
        spike_counts = compute_total_sum_of_spikes(raw_unsorted_data, zombies, valid_channels, time_chunk_size)
        trial_intervals = raw_unsorted_data['EpochStartStop']
        raw_spike_timestamp_data, sample_rate = get_raw_spike_tstamp_data(date, round_no)
        inter_trial_intervals, total_duration = get_inter_trial_intervals(trial_intervals.tolist())
        spontaneous_spikes_by_channel = get_all_spikes_within_interval_by_channel(raw_spike_timestamp_data,
                                                                                  inter_trial_intervals)
        spon_firing_rate_by_channel = compute_spontaneous_firing_rate_by_channel(
            spontaneous_spikes_by_channel, total_duration,
            valid_channels)
        spon_std_dev = compute_standard_deviation_of_spontaneous_spike_count_for_time_chunk(inter_trial_intervals, valid_channels,
                                                                             raw_spike_timestamp_data, time_chunk_size)
        spike_counts['response_windows'] = None
        for index, r in spike_counts.iterrows():
            total_spike_count_data = r['total_sum']
            if normalization_methods is None:
                mean = total_spike_count_data.mean()
                cusum_pos, cusum_neg, change_points = cusum(total_spike_count_data, mean, k, h)
                windows = extract_consecutive_ranges(change_points)
                time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            elif normalization_methods == 'spon mean':
                spon_mean = spon_firing_rate_by_channel.get(index)
                spon_mean_spike_count = spon_mean * time_chunk_size
                cusum_pos, cusum_neg, change_points = cusum(total_spike_count_data, spon_mean_spike_count, k, h)
                windows = extract_consecutive_ranges(change_points)
                time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            elif normalization_methods == 'z-score':
                normalized_data = z_score(total_spike_count_data)
                cusum_pos, cusum_neg, change_points = cusum(normalized_data, 0, k, h)
                windows = extract_consecutive_ranges(change_points)
                time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            elif normalization_methods == 'min-max scaling':
                normalized_data = min_max_scale(total_spike_count_data)
                mean = np.mean(normalized_data)
                std = np.std(normalized_data)
                cusum_pos, cusum_neg, change_points = cusum(normalized_data, mean, k, h)
                windows = extract_consecutive_ranges(change_points)
                time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)

            print(f"---------------- {date_only} round no. {round_no} {index}----------------")
            print(time_windows)
            results.append({
                'Date': date_only,
                'Round No.': round_no,
                'Cell': str(index),
                'Time Window': time_windows
            })
            # Plotting
            # plot_spike_count_with_response_windows(rounded_time, normalized_data, total_spike_count_data, cusum_pos, cusum_neg, change_points, h)

    results_df = pd.DataFrame(results)
    results_sorted = results_df.sort_values(by=['Date', 'Round No.', 'Cell'])
    # results_sorted.to_excel('cusum_window_before_explode.xlsx')

    # print(results_sorted.head())
    results_expanded = results_sorted.explode('Time Window')
    # results_expanded.to_excel('cusum_window_after_explode.xlsx')
    # print('cusum results saved!')
    print(results_expanded.shape)
    '''
        # Date Created: 2025-01-29
        # ANOVA for windows found from cusum algorithm
    '''
    # results_expanded['Time Window'] = results_expanded['Time Window'].apply(
    #     lambda s: tuple(int(float(num) * 1000) for num in s.strip('()').split(',')))

    print(
        "--------------------------------------------- cusum windows ----------------------------------------------------------")
    print(results_expanded)

    cusum_spike_count = get_spike_count_for_single_neuron_with_time_window(results_expanded)
    print(cusum_spike_count)

    zombies_columns = [col for col in zombies if col in cusum_spike_count.columns]
    additional_columns = ['Date', 'Round No.', 'Time Window']
    zombies_cusum_spike_count = cusum_spike_count[zombies_columns + additional_columns]
    cusum_anova_results, cusum_sig_results = perform_anova_on_dataframe_rows_for_time_windowed(
        zombies_cusum_spike_count)

    print('------------------------------------ cusum window results -----------------------------------')
    # print(cusum_anova_results)
    print(cusum_sig_results)
    print(cusum_sig_results.shape)
    # cusum_anova_results.to_excel('cusum_anova_results.xlsx')
    # cusum_sig_results.to_excel('cusum_sig_results.xlsx')
    ratio = cusum_sig_results.shape[0]/results_expanded.shape[0]
    return results_expanded.shape[0], cusum_sig_results.shape[0], ratio

if __name__ == '__main__':
    parameters = []
    for h in np.linspace(0.2, 1, 2):
        for k in np.linspace(0.2, 1, 2):
            a, b, c = cusum_operation(h, k)
            parameters.append({
                'all_windows': a,
                'sig_windows': b,
                'ratio': c,
                'h value': h,
                'k value': k
            })
    parameters_df = pd.DataFrame(parameters)
    parameters_df.to_excel('cumulative_results.xlsx')
    print(parameters)