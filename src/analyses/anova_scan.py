import pandas as pd

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from channel_enum_resolvers import drop_duplicate_channels_with_matching_time_window
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis
from monkey_names import Zombies
from spike_count import add_metadata_to_spike_counts, \
    get_spike_counts_for_given_time_window
from spike_rate_computation import get_raw_data_and_channels_from_files


def generate_time_windows_for_given_window_size(window_size):
    """
    Generate a list of time windows (tuples) representing ranges with a specified window size.

    Parameters:
    window_size (int): Must be a positive integer and should not exceed 2000 ms.

    Returns:
    time_windows (list of tuples): Each tuple represents a range.

    Raises:
    ValueError:
        If the window_size is not a positive integer or exceeds 2000.
    """
    if not isinstance(window_size, int) or window_size <= 0 or window_size > 2000:
        raise ValueError("Window size must be a positive integer and not exceed 2000 ms.")

    return [(i, i + window_size) for i in range(0, 2000, window_size)]


if __name__ == '__main__':

    # Set time window size to be scanned
    time_windows = generate_time_windows_for_given_window_size(300)

    # Get cells to be scanned
    metadata = get_metadata_for_preliminary_analysis()
    neural_population = metadata[metadata['Location'] == 'ER']

    # Get stimuli to be scanned
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]

    # Perform ANOVA scan
    anova_sig_results = []
    for index, row in neural_population.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

        for time_window in time_windows:
            spike_counts_for_unsorted = get_spike_counts_for_given_time_window(zombies, raw_unsorted_data, valid_channels, time_window)
            spike_counts_for_unsorted = add_metadata_to_spike_counts(spike_counts_for_unsorted, date, round_no, time_window)
            unsorted_data_results, unsorted_data_sig_results = perform_anova_on_dataframe_rows_for_time_windowed(spike_counts_for_unsorted)
            if not unsorted_data_sig_results.empty:
                anova_sig_results.append(unsorted_data_sig_results)

            if sorted_data is not None:
                unique_channels = set()
                unique_channels.update(sorted_data['SpikeTimes'][0].keys())
                spike_counts_for_sorted = get_spike_counts_for_given_time_window(zombies, sorted_data, unique_channels, time_window)
                spike_counts_for_sorted = add_metadata_to_spike_counts(spike_counts_for_sorted, date, round_no, time_window)
                sorted_data_results, sorted_data_sig_results = perform_anova_on_dataframe_rows_for_time_windowed(spike_counts_for_sorted)
                if not sorted_data_sig_results.empty:
                    anova_sig_results.append(sorted_data_sig_results)

    all_anova_sig_results = pd.concat(anova_sig_results, ignore_index=True)
    results_df_final = drop_duplicate_channels_with_matching_time_window(all_anova_sig_results)
    print(results_df_final)
    results_df_final.to_excel('/home/connorlab/Documents/GitHub/Julie/anova_scan/ER_ANOVA_scan_size_300ms.xlsx')

