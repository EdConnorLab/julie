import math
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from channel_enum_resolvers import is_channel_in_dict, get_value_from_dict_with_channel
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis
from monkey_names import Zombies
from recording_metadata_reader import RecordingMetadataReader
from single_channel_analysis import read_pickle, calculate_spike_rate, get_spike_count
from spike_count import generate_time_windows_for_given_window_size, add_metadata_to_spike_counts




metadata = get_metadata_for_preliminary_analysis()
neural_population = metadata[metadata['Location'] == 'Amygdala']

reader = RecordingMetadataReader()

for index, row in neural_population.iterrows():
    list_time_windows = generate_time_windows_for_given_window_size(500)
    date = row['Date'].strftime('%Y-%m-%d')
    round_no = row['Round No.']
    pickle_filepath, valid_channels, _ = reader.get_metadata_for_spike_analysis(date, round_no)
    raw_trial_data = read_pickle(pickle_filepath)
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    all_results = []
    for time_window in list_time_windows:
        spike_rate_per_channel = pd.DataFrame()
        for monkey in zombies:
            monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
            monkey_spike_rates = {}
            for channel in valid_channels:
                spike_rates = []
                for index, row in monkey_data.iterrows():
                    if is_channel_in_dict(channel, row['SpikeTimes']):
                        data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                        start_time, _ = row['EpochStartStop']
                        window_start_micro, window_end_micro = time_window
                        window_start_sec = window_start_micro * 0.001
                        window_end_sec = window_end_micro * 0.001
                        spike_rates.append(calculate_spike_rate(data, (start_time + window_start_sec,
                                                                       start_time + window_end_sec)))
                    else:
                        print(f"No data for {channel} in row {index}")
                monkey_spike_rates[channel] = spike_rates
            spike_rate_per_channel[monkey] = pd.Series(monkey_spike_rates)
        spike_rate_per_channel_df = add_metadata_to_spike_counts(spike_rate_per_channel, date, round_no, time_window)
        results, sig_results = perform_anova_on_dataframe_rows_for_time_windowed(spike_rate_per_channel_df)
        if not sig_results.empty:
            # print(f" SIGNIFICANT RESULTS for {time_window}")
            # print(sig_results)
            # print(" ")
            all_results.append(sig_results)

    if all_results:
        all_results_df = pd.concat(all_results)
        print(f"significant results for {date}, {round_no}")
        print(all_results_df)


