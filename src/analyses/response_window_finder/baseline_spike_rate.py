import os

from intan.spike_file import fetch_spike_tstamps_from_file

from channel_enum_resolvers import get_value_from_dict_with_channel
from spike_rate_computation import get_raw_data_and_channels_from_files, get_raw_spike_tstamp_data

date = "2023-09-26"
round_no = 1
raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
trial_epochs = raw_unsorted_data['EpochStartStop']
trial_epoch_list = trial_epochs.tolist()

non_trial_epochs = []
non_trial_duration = 0
for i in range(0, len(trial_epoch_list)):
    if i == 0:
        # start, _ = trial_epoch_list[i]
        # pre_experiment = (0, start)
        # total_duration += start
        # non_trial_epochs.append(pre_experiment)
        continue
    else:
        _, stop = trial_epoch_list[i - 1]
        start, _ = trial_epoch_list[i]
        inbetween_trials = (stop, start)
        non_trial_duration += (start - stop)
        non_trial_epochs.append(inbetween_trials)

spike_data, sample_rate = get_raw_spike_tstamp_data(date, round_no)
# print(spike_data)
for non_trial_epoch in non_trial_epochs:
    start, stop = non_trial_epoch
    baseline_spikes_by_channel = {}
    for channel, tstamps in spike_data.items():
        spike_times = []
        for tstamp in tstamps:
            if start < tstamp < stop:
                spike_times.append(tstamp)
        baseline_spikes_by_channel[channel] = spike_times
# print(baseline_spikes_by_channel)

for channel in valid_channels:
    spikes = get_value_from_dict_with_channel(channel, baseline_spikes_by_channel)
    if spikes is not None:
        spike_count = len(spikes)
        spike_rate = spike_count / non_trial_duration
        print(f"baseline spike rate (sp/s) for {channel} is {spike_rate} with total spikes {spike_count}")
        # print(f"for 50 ms... {spike_rate * 0.05}")

#### Just checking trial spike rate
total_trial_duration = 0
for trial_epoch in trial_epoch_list:
    trial_start, trial_stop = trial_epoch
    total_trial_duration += trial_stop - trial_start
    trial_spikes_by_channel = {}
    for channel, trial_spike_tstamps in spike_data.items():
        trial_spike_times = []
        for trial_spike_tstamp in trial_spike_tstamps:
            if trial_start < trial_spike_tstamp < trial_stop:
                trial_spike_times.append(trial_spike_tstamp)
        trial_spikes_by_channel[channel] = trial_spike_times


for channel in valid_channels:
    final_trial_spikes = get_value_from_dict_with_channel(channel, trial_spikes_by_channel)
    if final_trial_spikes is not None:
        trial_spike_count = len(final_trial_spikes)
        trial_spike_rate = trial_spike_count / total_trial_duration
        print(f"spike rate (sp/s) for {channel} is {trial_spike_rate} with total spikes {trial_spike_count}")

