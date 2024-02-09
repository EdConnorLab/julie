import os
import pandas as pd
from clat.intan.rhd import load_intan_rhd_format

from julie.single_channel_analysis import read_pickle, calculate_spike_rate
from julie.single_unit_analysis import calculate_spike_timestamps


def main():
    date = "2023-10-30"
    round = "1698699440778381_231030_165721"
    sorted_spikes_filename = "sorted_spikes.pkl"

    cortana_path = "/home/connorlab/Documents/IntanData"
    round_path = os.path.join(cortana_path, date, round)
    compiled_trials_filepath = os.path.join(round_path, "compiled.pk1")
    experiment_name = os.path.basename(os.path.dirname(compiled_trials_filepath))
    raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)

    # TODO: specify which sorting pickle to use and which units to plot, then add them to dataframe
    rhd_file_path = os.path.join(round_path, "info.rhd")
    sorted_spikes_filepath = os.path.join(cortana_path, date, round, sorted_spikes_filename)
    sorted_spikes = read_pickle(sorted_spikes_filepath)
    sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
    sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)

    # average spike rate for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()

    for monkey in unique_monkeys:
        print(monkey)
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]

        unique_channels = set()
        for index, row in raw_trial_data.iterrows():
            unique_channels.update(row['SpikeTimes'].keys())
        print(f'Num of Unique Channels: {len(unique_channels)}')

        for channel in unique_channels:
            print(channel)
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if channel in row['SpikeTimes']:
                    data = row['SpikeTimes'][channel]
                if channel in row['SpikeTimes']:
                    data = row['SpikeTimes'][channel]
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            print(f'the average spike rate is {avg_spike_rate}')

    # spike rate for each picture
    # for index, row in raw_trial_data.iterrows():
    #     for unit, data in row['SpikeTimes'].items():
    #         start, stop = row['EpochStartStop']
    #         duration = stop - start
    #         rate = len(data)/duration
    #         print(f'spike rate manually calculated: {rate}')
    #         spike_rate = calculate_spike_rate(data, row['EpochStartStop'])
    #         print(f'For unit {unit}, the spike rate is {spike_rate}')


if __name__ == '__main__':
    main()