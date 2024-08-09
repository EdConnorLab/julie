import pandas as pd

from channel_enum_resolvers import is_channel_in_dict, get_value_from_dict_with_channel, drop_duplicate_channels
from single_channel_analysis import read_pickle, calculate_spike_rate
from single_unit_analysis import read_sorted_data
from data_readers.recording_metadata_reader import RecordingMetadataReader

"""

Structure of spike_rate_computation.py

get_spike_rates_for_each_trial:
    read_pickle 
    compute_spike_rates_from_raw_trial_data
    read_sorted_data
    compute_spike_rates_from_sorted_data

get_average_spike_rates_for_each_monkey:
    read_pickle
    compute_average_spike_rates_from_raw_trial_data
    read_sorted_data
    compute_average_spike_rates_from_sorted_data
    
compute_average_spike_rate_for_single_neuron_for_specific_time_window

compute_population_spike_rates_for_ER
compute_population_spike_rates_for_AMG
compute_overall_average_spike_rates_for_each_round


Note on raw_trial_data
raw_trial_data is a pandas dataframe with the following columns:
    - 'TaskField' (i.e. 1695750846583000)
    - 'FileName' (i.e. 4331.JPG)
    - 'MonkeyId' (i.e. 10294)
    - 'MonkeyName' (i.e. 167I)
    - 'MonkeyGroup' (i.e. Stranger Things)
    - 'SpikeTimes' (i.e. {Channel.C_002:[13.3532, 18.0429], Channel.C_003:[13.3532, 18.0429]}
    - 'EpochStartStop' (i.e. (8.3331, 10.8741))

"""


def get_spike_rates_for_each_trial(date, round_number):
    """
    Computes spike rates (sp/s) for each trial for a given experimental round
    """
    reader = RecordingMetadataReader()
    pickle_filepath, valid_channels, round_dir_path = reader.get_metadata_for_spike_analysis(date, round_number)

    raw_trial_data = read_pickle(pickle_filepath)
    raw_data_spike_rates = compute_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels)

    # Check if the experimental round is sorted
    sorted_file = round_dir_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        sorted_data = read_sorted_data(round_dir_path)
        sorted_data_spike_rates = compute_spike_rates_from_sorted_data(sorted_data)

        raw_data_spike_rates = drop_duplicate_channels(raw_data_spike_rates, sorted_data)
        spike_rates = pd.concat([sorted_data_spike_rates, raw_data_spike_rates])
    else:
        spike_rates = raw_data_spike_rates

    return spike_rates


def get_average_spike_rates_for_each_monkey(date, round_number):
    """
    Computes average spike rates for each monkey photo -- spike rates (sp/s) are averaged over 10 trials

    Parameters:
        date (str): the date of the experimental round (e.g. '2023-10-03')
        round_number (int): the experimental round number
    Returns:
        average spike rates across 10 trials for each monkey
    """
    reader = RecordingMetadataReader()
    pickle_filepath, valid_channels, round_dir_path = reader.get_metadata_for_spike_analysis(date, round_number)
    raw_trial_data = read_pickle(pickle_filepath)
    average_spike_rates_from_raw_data = compute_average_spike_rates_from_raw_trial_data(raw_trial_data,
                                                                                        valid_channels)
    # Check if the experimental round is sorted
    sorted_file = round_dir_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        # read sorted data and compute avg spike rate
        sorted_data = read_sorted_data(round_dir_path)
        average_spike_rates_from_sorted_data = compute_average_spike_rates_from_sorted_data(sorted_data)

        average_spike_rates_from_raw_data = drop_duplicate_channels(average_spike_rates_from_raw_data, sorted_data)
        average_spike_rates = pd.concat([average_spike_rates_from_sorted_data, average_spike_rates_from_raw_data])
    else:
        average_spike_rates = average_spike_rates_from_raw_data

    return average_spike_rates


def compute_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels):
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    spike_rate_per_channel = pd.DataFrame()
    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_spike_rates = {}
        for channel in valid_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")
            monkey_spike_rates[channel] = spike_rates
        spike_rate_per_channel[monkey] = pd.Series(monkey_spike_rates)
    return spike_rate_per_channel


def compute_average_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels):
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}
        for channel in valid_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            monkey_specific_spike_rate[channel] = avg_spike_rate

        # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)

    return avg_spike_rate_by_unit


def compute_spike_rates_from_sorted_data(sorted_data):
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())
    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}
        for channel in unique_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                   print(f"No data for {channel} in row {index}")
            monkey_specific_spike_rate[channel] = spike_rates
        spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)
    return spike_rate_by_unit


def compute_average_spike_rates_from_sorted_data(sorted_data):
    # average spike rates for each monkey
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())

    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}

        for channel in unique_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))

                else:
                   print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            monkey_specific_spike_rate[channel] = avg_spike_rate

        # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)

    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit


def compute_average_spike_rate_for_single_neuron_for_specific_time_window(raw_trial_data, cell, time_window):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}
        spike_rates = []
        for index, row in monkey_data.iterrows():
            if is_channel_in_dict(cell, row['SpikeTimes']):
                data = get_value_from_dict_with_channel(cell, row['SpikeTimes'])
                if time_window is not None:
                    window_start_sec, window_end_sec = [tw * 0.001 for tw in time_window]
                    start_time, _ = row['EpochStartStop']
                    spike_rates.append(
                        calculate_spike_rate(data, (start_time + window_start_sec, start_time + window_end_sec)))
                else:
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))

            else:
                print(f"No data for {cell} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else None
            monkey_specific_spike_rate[cell] = avg_spike_rate

    # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)
    if spike_rates is None:
        avg_spike_rate_by_unit = None

    return avg_spike_rate_by_unit


def compute_population_spike_rates_for_ER():
    # ER Population Average Spike Rate
    ER_population = pd.DataFrame()
    reader = RecordingMetadataReader()
    ER = reader.get_metadata_for_brain_region('ER')

    for index, row in ER.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round = row['Round No.']
        avg_spike_rates_for_specific_round = get_average_spike_rates_for_each_monkey(date, round)
        ER_population = pd.concat([ER_population, avg_spike_rates_for_specific_round])
    population_spike_rate = ER_population.mean()
    print(population_spike_rate)
    return population_spike_rate


def compute_population_spike_rates_for_AMG():
    AMG_population = pd.DataFrame()
    reader = RecordingMetadataReader()
    AMG = reader.get_metadata_for_brain_region('AMG')

    for index, row in AMG.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round = row['Round No.']
        avg_spike_rates_for_specific_round = get_average_spike_rates_for_each_monkey(date, round)
        AMG_population = pd.concat([AMG_population, avg_spike_rates_for_specific_round])
    population_spike_rate = AMG_population.mean()
    print(population_spike_rate)
    return population_spike_rate


def compute_overall_average_spike_rates_for_each_round(date, round_number):
    overall_average_spike_rates = get_average_spike_rates_for_each_monkey(date, round_number).mean()
    return overall_average_spike_rates


if __name__ == '__main__':

    df = get_spike_rates_for_each_trial("2023-10-04", 4)
    dat = compute_overall_average_spike_rates_for_each_round("2023-09-29", 2)
    # ones with errors
    # avg_spike_rates = compute_average_spike_rates("2023-09-29", 1)
    # avg_spike_rates = compute_average_spike_rates("2023-11-08", 1)
