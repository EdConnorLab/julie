from windowsort.datahandler import InputDataManager, SortedSpikeExporter, SortingConfigManager
from windowsort.threshold import threshold_spikes_absolute
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
import numpy as np
from clat.intan.channels import Channel

def min_in_chunks(data, chunk_size_samples):
    """
    Calculate the maximum value in each chunk of the data.

    Parameters:
        data (np.array): The array containing voltage data.
        chunk_size_samples (int): The number of samples in each chunk.

    Returns:
        np.array: An array of maximum values for each chunk.
    """
    # Calculate the number of chunks
    num_chunks = int(len(data) // chunk_size_samples)
    # Reshape the data to create an array where each row is a chunk
    reshaped_data = data[:num_chunks * chunk_size_samples].reshape(num_chunks, chunk_size_samples)
    # Calculate the maximum for each chunk
    min_values = reshaped_data.min(axis=1)
    return min_values

def max_in_chunks(data, chunk_size_samples):
    """
    Calculate the maximum value in each chunk of the data.

    Parameters:
        data (np.array): The array containing voltage data.
        chunk_size_samples (int): The number of samples in each chunk.

    Returns:
        np.array: An array of maximum values for each chunk.
    """
    # Calculate the number of chunks
    num_chunks = int(len(data) // chunk_size_samples)
    # Reshape the data to create an array where each row is a chunk
    reshaped_data = data[:num_chunks * chunk_size_samples].reshape(num_chunks, chunk_size_samples)
    # Calculate the maximum for each chunk
    max_values = reshaped_data.max(axis=1)
    return max_values

data_handler = InputDataManager("/home/connorlab/Documents/IntanData/Cortana/2023-10-24/231024_round2")
data_handler.read_data()
channel = Channel.C_002
voltages = data_handler.voltages_by_channel.get(channel)

# Setting threshold to detect spikes
# threshold_voltage = -100
# crossing_indices = threshold_spikes_absolute(threshold_voltage, voltages)
# print(threshold_spikes_absolute(-100, voltages))

# sampling rate = 20,000 Hz
sampling_rate = 20000
time_chunk = 0.005 # 5ms
chunk_size_samples = int(sampling_rate * time_chunk)
min_values = min_in_chunks(voltages, chunk_size_samples)
print(min_values)
print(len(min_values))
print(f"minimum:{np.min(min_values)}")
print(f"mean: {np.mean(min_values)}")
print(f"median: {np.median(min_values)}")
print(f"Q3: {np.percentile(min_values, 75)}")
print(f"Q1: {np.percentile(min_values, 25)}")

plt.hist(min_values, bins=400, alpha=0.7, color='blue', edgecolor='black')
plt.xlim(-100,0)
plt.show()

'''
crossing_indices = 0
if len(crossing_indices) == 0:
    pass
else:
    # sampling rate of 20,000 Hz
    window = 20
    start = max(0, crossing_indices[0] - window)  # Make sure start is not less than 0
    end = min(len(voltages), crossing_indices[0] + window)  # Make sure end does not exceed array length

    # Extract the part of the array to plot
    spike_template = voltages[start:end]
    flipped_spike_template = spike_template[::-1] # need to flip because scipy's correlate method flips the template automatically

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(spike_template, linestyle='-')

    plt.title(f'{channel} first spike detected')
    plt.xlabel('Position')
    plt.show()

    corr = correlate(voltages, flipped_spike_template, mode='full')
    max_corr = np.max(corr)
    threshold_voltage = 0.8 * max_corr
    peaks, _ = find_peaks(corr, height=threshold_voltage)
    match_indices = peaks - len(spike_template) + 1
    plt.figure(figsize=(14, 7))
    plt.subplot(211)
    plt.plot(voltages, label='Signal')
    for index in match_indices:
        plt.plot(np.arange(index, index + len(spike_template)), spike_template,
                 label=f'Match starting at index {index}',
                 linewidth=1.5)
    plt.title('Signal and Matches')

    plt.subplot(212)
    plt.plot(corr, label='Correlation')
    plt.plot(peaks, corr[peaks], "x")
    plt.axhline(y=threshold_voltage, color='r', linestyle='--', label=f'Threshold: {threshold_voltage}')
    plt.title('Correlation and Peaks')
    plt.show()

# for channel, voltages in data_handler.voltages_by_channel.items():
'''