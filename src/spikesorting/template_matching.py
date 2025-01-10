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

data_handler = InputDataManager("/home/connorlab/Documents/IntanData/Cortana/2023-10-04/231004_round2")
data_handler.read_data()
channel = Channel.C_020
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

# plt.hist(min_values, bins=400, alpha=0.7, color='blue')
# plt.xlim(-100,0)
# plt.show()


# set -40 as threshold
threshold = -40
milliseconds_before = 0.8
milliseconds_after = 1.4
data_points_before = int(milliseconds_before / 1000 * sampling_rate) # sampling rate is 20000 Hz
data_points_after = int(milliseconds_after / 1000 * sampling_rate)

supra_threshold_indices = np.where(voltages < threshold)[0]
# first_index = supra_threshold_indices[0]
# start_index = int(max(first_index - data_points_before, 0))
# end_index = int(min(first_index + data_points_after, len(voltages)))
# first_spike = voltages[start_index:end_index]
# x_axis = (np.arange(-data_points_before, data_points_after, 1) / sampling_rate) * 1000
# plt.plot(x_axis, first_spike)
# plt.show()

# Initialize a list to collect non-overlapping segments
collected_segments = []
last_end_index = -1  # Tracks the end of the last added segment

# Loop through each threshold crossing index
for index in supra_threshold_indices:
    start_index = int(max(index - data_points_before, 0))
    end_index = int(min(index + data_points_after, len(voltages)))

    # Only add this segment if it does not overlap with the last segment added
    if start_index > last_end_index:
        # Extract the relevant slice of the voltage data
        sliced_data = voltages[start_index:end_index]

        # Only add segments that have the full desired length
        if len(sliced_data) == (data_points_before + data_points_after):
            collected_segments.append(sliced_data)
            last_end_index = end_index  # Update the end index of the last added segment

# Average the collected segments if we have enough
if len(collected_segments) >= 50:
    average_signal = np.mean(collected_segments[:50], axis=0)
    x_axis = (np.arange(-data_points_before, data_points_after, 1) / sampling_rate) * 1000

    # Plot the average signal
    plt.plot(x_axis, average_signal)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Average Voltage (mV)')
    plt.title('Average Voltage Profile of First 50 Non-Overlapping Segments')
    plt.axvline(x=0, color='red', linestyle='--')  # Add a vertical line at the threshold crossing
    plt.show()
else:
    print("Not enough non-overlapping segments were collected to average.")





plt.figure(figsize=(10, 6))
# Initialize the list to keep track of segments plotted to avoid overlap
plotted_segments = []

# Loop through each threshold crossing index
for index in supra_threshold_indices[:100]:
    if not any((seg[0] <= index <= seg[1]) for seg in plotted_segments):
        # Calculate the start and end indices for slicing
        start_index = int(max(index - data_points_before, 0))
        end_index = int(min(index + data_points_after, len(voltages)))

        # Check if this segment overlaps with previously plotted segments
        if not any((start_index <= seg[1] and end_index >= seg[0]) for seg in plotted_segments):
            # Extract the relevant slice of the voltage data
            sliced_data = voltages[start_index:end_index]

            # Create an x-axis centered at 0, convert index to milliseconds
            x_axis = (np.arange(-data_points_before, data_points_after, 1) / sampling_rate) * 1000

            # Plot this slice
            plt.plot(x_axis, sliced_data, alpha=0.8)  # Set a low alpha to see overlapping patterns

            # Add the current segment to the list of plotted segments
            plotted_segments.append((start_index, end_index))

# Label the axes
plt.xlabel('Time (ms)')
plt.ylabel('Voltage')
plt.title('Non-overlapping Voltage Threshold Crossings Aligned at 0 ms')
plt.axvline(x=0, color='red', linestyle='--', alpha = 0.3)  # Add a vertical line at the crossing point
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



''' 
Example code for using PCA for spike waveform feature extraction

from sklearn.decomposition import PCA


def pca_spike_features(spikes, n_components=3):
    """
    Perform PCA on an array of spike waveforms to reduce dimensionality and extract features.

    Parameters:
        spikes (np.array): 2D array where each row is a spike waveform.
        n_components (int): Number of principal components to retain.

    Returns:
        np.array: Transformed spike data into principal components.
    """
    pca = PCA(n_components=n_components)
    transformed_spikes = pca.fit_transform(spikes)
    return transformed_spikes, pca.components_


# Example usage
transformed_spikes, principal_components = pca_spike_features(segments, n_components=3)

# Plotting the principal components
plt.figure(figsize=(8, 6))
for i, component in enumerate(principal_components):
    plt.plot(component, label=f'PC{i + 1}')
plt.legend()
plt.title('Principal Components of Spike Waveforms')
plt.xlabel('Time Points')
plt.ylabel('PCA Weights')
plt.show()

'''