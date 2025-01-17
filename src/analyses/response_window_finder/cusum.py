import numpy as np
import matplotlib.pyplot as plt
from intan.channels import Channel
from monkey_names import Zombies
from response_window_finder import compute_fractional_and_rate_of_change
from spike_count import get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files


def cusum(data, k, h):
    """
    Perform CUSUM change detection.

    Parameters:
        data (np.array): Input data.
        target_mean (float): Target or reference mean.
        k (float): Reference value or allowance.
        h (float): Threshold for detecting change.

    Returns:
        cusum_pos (np.array): Positive CUSUM statistic.
        cusum_neg (np.array): Negative CUSUM statistic.
        change_points (list): Indices where change is detected.
    """
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    change_points = []

    for t in range(1, len(data)):
        cusum_pos[t] = max(0, cusum_pos[t - 1] + data[t] - k)
        cusum_neg[t] = max(0, cusum_neg[t - 1] - data[t] - k)

        if cusum_pos[t] > h or cusum_neg[t] > h:
            change_points.append(t)

    return cusum_pos, cusum_neg, change_points

def sum_lists(row):
    # Using zip to pair up corresponding elements and sum them
    return [sum(elements) for elements in zip(*row)]

zombies = [member.value for name, member in Zombies.__members__.items()]
del zombies[6]
del zombies[-1]

# date = "2023-09-26"
# round_no = 1
date = "2023-10-03"
round_no = 3
raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

time_chunk_size = 0.05  # in sec

unsorted_df = compute_fractional_and_rate_of_change(raw_unsorted_data, zombies, valid_channels, time_chunk_size)
spike_counts = get_spike_counts_for_time_chunks(zombies, raw_unsorted_data, valid_channels, time_chunk_size)
spike_counts['total_sum'] = spike_counts.apply(lambda row: sum_lists(row), axis=1)

for index, row in spike_counts.iterrows():
    if str(index) == "Channel.C_019":
        data = row['total_sum']
        std_dev = np.std(data)
        mean = np.mean(data) # baseline mean
        normalized_data = (data - mean)/std_dev
        print(normalized_data)
# Parameters
k = 0.8  # sensitivity parameter
# h = 3  # threshold
#
cusum_pos, cusum_neg, change_points = cusum(normalized_data, k, k)
# Extract y-values from data at these indices
y_values_at_change_points = [data[i] for i in change_points]
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data, label='Data')
plt.plot(cusum_pos, label='CUSUM+', linestyle='--')
plt.plot(cusum_neg, label='CUSUM-', linestyle='--')

plt.scatter(change_points, y_values_at_change_points, color='red', zorder=5)
plt.axhline(y=k, color='green', linestyle='--', label='Threshold')
plt.title('CUSUM Test for Change Detection')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()