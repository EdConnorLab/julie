import os
import random
import matplotlib.pyplot as plt
from monkeyids import Monkey as M
from pathlib import Path

from social_data_reader import read_raw_social_data, clean_raw_social_data, extract_pairwise_interactions, \
    generate_edgelist_from_pairwise_interactions, combine_edgelists
from spike_rate_analysis import read_raw_trial_data, compute_spike_rates_per_channel_per_monkey


current_dir = os.getcwd()
raw_data_file_name = 'ZombiesFinalRawData.xlsx'
file_path = Path(current_dir).parent.parent.parent / 'resources' / raw_data_file_name
raw_social_data = read_raw_social_data(file_path)
social_data = clean_raw_social_data(raw_social_data)

affiliative = extract_pairwise_interactions(social_data, 'affiliative')
edgelist_affiliative = generate_edgelist_from_pairwise_interactions(affiliative)
print(f'edgelist: {edgelist_affiliative}')
date = "2023-10-30"
round = "1698699440778381_231030_165721"
cortana_path = "/home/connorlab/Documents/IntanData"
round_path = os.path.join(cortana_path, date, round)
raw_trial_data = read_raw_trial_data(round_path)
avg_spike_rate = compute_spike_rates_per_channel_per_monkey(raw_trial_data)
print(f'avg_spike_rate: {avg_spike_rate}')
random_row = avg_spike_rate.loc["Channel.C_017_Unit 1"]
norm_values = ((random_row - random_row.min()) / (random_row.max() - random_row.min())).to_dict()
print(norm_values)

# Get all zombies
zombies = [member.value for name, member in M.__members__.items() if name.startswith('Z_')]
filtered = edgelist_affiliative[edgelist_affiliative['Focal Name'].isin(zombies)]
print(edgelist_affiliative)
sorted = filtered.sort_values(by='weight', ascending=False)
print(sorted)

# Single dimension analysis

# actor
for zombie in zombies:
    actor_specific = sorted[sorted['Focal Name'] == zombie].copy()
    actor_specific['Spike Rate'] = actor_specific['Social Modifier'].map(norm_values)

    # Remove 81G spike rate since it's nan
    actor_specific_cleaned = actor_specific.dropna(subset=['Spike Rate'])

    plt.plot(actor_specific_cleaned['weight'], actor_specific_cleaned['Spike Rate'], marker='o', linestyle='None')
    for index, row in actor_specific_cleaned.iterrows():
        plt.text(row['weight'], row['Spike Rate'], row['Social Modifier'], fontsize=12, ha='right', va='bottom')
    plt.xlabel('Frequency of Affiliative Behavior')
    plt.ylabel('Firing Rate')
    plt.title(f'affiliative behavior from {zombie}')
    plt.show()

# receiver
for zombie in zombies:
    receiver_specific = sorted[sorted['Social Modifier'] == zombie].copy()
    receiver_specific['Spike Rate'] = receiver_specific['Focal Name'].map(norm_values)

    # Remove 81G spike rate since it's nan
    receiver_specific_cleaned = receiver_specific.dropna(subset=['Spike Rate'])

    plt.plot(receiver_specific_cleaned['weight'], receiver_specific_cleaned['Spike Rate'], marker='o', linestyle='None')
    for index, row in receiver_specific_cleaned.iterrows():
       plt.text(row['weight'], row['Spike Rate'], row['Focal Name'], fontsize=12, ha='right', va='bottom')
    plt.xlabel('Frequency of Affiliative Behavior')
    plt.ylabel('Firing Rate')
    plt.title(f'affiliative behavior to {zombie}')
    plt.show()
