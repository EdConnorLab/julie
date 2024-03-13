import os
import matplotlib.pyplot as plt

import spike_rate_analysis
from monkey_names import Monkey as M
from pathlib import Path

from social_data_reader import read_raw_social_data, clean_raw_social_data, extract_specific_interaction_type, \
    generate_edgelist_from_pairwise_interactions, read_social_data_and_validate

social_data = read_social_data_and_validate()

affiliative = extract_specific_interaction_type(social_data, 'affiliative')
edgelist_affiliative = generate_edgelist_from_pairwise_interactions(affiliative)
print(f'edgelist: {edgelist_affiliative}')
avg_spike_rate = spike_rate_analysis.compute_overall_average_spike_rates_for_each_round("2023-09-29", 2)
print(f'avg_spike_rate: {avg_spike_rate}')

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
    actor_specific['Spike Rate'] = actor_specific['Social Modifier'].map(avg_spike_rate)

    # Remove 81G spike rate since it's nan
    actor_specific_cleaned = actor_specific.dropna(subset=['Spike Rate'])

    plt.plot(actor_specific_cleaned['weight'], actor_specific_cleaned['Spike Rate'], marker='o', linestyle='None')
    for index, row in actor_specific_cleaned.iterrows():
        plt.text(row['weight'], row['Spike Rate'], row['Social Modifier'], fontsize=12, ha='right', va='bottom')
    plt.xlabel('Frequency of Affiliative Behavior')
    plt.ylabel('Firing Rate')
    plt.title(f'affiliative behavior from {zombie}')
    plt.show()
#
# receiver
# for zombie in zombies:
#     receiver_specific = sorted[sorted['Social Modifier'] == zombie].copy()
#     receiver_specific['Spike Rate'] = receiver_specific['Focal Name'].map(norm_values)
#
#     # Remove 81G spike rate since it's nan
#     receiver_specific_cleaned = receiver_specific.dropna(subset=['Spike Rate'])
#
#     plt.plot(receiver_specific_cleaned['weight'], receiver_specific_cleaned['Spike Rate'], marker='o', linestyle='None')
#     for index, row in receiver_specific_cleaned.iterrows():
#        plt.text(row['weight'], row['Spike Rate'], row['Focal Name'], fontsize=12, ha='right', va='bottom')
#     plt.xlabel('Frequency of Affiliative Behavior')
#     plt.ylabel('Firing Rate')
#     plt.title(f'affiliative behavior to {zombie}')
#     plt.show()
