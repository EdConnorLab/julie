import os
from pathlib import Path

import pandas as pd


def get_valid_channels(metadata, date, round) -> list:
    matching_round = metadata[(metadata['Date'] == date) & (metadata['Pickle File Name'].str.contains(round))]
    channels = matching_round['Channels1'].apply(lambda x: [int(i.strip()) for i in x.split(',')]).tolist()
    formatted_channels = ['Channel.C_{:03.0f}'.format(channel) for channel in channels[0]]
    return formatted_channels

def get_pickle_filename_for_specific_round(metadata, date, round_number):
    matching_round = metadata[(metadata['Date'] == date) & metadata['Round No.'] == round_number]
    return matching_round['Pickle File Name']


current_dir = os.getcwd()
raw_data_file_name = 'Cortana_Recording_Metadata.xlsx'
file_path = Path(current_dir).parent.parent.parent / 'resources' / raw_data_file_name
xl = pd.ExcelFile(file_path)
sheet_names = xl.sheet_names
# print(sheet_names)
channels = xl.parse('Channels')

locations = xl.parse('Location')
metadata = pd.merge(channels, locations, on=['Date', 'Round No.'])

print(metadata.columns)
ER_data = metadata[metadata['Location'] == 'ER']
AMG_data = metadata[metadata['Location'] == 'Amygdala']
print(ER_data)
print(AMG_data)
get_valid_channels(metadata, "10-10-2023", "1696957915096002_231010_131155")
