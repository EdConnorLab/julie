import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.getcwd()
# raw_data_file_name = 'RawData_Organized_on20240208.xlsx'
raw_data_file_name = 'ZombiesFinalRawData.xlsx'
file_path = '/Users/julie/General/LocalDocuments/GitHub/Julie/resources/ZombiesFinalRawData.xlsx'
# file_path = Path(current_dir).parent / 'resources' / raw_data_file_name
xl = pd.ExcelFile(file_path)
sheet_names = xl.sheet_names
# print(sheet_names)
last_sheet = sheet_names[-1]
df = xl.parse(last_sheet)
print(df.columns)

# Rename columns for convenience
df.rename(columns={'All Occurrence Value':'Behavior'},inplace=True)
df.rename(columns={'All Occurrence Behavior Social Modifier':'Social Modifier'},inplace=True)
df.rename(columns={'All Occurrence Space Use Coordinate XY':'Space Use'},inplace=True)

# Rename the last column to Time
df.columns = [*df.columns[:-1], 'Time']

# Combine Year, Month, Day columns into VideoDate
df['VideoDate'] = df.iloc[:,-4].astype(str).str.zfill(2) + df.iloc[:,-3].astype(str).str.zfill(2) + df.iloc[:,-2].astype(str).str.zfill(2)
#print(df.columns)


extracted_df = df[['Observer','Focal Name', 'Behavior', 'Social Modifier', 'Space Use', 'VideoDate', 'Time']]

print(extracted_df['VideoDate'].unique())
print(f"{extracted_df['VideoDate'].nunique()} days of recording")

''' CHECK NUMBER OF MONKEYS '''
# For dates before 06/13/2022, 10 monkeys
# after 06/13/2022, 8 monkeys
grouped = df.groupby('VideoDate')['Focal Name'].agg(['nunique', 'unique']).reset_index()

for index, row in grouped.iterrows():
    video_date = row['VideoDate']
    # unique_values = ', '.join(row['unique'])
    count = row['nunique']
    print(f"VideoDate: {video_date}, # of unique monkeys: {count}")

# For each unique date, there should be

''' CHECK NUMBER OF INTERVAL DATA '''
df['Behavior Abbrev'] = df['Behavior'].str[:4].str.replace(' ', '')

# Create a mask to check if 'Behavior Abbrev' starts with 'I'
mask = df['Behavior Abbrev'].str.startswith('I')

# Filter the DataFrame to include only rows where 'Behavior Abbrev' starts with 'I'
interval = df[mask].groupby('VideoDate')['Behavior Abbrev'].count().reset_index()
filtered = interval[(interval['Behavior Abbrev'] != 120) & (interval['Behavior Abbrev'] != 96)]
print(filtered)

# Group by 'VideoDate' and 'Focal Name' and count the occurrences
result = df[mask].groupby(['VideoDate', 'Focal Name']).size().reset_index(name='Count')
filtered_result = result[(result['Count'] != 12)]
print(filtered_result)


''' GET ALL AGONISTIC BEHAVIORS '''
agonistic = df[(df['Behavior Abbrev'] == 'AOA') | (df['Behavior Abbrev'] == 'IAG')]
agonistic = agonistic[['Focal Name', 'Social Modifier']]
print(agonistic['Social Modifier'].unique())
