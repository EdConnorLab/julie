import os
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def read_social_data_and_validate():
    # Load raw data file
    current_dir = os.getcwd()
    raw_data_file_name = 'ZombiesFinalRawData.xlsx'
    file_path = Path(current_dir).parent.parent.parent / 'resources' / raw_data_file_name
    raw_social_data = read_raw_social_data(file_path)

    # Clean raw data
    social_data = clean_raw_social_data(raw_social_data)

    # Check number of monkeys recorded for each day
    validate_number_of_monkeys(social_data)

    # Check number of interval datapoints for each day
    validate_number_of_interval_datapoints(social_data)
    return social_data


def read_raw_social_data(filepath):
    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    # print(sheet_names)
    last_sheet = sheet_names[-1]
    df = xl.parse(last_sheet)
    # print(df.columns)
    return df


def clean_raw_social_data(raw_social_data):
    # Rename columns for convenience
    raw_social_data.rename(columns={'All Occurrence Value': 'Behavior'}, inplace=True)
    raw_social_data.rename(columns={'All Occurrence Behavior Social Modifier': 'Social Modifier'}, inplace=True)
    raw_social_data.rename(columns={'All Occurrence Space Use Coordinate XY': 'Space Use'}, inplace=True)

    # Rename the last column to Time
    raw_social_data.columns = [*raw_social_data.columns[:-1], 'Time']

    # Combine Year, Month, Day columns into VideoDate
    raw_social_data['VideoDate'] = (
                raw_social_data.iloc[:, -4].astype(str).str.zfill(2) + raw_social_data.iloc[:, -3].astype(
            str).str.zfill(2)
                + raw_social_data.iloc[:, -2].astype(str).str.zfill(2))
    # print(df.columns)

    social_data = raw_social_data[
        ['Observer', 'Focal Name', 'Behavior', 'Social Modifier', 'Space Use', 'VideoDate', 'Time']].copy()
    print(social_data)
    return social_data

def validate_number_of_monkeys(social_data):
    # For dates before 06/13/2022, 10 monkeys
    # after 06/13/2022, 8 monkeys
    # exception: 05/19/2022, 8 monkeys
    grouped = social_data.groupby('VideoDate')['Focal Name'].agg(['nunique', 'unique']).reset_index()

    for index, row in grouped.iterrows():
        video_date = row['VideoDate']
        # unique_values = ', '.join(row['unique'])
        count = row['nunique']
        # print(f"VideoDate: {video_date}, # of unique monkeys: {count}")

        expected_count = 10 if video_date < '20220613' else 8
        if video_date == '20220519':
            expected_count = 8
        else:
            expected_count = 10 if video_date < '20220613' else 8

        if count != expected_count:
            raise ValueError(
                f"Unexpected number of monkeys ({count}) observed on {video_date}. Expected: {expected_count} monkeys.")
        else:
            print(f"Validation passed! Valid number of monkeys for {video_date}")

def validate_number_of_interval_datapoints(social_data):
    if 'Behavior Abbrev' not in social_data.columns:
        social_data['Behavior Abbrev'] = social_data['Behavior'].str[:4].str.replace(' ', '')

    ''' CHECK NUMBER OF INTERVAL DATA '''
    # Create a mask to check if 'Behavior Abbrev' starts with 'I'
    mask = social_data['Behavior Abbrev'].str.startswith('I')

    # Filter the DataFrame to include only rows where 'Behavior Abbrev' starts with 'I'
    interval = social_data[mask].groupby('VideoDate')['Behavior Abbrev'].count().reset_index()
    filtered = interval[(interval['Behavior Abbrev'] != 120) & (interval['Behavior Abbrev'] != 96)]
    result = social_data[mask].groupby(['VideoDate', 'Focal Name']).size().reset_index(name='Count')
    filtered_result = result[(result['Count'] != 12)]

    if filtered.empty:
        print("Validation passed! Valid number of interval datapoints for all dates :)")
    else:
        raise ValueError(f'Invalid number of interval datapoints! : {filtered}')
        raise ValueError(f'Monkey specific interval datapoint count: {filtered_result}')


def extract_grooming_interactions(social_data):
    grooming = social_data[social_data['Behavior'].str.contains('groom')]
    grooming = grooming[['Focal Name', 'Social Modifier', 'Behavior']]
    print(grooming)

def extract_pairwise_interactions(social_data, social_interaction_type):
    social_data['Behavior Abbrev'] = social_data['Behavior'].str[:4].str.replace(' ', '')
    social_data = social_data[['Focal Name', 'Social Modifier', 'Behavior Abbrev']]

    if social_interaction_type.lower() == 'agonistic':
        subset = social_data[(social_data['Behavior Abbrev'] == 'AOA') | (social_data['Behavior Abbrev'] == 'IAG')]
    elif social_interaction_type.lower() == 'submissive':
        subset = social_data[(social_data['Behavior Abbrev'] == 'AOS') | (social_data['Behavior Abbrev'] == 'ISU')]
    elif social_interaction_type.lower() == 'affiliative':
        subset = social_data[(social_data['Behavior Abbrev'] == 'IAF')]
    else:
        raise ValueError("Invalid social interaction type. Please provide 'affiliative', 'submissive', or 'agonistic'.")

    subset = subset.dropna(subset=['Social Modifier'])  # remove all nan
    subset = expand_rows_on_comma(subset)
    subset_df = pd.DataFrame(subset)
    interaction_df = subset_df[['Focal Name', 'Social Modifier', 'Behavior Abbrev']]

    return interaction_df

def expand_rows_on_comma(df):
    # Splitting values with comma and creating new rows
    new_rows = []
    for index, row in df.iterrows():
        if ',' in row['Social Modifier']:
            modifiers = row['Social Modifier'].split(',')
            for modifier in modifiers:
                new_rows.append({col: row[col] for col in df.columns})
                new_rows[-1]['Social Modifier'] = modifier
        else:
            new_rows.append({col: row[col] for col in df.columns})
    return new_rows

def generate_edgelist_from_pairwise_interactions(interaction_df):

    if (interaction_df['Behavior Abbrev'].isin(['ISU', 'AOS'])).all():
        # Switch actor and receiver for the submissive behaviors
        temp = interaction_df['Focal Name'].copy()
        interaction_df['Focal Name'] = interaction_df['Social Modifier']
        interaction_df['Social Modifier'] = temp

    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edgelist = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    print(f'edge list {edgelist}')

    return edgelist


def combine_edgelists(edgelist1, edgelist2):
    # Concatenate the two edge lists
    combined_edgelist = pd.concat([edgelist1, edgelist2])
    # Group by 'Focal Name' and 'Social Modifier' and sum the weights
    combined_edgelist = combined_edgelist.groupby(['Focal Name', 'Social Modifier']).sum().reset_index()
    return combined_edgelist




if __name__ == '__main__':
    social_data = read_social_data_and_validate()
    extract_grooming_interactions(social_data)

