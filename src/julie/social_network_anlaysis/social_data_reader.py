import os
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def main():
    current_dir = os.getcwd()
    raw_data_file_name = 'ZombiesFinalRawData.xlsx'
    # file_path = '/Users/julie/General/LocalDocuments/GitHub/Julie/resources/ZombiesFinalRawData.xlsx'
    file_path = Path(current_dir).parent.parent.parent / 'resources' / raw_data_file_name
    df = read_social_raw_data(file_path)
    extracted_df = clean_social_raw_data(df)

    # Validate

    # Check number of monkeys recorded for each day
    try:
        validate_number_of_monkeys(df)
        print("Valid number of Monkeys")
    except ValueError as e:
        print("Validation failed: ", e)

    # Check number of interval datapoints for each day
    validate_number_of_interval_datapoints(df)

    ''' GET ALL AGONISTIC BEHAVIORS '''
    agonistic = extract_pairwise_interactions(df, 'agonistic')
    edge_weights = generate_weights_from_pairwise_interactions(agonistic)

    edge_weights.to_csv('edge_weights.csv', index=False)

    G = nx.from_pandas_edgelist(edge_weights, source='Focal Name', target='Social Modifier', edge_attr='weight',
                                    create_using=nx.DiGraph)
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    normalized_matrix = adj_matrix / np.sum(adj_matrix)
    print(adj_matrix)
    adj_df = pd.DataFrame(normalized_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))
    print(G.edges(data=True))
    # Extract weights for each edge
    weights = [d['weight'] for _, _, d in G.edges(data=True)]

    # Visualize the graph with a different layout and edge color
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes, seed for reproducibility
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, arrowsize=15, width=weights,
            edge_color='darkblue')
    plt.show()
def read_social_raw_data(filepath):
    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    # print(sheet_names)
    last_sheet = sheet_names[-1]
    df = xl.parse(last_sheet)
    # print(df.columns)
    return df

def clean_social_raw_data(df):
    # Rename columns for convenience
    df.rename(columns={'All Occurrence Value': 'Behavior'}, inplace=True)
    df.rename(columns={'All Occurrence Behavior Social Modifier': 'Social Modifier'}, inplace=True)
    df.rename(columns={'All Occurrence Space Use Coordinate XY': 'Space Use'}, inplace=True)

    # Rename the last column to Time
    df.columns = [*df.columns[:-1], 'Time']

    # Combine Year, Month, Day columns into VideoDate
    df['VideoDate'] = (df.iloc[:, -4].astype(str).str.zfill(2) + df.iloc[:, -3].astype(str).str.zfill(2)
                       + df.iloc[:,-2].astype(str).str.zfill(2))
    # print(df.columns)

    extracted_df = df[['Observer', 'Focal Name', 'Behavior', 'Social Modifier', 'Space Use', 'VideoDate', 'Time']]

    # print(extracted_df['VideoDate'].unique())
    print(f"{extracted_df['VideoDate'].nunique()} days of recording")



def generate_weights_from_pairwise_interactions(interaction_df):
    # Splitting values with comma and creating new rows
    new_rows = []
    for index, row in interaction_df.iterrows():
        if ',' in row['Social Modifier']:
            modifiers = row['Social Modifier'].split(',')
            for modifier in modifiers:
                new_rows.append({col: row[col] for col in interaction_df.columns})
                new_rows[-1]['Social Modifier'] = modifier
        else:
            new_rows.append({col: row[col] for col in interaction_df.columns})

    edge_weights = pd.DataFrame(new_rows).groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    return edge_weights

def extract_pairwise_interactions(df, social_interaction_type):
    if social_interaction_type.lower() == 'agonistic':
        subset = df[(df['Behavior Abbrev'] == 'AOA') | (df['Behavior Abbrev'] == 'IAG')]
    elif social_interaction_type.lower() == 'affiliative':
        subset = df[(df['Behavior Abbrev'] == 'IAF')]
    elif social_interaction_type.lower() == 'submissive':
        subset = df[(df['Behavior Abbrev'] == 'AOS') | (df['Behavior Abbrev'] == 'ISU')]
    else:
        raise ValueError("Invalid social interaction type. Please provide 'agonistic', 'affiliative', or 'submissive'.")
    subset = subset[['Focal Name', 'Social Modifier']]
    subset = subset.dropna(subset=['Social Modifier']) # remove all nan

    return subset

def validate_number_of_monkeys(df):
    # For dates before 06/13/2022, 10 monkeys
    # after 06/13/2022, 8 monkeys
    # exception: 05/19/2022, 8 monkeys
    grouped = df.groupby('VideoDate')['Focal Name'].agg(['nunique', 'unique']).reset_index()

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

def validate_number_of_interval_datapoints(df):

    ''' CHECK NUMBER OF INTERVAL DATA '''
    df['Behavior Abbrev'] = df['Behavior'].str[:4].str.replace(' ', '')

    # Create a mask to check if 'Behavior Abbrev' starts with 'I'
    mask = df['Behavior Abbrev'].str.startswith('I')

    # Filter the DataFrame to include only rows where 'Behavior Abbrev' starts with 'I'
    interval = df[mask].groupby('VideoDate')['Behavior Abbrev'].count().reset_index()
    filtered = interval[(interval['Behavior Abbrev'] != 120) & (interval['Behavior Abbrev'] != 96)]
    print(f'Invalid number of interval datapoints! : {filtered}')

    # Group by 'VideoDate' and 'Focal Name' and count the occurrences
    result = df[mask].groupby(['VideoDate', 'Focal Name']).size().reset_index(name='Count')
    filtered_result = result[(result['Count'] != 12)]
    print(f'Monkey specific interval datapoint count: {filtered_result}')



if __name__ == '__main__':
    main()