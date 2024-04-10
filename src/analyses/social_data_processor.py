import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS

from excel_data_reader import ExcelDataReader
from social_data_reader import SocialDataReader
from sklearn.linear_model import LinearRegression

from behaviors import AgonisticBehaviors as Agonistic
from behaviors import SubmissiveBehaviors as Submissive
from behaviors import AffiliativeBehaviors as Affiliative
from behaviors import IndividualBehaviors as Individual
from monkey_names import Monkey
import spike_rate_analysis


def extract_specific_social_behavior(social_data, social_behavior):
    if isinstance(social_behavior, (Agonistic or Submissive or Affiliative or Individual)):
        specific_behavior = social_data[social_data['Behavior'].str.contains(social_behavior.value, case=False)]
        specific_behavior = specific_behavior[['Focal Name', 'Social Modifier', 'Behavior', 'Behavior Abbrev']]
        subset = specific_behavior.dropna(subset=['Social Modifier'])  # remove all nan
        subset = expand_rows_on_comma(subset)
        subset_df = pd.DataFrame(subset)
    elif isinstance(social_behavior, list):
        specific_behavior = pd.DataFrame()
        for beh in social_behavior:
            temp = social_data[social_data['Behavior'].str.contains(beh.value, case=False)]
            extracted_behavior = temp[['Focal Name', 'Social Modifier', 'Behavior', 'Behavior Abbrev']]
            specific_behavior = pd.concat([specific_behavior, extracted_behavior])
            # print("Length of specific behavior updated to")
            # print(specific_behavior.shape[0])
        subset = specific_behavior.dropna(subset=['Social Modifier'])  # remove all nan
        subset = expand_rows_on_comma(subset)
        subset_df = pd.DataFrame(subset)
    else:
        raise ValueError('Invalid social behavior')
    return subset_df


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


def combine_edge_lists(edge_list1, edge_list2):
    # Concatenate the two edge lists
    combined_edge_list = pd.concat([edge_list1, edge_list2])
    # Group by 'Focal Name' and 'Social Modifier' and sum the weights
    combined_edge_list = combined_edge_list.groupby(['Focal Name', 'Social Modifier']).sum().reset_index()
    return combined_edge_list


def generate_edge_list_from_extracted_interactions(interaction_df):
    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edge_list = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    return edge_list


def generate_edge_list_from_pairwise_interactions(interaction_df):
    if (interaction_df['Behavior Abbrev'].isin(['ISU', 'AOS'])).all():
        # Switch actor and receiver for the submissive behaviors
        temp = interaction_df['Focal Name'].copy()
        interaction_df['Focal Name'] = interaction_df['Social Modifier']
        interaction_df['Social Modifier'] = temp
    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edge_list = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    print(f'edge list {edge_list}')
    return edge_list


def partition_behavior_variance_from_excel_file(file_name):
    excel_data_reader = ExcelDataReader(file_name=file_name)
    beh = excel_data_reader.get_first_sheet()
    beh = beh.iloc[:, 1:]  # extract only the values
    beh_final, Sm_arrow_values, Sarrow_m_values = partition_behavior_variance(beh)
    return beh_final, Sm_arrow_values, Sarrow_m_values


def partition_behavior_variance(beh: pd.DataFrame):
    Sm_arrow = beh.sum(axis=1) / (beh.shape[1] - 1)  # average frequency of the monkey m submitting to other monkeys
    Sarrow_m = beh.sum(axis=0) / (beh.shape[0] - 1)  # average frequency of other monkeys submitting to the monkey m

    sum_Sm_arrow = Sm_arrow.sum()  # this should be the same as sum_Sarrow_m = Sarrow_m.sum()

    Sm_arrow_values = Sm_arrow.values.reshape(-1, 1)
    Sarrow_m_values = Sarrow_m.values.reshape(-1, 1)

    RSm_n = beh.values - ((Sm_arrow_values * Sarrow_m_values.T) / sum_Sm_arrow)
    np.fill_diagonal(RSm_n, 0)
    beh_final = pd.DataFrame(RSm_n, columns=beh.columns)
    # print(beh_final)  # this table is RSm->n
    return beh_final, Sm_arrow_values, Sarrow_m_values


def generate_feature_matrix_from_edge_list(edge_list, monkey_group):
    feature_df = pd.DataFrame({'Focal Name': monkey_group})
    for monkey in monkey_group:
        fill_in = pd.DataFrame({'Focal Name': monkey_group, 'Social Modifier': monkey, 'weight': 0})
        dim = edge_list[edge_list['Social Modifier'] == monkey]
        one_dim = pd.merge(fill_in, dim, how='left', on=['Focal Name', 'Social Modifier'])
        one_dim['weight'] = (one_dim['weight_x'] + one_dim['weight_y']).fillna(0)
        one_dim.drop(['weight_x','weight_y'], axis=1, inplace=True)
        one_dim['weight'] = one_dim['weight'].astype(int)
        one_dim.rename(columns={'weight': f'Behavior Towards {monkey}'}, inplace=True)
        one_dim.drop('Social Modifier', axis=1, inplace=True)
        feature_df = pd.merge(feature_df, one_dim, on='Focal Name')
    return feature_df



if __name__ == '__main__':

    social_data = SocialDataReader().social_data
    # Agonistic
    agonistic_behaviors = list(Agonistic)
    agon = extract_specific_social_behavior(social_data, agonistic_behaviors)
    edge_list_agon = generate_edge_list_from_extracted_interactions(agon)

    # Submissive
    submissive_behaviors = list(Submissive)
    sub = extract_specific_social_behavior(social_data, submissive_behaviors)
    edge_list_sub = generate_edge_list_from_extracted_interactions(sub)

    # Affiliative
    affiliative_behaviors = list(Affiliative)
    aff = extract_specific_social_behavior(social_data, affiliative_behaviors)
    edge_list_aff = generate_edge_list_from_extracted_interactions(aff)

    # Get genealogy matrix
    excel_data_reader = ExcelDataReader(file_name='genealogy_matrix.xlsx')
    genealogy_data = excel_data_reader.get_first_sheet()
    print(genealogy_data)
    genealogy_data['Focal Name'] = genealogy_data['Focal Name'].astype(str)

    zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
    matrix = generate_feature_matrix_from_edge_list(edge_list_agon, zombies)

    # Normalize
    # cols_to_normalize = feature_df.select_dtypes(include='int64').columns[1:]
    # normalized_cols = feature_df[cols_to_normalize] / feature_df[cols_to_normalize].max()
    # normalized_df = pd.concat([feature_df.iloc[:, 0], normalized_cols, feature_df.select_dtypes(exclude='int64')], axis=1)
    # print(normalized_df.shape)

    # Get only numbers
    # numeric_columns = normalized_df.iloc[:, 1:-1]
    # Convert selected numeric columns to a NumPy array
    # numeric_array = numeric_columns.values
    # # Append a column of 1s to the array to represent the last column
    # numeric_array_with_ones = np.hstack((numeric_array, np.ones((numeric_array.shape[0], 1))))
    # # Compute pseudoinverse of the resulting array
    # X = numeric_array_with_ones


    # Get Spike Rate -- Y

    #ER_population_spike_rate = spike_rate_analysis.compute_population_spike_rates_for_ER()
    population_spike_rate = spike_rate_analysis.compute_overall_average_spike_rates_for_each_round("2023-09-29", 2)
    print("population spike rate")
    # print(population_spike_rate)
    # matching_indices = zombies_df['Focal Name'].isin(population_spike_rate.index)
    # matching_rows = population_spike_rate.loc[zombies_df.loc[matching_indices, 'Focal Name'].values]
    # spike_rate_df = matching_rows.to_frame(name='Spike Rates')
    # spike_rate_df['Focal Name'] = spike_rate_df.index
    # spike_rate_df = pd.merge(zombies_df, spike_rate_df, on='Focal Name', how='left').fillna(0)
    #
    # # Extract values from a column as a NumPy array
    # column_values = spike_rate_df['Spike Rates'].values
    # # Convert the column values to a column matrix
    # Y = column_values.reshape(-1, 1)
    #
    # lr = LinearRegression(fit_intercept=False)
    # lr.fit(numeric_array, Y)
    #
    # print('coeff')
    # print(lr.coef_)
    # model = OLS(Y, X)
    # results = model.fit()
    # print(results.summary())

    # submissive_behavior_list = list(Submissive)
    # submissive = extract_specific_social_behavior(social_data, submissive_behavior_list)
    # edgelist_submissive = generate_edgelist_from_extracted_interactions(submissive)
    #
    # print(edgelist_submissive['Social Modifier'].unique())

