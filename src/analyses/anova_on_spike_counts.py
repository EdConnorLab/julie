import math
import numpy as np
import pandas as pd


from initial_4feature_lin_reg import get_spike_count_for_single_neuron_with_time_window, \
    get_metadata_for_list_of_cells_with_time_window, get_metadata_for_preliminary_analysis
import spike_count
from monkey_names import Zombies, BestFrans
from data_readers.recording_metadata_reader import RecordingMetadataReader
from scipy.stats import f_oneway

from spike_rate_computation import get_average_spike_rates_for_each_monkey, get_spike_rates_for_each_trial

def perform_anova_on_dataframe_rows(df):
    """
    Perform one-way ANOVA on rows of a DataFrame
    """
    results = []
    significant_results = []
    for index, row in df.iterrows():
        # Extract groups as lists
        groups = [group for group in row if isinstance(group, list)]
        # Perform OneWay ANOVA
        f_val, p_val = f_oneway(*groups)
        results.append((f_val, p_val))
        # if p_val < 0.05:
        #     significant_results.append((date, round_no, index, p_val))
    return results, significant_results


def perform_anova_on_dataframe_rows_for_time_windowed(df):
    """
    Perform one-way ANOVA on rows of a DataFrame
    """
    results = []
    significant_results = []
    for index, row in df.iterrows():
        # Extract groups as lists
        groups = [group for group in row if isinstance(group, list)]
        # Perform OneWay ANOVA
        f_val, p_val = f_oneway(*groups)
        results.append({'Date': row['Date'], 'Round No.': row['Round No.'],
                        'Time Window': row['Time Window'],
                        'Cell': index, 'F Value': f_val, 'P Value': p_val})
        if p_val < 0.05:

            # Collect significant result data
            significant_results.append({
                'Date': row['Date'],
                'Round No.': row['Round No.'],
                'Time Window': row['Time Window'],
                'Cell': index,
                'P Value': p_val
            })
    results_df = pd.DataFrame(results)
    significant_results_df = pd.DataFrame(significant_results)
    return results_df, significant_results_df


def anova_permutation_test(groups, num_permutations=1000):
    data = np.concatenate(groups)
    original_group_sizes = [len(group) for group in groups]
    observed_f_stat, _ = f_oneway(*groups)

    permutation_f_stats = []
    for _ in range(num_permutations):
        np.random.shuffle(data)
        new_groups = np.split(data, np.cumsum(original_group_sizes)[:-1])
        f_stat, _ = f_oneway(*new_groups)
        permutation_f_stats.append(f_stat)

    p_value = np.mean([f_stat >= observed_f_stat for f_stat in permutation_f_stats])
    return observed_f_stat, p_value


def perform_anova_permutation_test_on_rows(df, num_permutations=1000):
    results = []
    total_sig = 0
    for index, row in df.iterrows():
        groups = [np.array(cell) for cell in row if isinstance(cell, list)]
        f_stat, p_value = anova_permutation_test(groups, num_permutations=num_permutations)
        # results.append((f_stat, p_value))
        if p_value < 0.05 and not math.isnan(f_stat):
            results.append((index, f_stat, p_value))
            print(f"Row {index}: F-statistic = {f_stat}, p-value = {p_value}")
            total_sig += 1
    return results, total_sig


def two_sample_t_test(df):
    '''
    Compare if the means of two groups are different

    Use this for comparing if there is a differential response
    for one group (i.e. Zombies) vs rest of the groups (i.e. Best Frans, Instigators, etc.)
    '''
    cells = get_metadata_for_preliminary_analysis()
    neural_population = cells[cells['Location'] == 'ER']
    stat_param_list = []
    for index, row in neural_population.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        spike_rates = get_spike_rates_for_each_trial(date, round_no)
        print(spike_rates)
    # TODO: finish writing this function

if __name__ == '__main__':
    '''
    Date Created : 2024-04-29
    ANOVA for selected cells from Ed (time windowed)
    
    Last Updated: 2024-07-03 
    Latest Updates: running ANOVA for BestFrans
    '''


    # Get list of monkey names
    zombies = [member.value for name, member in Zombies.__members__.items()]
    bestfrans = [member.value for name, member in BestFrans.__members__.items()]

    # Get metadata for list of cells (time windowed)
    time_windowed_cells = get_metadata_for_list_of_cells_with_time_window("BestFrans_Cells")

    # calculate spike count
    spike_count = get_spike_count_for_single_neuron_with_time_window(time_windowed_cells)
    bestfrans_columns = [col for col in zombies if col in spike_count.columns]
    bestfrans_spike_count = spike_count[bestfrans_columns]
    # adding in the Date and Round No. columns because above operation got rid of it
    bestfrans_spike_count['Date'] = spike_count['Date']
    bestfrans_spike_count['Round No.'] = spike_count['Round No.']

    anova_results, sig_results = perform_anova_on_dataframe_rows_for_time_windowed(bestfrans_spike_count)
    sig_results.to_csv('sig_ANOVA_results_for_2nd_list_zombies.csv')
    anova_results.to_csv('Windowed_ANOVA_sig_results_Zombies_new.csv')

    '''
    Date Created: 2024-04-29
    Last Updated: 2024-??-??
    ANOVA or PermANOVA on all rounds from metadata
    '''
    # metadata_for_analysis = get_metadata_for_preliminary_analysis()
    # total_sig_cells = 0
    # all_significant_results = []
    # for _, row in metadata_for_analysis.iterrows():
    #     date = row['Date'].strftime('%Y-%m-%d')
    #     round_no = row['Round No.']
    #     spike_count_dataframe = get_spike_count_for_each_trial(date, round_no)
    #     # anova_results, sig_results = perform_anova_on_rows(spike_count_dataframe)
    #     results, sig = perform_anova_permutation_test_on_rows(spike_count_dataframe, num_permutations=1000)
    #     for result in results:
    #         index, f_stat, p_value = result
    #         to_be_saved = [date, round_no, index, f_stat, p_value]
    #         all_significant_results.append(to_be_saved)
    #     total_sig_cells = total_sig_cells + sig
    #     # print(f'For {date} Round No. {round_no}')
    #     # all_significant_results.extend(sig_results)
    #     # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'P-Value'])
    #     # results_file_path = 'significant_anova_results.xlsx'
    #     # results_df.to_excel(results_file_path, index=False)
    # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'F-statistics', 'P-Value'])
    # results_file_path = 'significant_anova_results.xlsx'
    # results_df.to_excel(results_file_path, index=False)
    # print(total_sig_cells)