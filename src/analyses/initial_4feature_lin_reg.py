import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from statsmodels.regression.linear_model import OLS

import social_data_processor
import spike_rate_analysis
from monkey_names import Monkey
from data_readers.recording_metadata_reader import RecordingMetadataReader
from single_channel_analysis import read_pickle


def construct_feature_matrix(beh, Sm_arrow, Sarrow_m, behavior_type):
    subjects_attraction_to_behavior = beh.iloc[:, 6]
    behavior_by_subject = pd.Series(beh.iloc[6, :].values)
    general_tendency_of_behavior = pd.Series(Sm_arrow)
    Sarrow_m = Sarrow_m.reset_index(drop=True)
    general_attraction_to_behavior = pd.Series(Sarrow_m)
    feat = pd.concat([subjects_attraction_to_behavior, behavior_by_subject,
                               general_tendency_of_behavior, general_attraction_to_behavior], axis=1,
                              keys=[f'subject\'s_attraction_to_{behavior_type}', f'{behavior_type}_by_subject',
                                    f'general_{behavior_type}', f'general_attraction_to_{behavior_type}'])
    feat.reset_index(drop=True, inplace=True)
    feature_names = [f'81G\'s attraction to {behavior_type}', f'{behavior_type} by 81G',
                     f'General {behavior_type}', f'General attraction to {behavior_type}']
    feature_matrix = np.delete(feat.values, 6, axis=0)  # delete information on 81G
    return feature_matrix, feature_names


def get_metadata_for_preliminary_analysis():
    metadata_reader = RecordingMetadataReader()
    raw_metadata = metadata_reader.get_raw_data()
    metadata_for_prelim_analysis = raw_metadata.parse('InitialRegression')

    return metadata_for_prelim_analysis


def create_overall_plot_for_single_feature_linear_regression_analysis(feature_matrix, response):
    """
    Creates 2x2 plot for different behavior types (Agonism, Submission, Affiliation) for each neuron and combine
    them into one figure -- three 2x2 raster_plots one for each behavior type

    @param feature_matrix:
    """
    zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
    zombies.remove('81G')
    fig = plt.figure(figsize=(26, 10))
    outer_grid = GridSpec(1, 3, figure=fig)

    behavior_types = ['agonism', 'submission', 'affiliation']

    feature_names = [
        f"{descriptor} {behavior}" for behavior in behavior_types for descriptor in [
            f"81G's attraction to",
            f"{behavior} by 81G",
            f"General",
            f"General attraction to"
        ]
    ]
    for i in range(len(behavior_types)):
        inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[i])
        start_col = i * 4
        beh_type_specific_feature_mat = feature_matrix[:, start_col:start_col + 4]

        for col_index in range(beh_type_specific_feature_mat.shape[1]):
            one_feature = beh_type_specific_feature_mat[:, col_index].reshape(-1, 1)
            augmented_x = np.hstack((one_feature, np.ones((one_feature.shape[0], 1))))
            model = OLS(response, augmented_x)
            results = model.fit()
            ax = plt.Subplot(fig, inner_grid[col_index])
            x = augmented_x[:, 0]
            ax.scatter(x, response)

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
            ax.plot(x_fit, y_fit, color='red')
            for n, label in enumerate(zombies):
                ax.text(x[n], response[n], label, ha='right', fontsize=6)
            # ax.set_xlabel(f"{feature_names[col_index]}")
            ax.set_ylabel(f"Average Firing Rate")
            ax.yaxis.set_label_coords(-0.1, 0.5)
            # ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")
            fig.add_subplot(ax)
    plt.tight_layout(pad=15.0)
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.035, right=0.99, hspace=0.19, wspace=0.165)
    plt.show()


def run_single_feature_linear_regression_analysis_individual_cells(features, spike_rates, feature_names, behavior_type):
    for index, row in spike_rates.iterrows():
        date = row['Date']
        round_no = row['Round No.']
        window = row['Time Window']
        int_window = (int(window[0]), int(window[1]))
        fig = plt.figure(figsize=(10, 14))
        fig.suptitle(f"{behavior_type} plots for {date} Round {round_no} {index} for {int_window[0]}ms to {int_window[1]}ms")
        cleaned_row = row.drop(['Date', 'Round No.', 'Time Window'])
        labels = cleaned_row.index
        y = np.array(cleaned_row.values).astype(float).reshape(-1, 1)
        for col_index in range(features.shape[1]):
            one_feature = features[:, col_index].reshape(-1, 1)
            augmented_X = np.hstack((one_feature, np.ones((one_feature.shape[0], 1))))
            model = OLS(y, augmented_X)
            results = model.fit()
            # print(results.summary())
            ax = fig.add_subplot(2, 2, col_index + 1)
            x = augmented_X[:, 0]
            ax.scatter(x, y)
            # Draw the fitted line
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
            ax.plot(x_fit, y_fit, color='red')
            for n, label in enumerate(labels):
                ax.text(x[n], y[n], label, ha='right', fontsize=6)
            ax.set_xlabel(f"{feature_names[col_index]}")
            ax.set_ylabel(f"Average Firing Rate")
            ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")

        # plt.show()
        fig.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
                        f'time_windowed_cells/{behavior_type}_{date}_round{round_no}_{index} for {int_window}.png')
        plt.close()


def run_single_feature_linear_regression_analysis(X, metadata_for_regression, feature_names, behavior_type):
    all_results = []
    cell_count = 0
    for index, row in metadata_for_regression.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        location = row['Location']
        spike_rates = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_no)
        spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
        labels = spike_rates_zombies.columns.tolist()
        for ind, zombies_row in spike_rates_zombies.iterrows():
            print(f"{date} round {round_no} : performing linear regression for {ind}")
            Y = np.array(zombies_row.values).reshape(-1, 1)
            fig = plt.figure(figsize=(10, 14))
            fig.suptitle(f"{date} Round #{round_no}: {ind}")
            cell_count += 1
            should_save_fig = False  # Initialize the flag

            for col_index in range(X.shape[1]):
                one_feature = X[:, col_index].reshape(-1, 1)
                augmented_X = np.hstack((one_feature, np.ones((one_feature.shape[0], 1))))
                model = OLS(Y, augmented_X)
                results = model.fit()
                ax = fig.add_subplot(2, 2, col_index + 1)
                x = augmented_X[:, 0]
                ax.scatter(x, Y)
                # Draw the fitted line
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
                ax.plot(x_fit, y_fit, color='red')
                for n, label in enumerate(labels):
                    ax.text(x[n], Y[n], label, ha='right', fontsize=6)
                ax.set_xlabel(f"{feature_names[col_index]}")
                ax.set_ylabel(f"Average Firing Rate")
                ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")
                # print(results.summary())
                if True:  # (results.rsquared > 0.7) and (results.pvalues[0] < 0.05):
                    should_save_fig = True
                    stat_params_to_be_saved = [date, round_no, ind, feature_names[col_index], results.rsquared,
                                               results.pvalues[0], results.params]
                    all_results.append(stat_params_to_be_saved)

            if should_save_fig:
                fig.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
                            f'single_feature_plots/{behavior_type}_{date}_round{round_no}_{location}_{ind}.png')
                # plt.show()
    # final_df = pd.DataFrame(all_results, columns=['Date', 'Round No.', 'Cell', 'Feature Name', 'R-squared',
    #                                               'p-value', 'coefficients'])
    # final_df.to_excel(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
    #                   f'single_feature_{behavior_type}.xlsx', index=False)
    # print(f'cell count: {cell_count}')
    # return fig


def generate_r_squared_histogram_for_specific_population(X, feature_names, location):
    metadata = get_metadata_for_preliminary_analysis()
    neural_population = metadata[metadata['Location'] == location]
    stat_param_list = []
    for index, row in neural_population.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        spike_rates = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_no)
        spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]

        for ind, zombies_row in spike_rates_zombies.iterrows():
            print(f"{date} round {round_no} : performing linear regression for {ind}")
            Y = np.array(zombies_row.values).reshape(-1, 1)

            for col_index in range(X.shape[1]):
                column = X[:, col_index].reshape(-1, 1)
                augmented_X = np.hstack((column, np.ones((column.shape[0], 1))))
                model = OLS(Y, augmented_X)
                results = model.fit()
                stat_params_to_be_saved = [date, round_no, ind, feature_names[col_index], results.rsquared,
                                           results.pvalues[0], results.params]
                stat_param_list.append(stat_params_to_be_saved)
    results_df = pd.DataFrame(stat_param_list, columns=['Date', 'Round', 'Neuron', 'Feature Name', 'R-squared',
                                                        'p-value', 'coefficients'])
    return results_df


def get_average_spike_rate_for_unsorted_cell_with_time_window(date, round_number, unsorted_cell, time_window):
    metadata_reader = RecordingMetadataReader()
    pickle_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_number) + ".pk1"
    compiled_dir = (Path(__file__).parent.parent.parent / 'compiled').resolve()
    pickle_filepath = os.path.join(compiled_dir, pickle_filename)
    print(f'Reading pickle file as raw data: {pickle_filepath}')
    raw_trial_data = read_pickle(pickle_filepath)
    average_spike_rate_for_unsorted_cell = (spike_rate_analysis.compute_avg_spike_rate_for_specific_cell
                                            (raw_trial_data, unsorted_cell, time_window))
    average_spike_rate_for_unsorted_cell['Date'] = date
    average_spike_rate_for_unsorted_cell['Round No.'] = round_number
    average_spike_rate_for_unsorted_cell['Time Window'] = [time_window]
    return average_spike_rate_for_unsorted_cell


def get_average_spike_rate_for_sorted_cell_with_time_window(date, round_number, sorted_cell, time_window):
    metadata_reader = RecordingMetadataReader()
    intan_dir = metadata_reader.get_intan_folder_name_for_specific_round(date, round_number)
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = Path(os.path.join(cortana_path, date, intan_dir))
    sorted_data = spike_rate_analysis.read_sorted_data(round_path)
    average_spike_rate_for_sorted_cell = (spike_rate_analysis.compute_avg_spike_rate_for_specific_cell
                                            (sorted_data, sorted_cell, time_window))
    average_spike_rate_for_sorted_cell['Date'] = date
    average_spike_rate_for_sorted_cell['Round No.'] = round_number
    average_spike_rate_for_sorted_cell['Time Window'] = [time_window]
    return average_spike_rate_for_sorted_cell


if __name__ == '__main__':
    zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
    '''
    Date: 2024-04-28
    Last Modified: 2024-04-29
    Generating 12 plots for the list of cells that Ed picked out -- with time window
    '''
    metadata_reader = RecordingMetadataReader()
    raw_metadata = metadata_reader.get_raw_data()
    metadata_for_prelim_analysis = raw_metadata.parse('Cells_fromEd')
    metadata_subset = metadata_for_prelim_analysis[['Date', 'Round No.', 'Cell',
                                                    'Time Window Start', 'Time Window End', 'Location']]
    metadata_cleaned = metadata_subset.dropna()
    mask = metadata_cleaned['Cell'].str.contains('Unit')
    sorted_cells = metadata_cleaned[mask]
    unsorted_cells = metadata_cleaned[~mask]

    unsorted_cells['Cell'] = unsorted_cells['Cell'].apply(enum_dict_resolvers.convert_to_enum)
    rows_for_unsorted = []
    for index, row in unsorted_cells.iterrows():
        time_window = (row['Time Window Start'], row['Time Window End'])
        spike_rate_unsorted = get_average_spike_rate_for_unsorted_cell_with_time_window(row['Date'].strftime('%Y-%m-%d'),
                                                                                        row['Round No.'], row['Cell'], time_window)
        rows_for_unsorted.append(spike_rate_unsorted)
    avg_spike_rate_for_unsorted = pd.concat(rows_for_unsorted)
    print(avg_spike_rate_for_unsorted)
    zombies_columns = [col for col in zombies if col in avg_spike_rate_for_unsorted.columns]
    required_columns = zombies_columns + [col for col in ['Date', 'Round No.', 'Time Window'] if
                                          col in avg_spike_rate_for_unsorted.columns]
    unsorted_spike_rates_zombies = avg_spike_rate_for_unsorted[required_columns]
    print(unsorted_spike_rates_zombies)

    rows_for_sorted_cells = []
    for index, row in sorted_cells.iterrows():
        time_window = (row['Time Window Start'], row['Time Window End'])
        spike_rate_sorted = get_average_spike_rate_for_sorted_cell_with_time_window(row['Date'].strftime('%Y-%m-%d'),
                                                                                    row['Round No.'], row['Cell'], time_window)
        rows_for_sorted_cells.append(spike_rate_sorted)
    avg_spike_rate_for_sorted = pd.concat(rows_for_sorted_cells)
    zombies_columns = [col for col in zombies if col in avg_spike_rate_for_sorted.columns]
    required_columns = zombies_columns + [col for col in ['Date', 'Round No.', 'Time Window'] if
                                          col in avg_spike_rate_for_sorted.columns]
    sorted_spike_rates_zombies = avg_spike_rate_for_sorted[required_columns]


    # '''
    #
    # Looking at each neuron using average spikes rate over 10 trials
    # 1D analysis for different types of behaviors (affiliation, submission, aggression)
    #
    # '''
    # # Get all experimental round information
    metadata_for_regression = get_metadata_for_preliminary_analysis()


    agon_beh, Sm_arrow_agon, Sarrow_m_agon = social_data_processor.partition_behavior_variance_from_excel_file(
        'feature_df_agonism.xlsx')
    sub_beh, Sm_arrow_sub, Sarrow_m_sub = social_data_processor.partition_behavior_variance_from_excel_file(
        'feature_df_submission.xlsx')
    aff_beh, Sm_arrow_aff, Sarrow_m_aff = social_data_processor.partition_behavior_variance_from_excel_file(
        'feature_df_affiliation.xlsx')

    X_agon, agon_feature_names = construct_feature_matrix(agon_beh, Sm_arrow_agon, Sarrow_m_agon, 'Agonism')
    X_sub, sub_feature_names = construct_feature_matrix(sub_beh, Sm_arrow_sub, Sarrow_m_sub, 'Submission')
    X_aff, aff_feature_names = construct_feature_matrix(aff_beh, Sm_arrow_aff, Sarrow_m_aff, 'Affiliation')

    run_single_feature_linear_regression_analysis_individual_cells(X_agon, sorted_spike_rates_zombies,
                                                                   agon_feature_names, 'Agonism')
    run_single_feature_linear_regression_analysis_individual_cells(X_sub, sorted_spike_rates_zombies,
                                                                   sub_feature_names, 'Submission')
    run_single_feature_linear_regression_analysis_individual_cells(X_aff, sorted_spike_rates_zombies,
                                                                   aff_feature_names, 'Affiliation')
    # run_single_feature_linear_regression_analysis(X_agon, metadata_for_regression, agon_feature_names, 'agonism')
    # run_single_feature_linear_regression_analysis(X_sub, metadata_for_regression, sub_feature_names, 'submission')
    # run_single_feature_linear_regression_analysis(X_aff, metadata_for_regression, aff_feature_names, 'affiliation')

    """
    Generating R-squared histograms
    
    """

    # location = 'Amygdala'
    # agon_df = generate_r_squared_histogram_for_specific_population(X_sub, sub_feature_names, location)
    # agon_sorted = agon_df.sort_values(by='R-squared', ascending=False)
    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle(f"R-squared values for Submission in {location}", fontsize=16)
    # axs = axs.flatten()
    # plot_number = 0
    #
    # for feature_name, group in agon_sorted.groupby('Feature Name'):
    #     filtered_group = group[group['R-squared'] > 0.1]
    #
    #     if not filtered_group.empty and plot_number < 4:
    #         ax = axs[plot_number]
    #         data = filtered_group['R-squared']
    #         counts, bins, patches = ax.hist(data, bins=20)
    #         top_20_percent = np.percentile(data, 80)
    #         for count, bin, patch in zip(counts, bins, patches):
    #             if bin >= top_20_percent:
    #                 patch.set_facecolor('red')
    #         ax.set_title(f'{feature_name}', fontsize=10)
    #         ax.set_xlabel('R-squared')
    #         ax.set_ylabel('Number of Cells')
    #         ax.grid(False)
    #         plot_number += 1
    #
    # plt.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/single_feature_r_squared_histograms/{location}_Submission_marked')
    # plt.show()

