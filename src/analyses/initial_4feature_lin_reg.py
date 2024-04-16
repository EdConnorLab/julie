import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS

import social_data_processor
import spike_rate_analysis
from monkey_names import Monkey
from recording_metadata_reader import RecordingMetadataReader


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



def run_single_feature_linear_regression_analysis(X, feature_names, behavior_type):
    # SINGLE FEATURE
    all_results = []
    for index, row in metadata_for_regression.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        spike_rates = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_no)
        spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
        labels = spike_rates_zombies.columns.tolist()
        for ind, zombies_row in spike_rates_zombies.iterrows():
            print(f"{date} round {round_no} : performing linear regression for {ind}")
            Y = np.array(zombies_row.values).reshape(-1, 1)
            fig = plt.figure(figsize=(12, 14))
            fig.suptitle(f"{date} Round #{round_no}: {ind}")

            should_save_fig = False  # Initialize the flag

            for col_index in range(X.shape[1]):
                column = X[:, col_index].reshape(-1, 1)
                augmented_X = np.hstack((column, np.ones((column.shape[0], 1))))
                model = OLS(Y, augmented_X)
                results = model.fit()
                ax = fig.add_subplot(2, 2, col_index + 1)
                ax.scatter(augmented_X[:, 0], Y)
                x = augmented_X[:, 0]
                # Draw the fitted line
                x_fit = np.linspace(min(augmented_X[:, 0]), max(augmented_X[:, 0]), 100)
                y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
                ax.plot(x_fit, y_fit, color='red')
                for n, label in enumerate(labels):
                    ax.text(x[n], Y[n], label, ha='right', fontsize=6)
                ax.set_xlabel(f"{feature_names[col_index]}")
                ax.set_ylabel(f"Average Firing Rate")
                ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")
                # print(results.summary())
                if (results.rsquared > 0.7) and (results.pvalues[0] < 0.05):
                    should_save_fig = True
                    stat_params_to_be_saved = [date, round_no, ind, feature_names[col_index], results.rsquared,
                                               results.pvalues[0], results.params]
                    all_results.append(stat_params_to_be_saved)
            if should_save_fig:
                fig.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
                                f'single_feature_plots/{behavior_type}_{date}_round{round_no}_{ind}.png')
                # plt.show()
    # final_df = pd.DataFrame(all_results, columns=['Date', 'Round', 'Neuron', 'Feature Name', 'R-squared',
    #                                              'p-value', 'coefficients'])
    # final_df.to_excel(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/single_feature_{behavior_type}.xlsx', index=False)


''' 

Looking at each neuron using average spikes rate over 10 trials
1D analysis for different types of behaviors (affiliation, submission, aggression)

'''
# Get all experimental round information
zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
metadata_reader = RecordingMetadataReader()
raw_metadata = metadata_reader.get_raw_data()
metadata_for_regression = raw_metadata.parse('InitialRegression')

agon_beh, Sm_arrow_agon, Sarrow_m_agon = social_data_processor.partition_behavior_variance_from_excel_file('feature_df_agonism.xlsx')
sub_beh, Sm_arrow_sub, Sarrow_m_sub = social_data_processor.partition_behavior_variance_from_excel_file('feature_df_submission.xlsx')
aff_beh, Sm_arrow_aff, Sarrow_m_aff = social_data_processor.partition_behavior_variance_from_excel_file('feature_df_affiliation.xlsx')

X_agon, agon_feature_names = construct_feature_matrix(agon_beh, Sm_arrow_agon, Sarrow_m_agon, 'agonism')
X_sub, sub_feature_names = construct_feature_matrix(sub_beh, Sm_arrow_sub, Sarrow_m_sub, 'submission')
X_aff, aff_feature_names = construct_feature_matrix(aff_beh, Sm_arrow_aff, Sarrow_m_aff, 'affiliation')

run_single_feature_linear_regression_analysis(X_agon, agon_feature_names, 'agonism')
run_single_feature_linear_regression_analysis(X_sub, sub_feature_names, 'submission')
run_single_feature_linear_regression_analysis(X_aff, aff_feature_names, 'affiliation')

