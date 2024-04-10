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
    general_tendency_of_behavior = Sm_arrow
    general_attraction_to_behavior = pd.Series(Sarrow_m)
    feat = pd.concat([subjects_attraction_to_behavior, behavior_by_subject,
                               general_tendency_of_behavior, general_attraction_to_behavior], axis=1,
                              keys=[f'subject\'s_attraction_to_{behavior_type}', f'{behavior_type}_by_subject',
                                    f'general_{behavior_type}', f'general_attraction_to_{behavior_type}'])
    feat.reset_index(drop=True, inplace=True)
    feature_names = [f'subject\'s_attraction_to_{behavior_type}', f'{behavior_type}_by_subject',
                     f'general_{behavior_type}', f'general_attraction_to_{behavior_type}']
    feature_matrix = np.delete(feat.values, 6, axis=0)  # delete information on 81G
    return feature_matrix, feature_names


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

        for col_index in range(X.shape[1]):
            column = X[:, col_index].reshape(-1, 1)
            augmented_X = np.hstack((column, np.ones((column.shape[0], 1))))
            model = OLS(Y, augmented_X)
            results = model.fit()
            # print(results.summary())
            if (results.rsquared > 0.7) and (results.pvalues[0] < 0.05):
                stat_params_to_be_saved = [date, round_no, ind, feature_names[col_index], results.rsquared,
                                           results.pvalues[0], results.params]
                all_results.append(stat_params_to_be_saved)

                plt.scatter(augmented_X[:, 0], Y)
                x = augmented_X[:, 0]
                # Draw the fitted line
                x_fit = np.linspace(min(augmented_X[:, 0]), max(augmented_X[:, 0]), 100)
                y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
                plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
                for n, label in enumerate(labels):
                    plt.text(x[n], Y[n], label, ha='right', fontsize=6)
                plt.text(1, 0, f"R-squared {results.rsquared:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes)
                plt.text(0, 0, f"P-value {results.pvalues[0]:.4f}", ha='left', va='bottom', transform=plt.gca().transAxes)
                plt.xlabel(f"{feature_names[col_index]}")
                plt.ylabel(f"Average Firing Rate")
                plt.title(f"{date} round {round_no}: {ind}")
                plt.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
                            f'single_feature_plots/{date}_{round_no}_{ind}.png')
                plt.show()
final_df = pd.DataFrame(all_results, columns=['Date', 'Round', 'Neuron', 'Feature Name', 'R-squared',
                                             'p-value', 'coefficients'])
final_df.to_excel('/home/connorlab/Documents/GitHub/Julie/linear_regression_results/single_feature_results.xlsx', index=False)

