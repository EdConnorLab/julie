from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import spike_rate_analysis
from excel_data_reader import ExcelDataReader
from monkey_names import Monkey

excel_data_reader = ExcelDataReader(file_name='feature_df_submissive.xlsx')
beh = excel_data_reader.get_first_sheet()
beh = beh.iloc[:, 2:]  # extract only the values

Sm_arrow = beh.sum(axis=1) / (beh.shape[1] - 1)  # average frequency of the monkey m submitting to other monkeys
Sarrow_m = beh.sum(axis=0) / (beh.shape[0] - 1)  # average frequency of other monkeys submitting to the monkey m

sum_Sm_arrow = Sm_arrow.sum()  # this should be the same as sum_Sarrow_m = Sarrow_m.sum()

Sm_arrow_values = Sm_arrow.values.reshape(-1, 1)
Sarrow_m_values = Sarrow_m.values.reshape(-1, 1)

RSm_n = beh.values - ((Sm_arrow_values * Sarrow_m_values.T) / sum_Sm_arrow)
np.fill_diagonal(RSm_n, 0)
beh_final = pd.DataFrame(RSm_n, columns=beh.columns)
print(beh_final)  # this table is RSm->n

# Get 4 feature dimensions
attraction_to_submission_81G = beh_final.iloc[:, 6]
submission_by_81G = pd.Series(beh_final.iloc[6, :].values)
general_submission = Sm_arrow
general_attraction_to_submission = pd.Series(Sarrow_m.values)

# Combine to make X
feature_names = ['81G\'s attraction to submission', 'submission by 81G', 'general (overall) submission',
                              'general (overall) attraction to submission']
combined_df = pd.concat([attraction_to_submission_81G, submission_by_81G, general_submission,
                         general_attraction_to_submission], axis=1,
                        keys=['attraction_to_submission_81G', 'submission_by_81G', 'general_submission',
                              'general_attraction_to_submission'])
combined_df.reset_index(drop=True, inplace=True)
numeric_array = combined_df.values
X = np.delete(numeric_array, 6, axis=0)  # delete information on 81G


zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
zombies_df = pd.DataFrame({'Focal Name': zombies})

''' Looking at average spike rates'''
# spike_rates = spike_rate_analysis.compute_overall_average_spike_rates_for_each_round("2023-10-05", 1)
# matching_indices = zombies_df['Focal Name'].isin(spike_rates.index)
# matching_rows = spike_rates.loc[zombies_df.loc[matching_indices, 'Focal Name'].values]
# spike_rate_df = matching_rows.to_frame(name='Spike Rates')
# spike_rate_df['Focal Name'] = spike_rate_df.index
# spike_rate_df = pd.merge(zombies_df, spike_rate_df, on='Focal Name', how='left').fillna(0)
#
# # Extract values from a column as a NumPy array
# column_values = spike_rate_df['Spike Rates'].values
# # Convert the column values to a column matrix
# Y = column_values.reshape(-1, 1)
# # Append a column of 1s to the array to represent the last column
# aug_X = np.hstack((X, np.ones((numeric_array.shape[0], 1))))
# # Linear Regression
# lr = LinearRegression(fit_intercept=False)
# lr.fit(aug_X, Y)
#
# print('coeff')
# print(lr.coef_)
# model = OLS(Y, aug_X)
# results = model.fit()
# print(results.summary())

''' Looking at each neuron '''
date = "2023-09-26"
round_number = 1
spike_rates = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_number)
spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
print(spike_rates_zombies.head())
labels = spike_rates_zombies.columns.tolist()
print(f'Labels: {labels}')
# labels = ['Z_M1', 'Z_F1', 'Z_F2', 'Z_F3', 'Z_F4', 'Z_F5', 'Z_F7', 'Z_J1', 'Z_J2']
aug_X = np.hstack((X, np.ones((X.shape[0], 1))))

# for index, row in spike_rates_zombies.iterrows():
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#     print(f"LOOKING AT EACH NEURON: {date} round number {round_number} : performing linear regression for {index}")
#     Y = np.array(row.values).reshape(-1, 1)
#     model = OLS(Y, aug_X)
#     results = model.fit()
#     print(results.summary())
#     print(results.pvalues)
#     for i, ax in enumerate(axes.flatten()):
#         if i < aug_X.shape[1] - 1:  # Make sure we don't exceed the number of features
#             x = aug_X[:, i]
#             ax.plot(x, Y, marker='o', linestyle='None')
#             for n, label in enumerate(labels):
#                 ax.text(x[n], Y[n], label, ha='right')
#             ax.set_title(f"{date} Round No.{round_number} -- Neuron from {index}")
#             ax.set_ylabel('Average spike rate')
#             ax.set_xlabel(f'{feature_names[i]}')
#
#     plt.tight_layout()  # Adjust layout to prevent overlap
#     plt.show()

# fit 1 feature at a time
all_results_df = pd.DataFrame()
for index, row in spike_rates_zombies.iterrows():
    print(f"{date} round number {round_number} : performing linear regression for {index}")
    Y = np.array(row.values).reshape(-1, 1)
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle(f"{date} Round #{round_number}: {index}")
    all_stat_params = []
    for col_index in range(X.shape[1]):
        column = X[:, col_index].reshape(-1, 1)
        augmented_X = np.hstack((column, np.ones((column.shape[0], 1))))
        model = OLS(Y, augmented_X)
        results = model.fit()
        print(results.summary())
        stat_params_to_be_saved = [date, round_number, index, feature_names[col_index], results.rsquared, results.pvalues[0]]
        all_stat_params.append(stat_params_to_be_saved)

        # ax = fig.add_subplot(2, 2, col_index+1)
        # ax.scatter(augmented_X[:, 0], Y)
        # x = augmented_X[:, 0]
        # # Draw the fitted line
        # x_fit = np.linspace(min(augmented_X[:, 0]), max(augmented_X[:, 0]), 100)
        # y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
        # ax.plot(x_fit, y_fit, color='red', label='Fitted Line')
        # for n, label in enumerate(labels):
        #     ax.text(x[n], Y[n], label, ha='right', fontsize=7)
        # ax.set_xlabel(f"{feature_names[col_index]}")
        # ax.set_ylabel(f"Average Firing Rate")
        # ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")

    results_df = pd.DataFrame(all_stat_params, columns=['Date', 'Round', 'Neuron', 'Feature Name', 'R-squared', 'p-value'])
    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    all_results_df.to_excel('/home/connorlab/Documents/GitHub/Julie/linear_regression_results/single_feature_results.xlsx', index=False)
    plt.show()



''' Looking at each neuron two features at a time'''
# date = "2023-10-04"
# round_number = 3
# spike_rates = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_number)
# spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
# print(spike_rates_zombies.head())
# labels = spike_rates_zombies.columns.tolist()
'''
# fit 2 features at a time
num_features = X.shape[1]
feature_combinations = list(combinations(range(num_features), 2))

for index, row in spike_rates_zombies.iterrows():
    print(f"{date} round number {round_number} : performing linear regression for {index}")
    Y = np.array(row.values).reshape(-1, 1)
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle(f"{date} Round #{round_number}: {index}")
    for idx, (feat1_idx, feat2_idx) in enumerate(feature_combinations):
        # Extract features
        X_subset = X[:, [feat1_idx, feat2_idx]]
        X_subset_adj = np.hstack((X_subset, np.ones((X_subset.shape[0], 1))))
        # Fit the model
        model = OLS(Y, X_subset_adj)
        results = model.fit()
        formatted_coeffs = ", ".join([f"{param:.2f}" for param in results.params])

        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        ax.scatter(X_subset_adj[:, 0], X_subset_adj[:, 1], Y)
        # Plot the regression plane
        x_surf = np.linspace(X_subset_adj[:, 0].min(), X_subset_adj[:, 0].max(), 100)
        y_surf = np.linspace(X_subset_adj[:, 1].min(), X_subset_adj[:, 1].max(), 100)
        x_surf, y_surf = np.meshgrid(x_surf, y_surf)
        exog = np.column_stack((x_surf.flatten(), y_surf.flatten(), np.ones(x_surf.flatten().shape)))
        out = results.predict(exog)
        ax.plot_surface(x_surf, y_surf, out.reshape(x_surf.shape), color='None', alpha=0.5)
        ax.text(0.05, 0.05, 0.05, f"R-squared: {results.rsquared:.2f} Coeff: {formatted_coeffs}", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left', fontsize=8, color='blue')
    plt.show()
'''

''' Looking at individual trial '''
# spike_rates = spike_rate_analysis.get_spike_rates_for_each_trial("2023-10-04", 3)
# spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
# print(spike_rates_zombies.head())
#
# for index, row in spike_rates_zombies.iterrows():
#     repeated_X_rows = []
#     spike_rate_list = []
#     row_index = 0
#     print(f"for {index}, linear regression computed")
#     for column_name, value in row.items():
#         print(f"\t{len(value)} spike rates for {column_name} and {value.count(0)} of them are zeros")
#         spike_rate_list.extend(value)
#         repeated_X_rows.extend([X[row_index]] * len(value))
#         row_index += 1
#     Y = np.array(spike_rate_list).reshape(-1, 1)
#     final_X = np.array(repeated_X_rows)
#
#     model = OLS(Y, final_X)
#     results = model.fit()
#     print(results.summary())
