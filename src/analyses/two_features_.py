import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from statsmodels.regression.linear_model import OLS

import spike_rate_analysis
from recording_metadata_reader import RecordingMetadataReader

# zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
# metadata_reader = RecordingMetadataReader()
# raw_metadata = metadata_reader.get_raw_data()
# metadata_for_regression = raw_metadata.parse('InitialRegression')
#
# # Construct X
#
# # fit 2 features at a time
# num_features = X.shape[1]
# feature_combinations = list(combinations(range(num_features), 2))
#
# all_results = []
# for index, row in metadata_for_regression.iterrows():
#     date = row['Date'].strftime('%Y-%m-%d')
#     round_no = row['Round No.']
#     spike_rates = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_no)
#     spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
#
#     for ind, zombies_row in spike_rates_zombies.iterrows():
#         print(f"{date} round {round_no} : performing linear regression for {ind}")
#         Y = np.array(zombies_row.values).reshape(-1, 1)
#         for idx, (feat1_idx, feat2_idx) in enumerate(feature_combinations):
#             # Extract features
#             X_subset = X[:, [feat1_idx, feat2_idx]]
#             X_subset_adj = np.hstack((X_subset, np.ones((X_subset.shape[0], 1))))
#             # Fit the model
#             model = OLS(Y, X_subset_adj)
#             results = model.fit()
#
#             if (results.rsquared > 0.7) and (results.pvalues[0] < 0.05) and (results.pvalues[1] < 0.05):
#                 stat_params_to_be_saved = [date, round_no, ind, (feature_names[feat1_idx], feature_names[feat2_idx]),
#                                            results.rsquared, results.pvalues[0],  results.pvalues[1],  results.params]
#                 all_results.append(stat_params_to_be_saved)
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111, projection='3d')
#                 ax.scatter(X_subset_adj[:, 0], X_subset_adj[:, 1], Y)
#                 # Plot the regression plane
#                 x_surf = np.linspace(X_subset_adj[:, 0].min(), X_subset_adj[:, 0].max(), 100)
#                 y_surf = np.linspace(X_subset_adj[:, 1].min(), X_subset_adj[:, 1].max(), 100)
#                 x_surf, y_surf = np.meshgrid(x_surf, y_surf)
#                 exog = np.column_stack((x_surf.flatten(), y_surf.flatten(), np.ones(x_surf.flatten().shape)))
#                 out = results.predict(exog)
#                 ax.plot_surface(x_surf, y_surf, out.reshape(x_surf.shape), color='None', alpha=0.5)
#                 ax.set_title(f"{date} Round #{round_no}: {ind}")
#                 ax.set_xlabel(f"{feature_names[feat1_idx]}")
#                 ax.set_ylabel(f"{feature_names[feat2_idx]}")
#                 ax.set_zlabel("Average Firing Rate")
#                 ax.text(0.05, 0.05, 0.05, f"R-squared: {results.rsquared:.2f}",
#                         transform=ax.transAxes,
#                         verticalalignment='bottom', horizontalalignment='left', fontsize=8, color='blue')
#                 ax.text(0.05, 0.05, 0.05, f"p-value: {results.pvalues[0]:.4f}, {results.pvalues[1]:.4f}",
#                         transform=ax.transAxes,
#                         verticalalignment='bottom', horizontalalignment='right', fontsize=8, color='blue')
#                 plt.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
#                             f'two_feature_plots/{date}_{round_no}_{ind}_{feat1_idx},{feat2_idx}.png')
#                 plt.show()
#
# final_df = pd.DataFrame(all_results, columns=['Date', 'Round', 'Neuron', 'Feature Names', 'R-squared',
#                                              'p-value1', 'p-value2', 'coefficients'])
# final_df.to_excel('/home/connorlab/Documents/GitHub/Julie/linear_regression_results/two_feature_results.xlsx', index=False)
