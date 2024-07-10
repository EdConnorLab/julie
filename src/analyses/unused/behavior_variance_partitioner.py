#
# zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
#
# ''' Looking at average spike rates'''
# # second_set_of_time_windowed_cells = spike_rate_analysis.compute_overall_average_spike_rates_for_each_round("2023-10-05", 1)
# # matching_indices = zombies_df['Focal Name'].isin(second_set_of_time_windowed_cells.index)
# # matching_rows = second_set_of_time_windowed_cells.loc[zombies_df.loc[matching_indices, 'Focal Name'].values]
# # spike_rate_df = matching_rows.to_frame(name='Spike Rates')
# # spike_rate_df['Focal Name'] = spike_rate_df.index
# # spike_rate_df = pd.merge(zombies_df, spike_rate_df, on='Focal Name', how='left').fillna(0)
# #
# # # Extract values from a column as a NumPy array
# # column_values = spike_rate_df['Spike Rates'].values
# # # Convert the column values to a column matrix
# # Y = column_values.reshape(-1, 1)
# # # Append a column of 1s to the array to represent the last column
# # aug_X = np.hstack((X, np.ones((numeric_array.shape[0], 1))))
# # # Linear Regression
# # lr = LinearRegression(fit_intercept=False)
# # lr.fit(aug_X, Y)
# #
# # print('coeff')
# # print(lr.coef_)
# # model = OLS(Y, aug_X)
# # results = model.fit()
# # print(results.summary())
#
# ''' Looking at each neuron '''
# date = "2023-11-28"
# round_number = 4
# second_set_of_time_windowed_cells = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_number)
# spike_rates_zombies = second_set_of_time_windowed_cells[[col for col in zombies if col in second_set_of_time_windowed_cells.columns]]
# print(spike_rates_zombies.head())
# labels = spike_rates_zombies.columns.tolist()
# print(f'Labels: {labels}')
# # labels = ['Z_M1', 'Z_F1', 'Z_F2', 'Z_F3', 'Z_F4', 'Z_F5', 'Z_F7', 'Z_J1', 'Z_J2']
# aug_X = np.hstack((X, np.ones((X.shape[0], 1))))
#
#
# # fit 1 feature at a time
# all_results_df = pd.DataFrame()
# for index, row in spike_rates_zombies.iterrows():
#     print(f"{date} round number {round_number} : performing linear regression for {index}")
#     Y = np.array(row.values).reshape(-1, 1)
#     fig = plt.figure(figsize=(12, 14))
#     fig.suptitle(f"{date} Round #{round_number}: {index}")
#     all_stat_params = []
#     for col_index in range(X.shape[1]):
#         column = X[:, col_index].reshape(-1, 1)
#         augmented_X = np.hstack((column, np.ones((column.shape[0], 1))))
#         model = OLS(Y, augmented_X)
#         results = model.fit()
#         print(results.summary())
#         stat_params_to_be_saved = [date, round_number, index, feature_names[col_index], results.rsquared,
#                                    results.pvalues[0], results.params]
#         all_stat_params.append(stat_params_to_be_saved)
#
#         ax = fig.add_subplot(2, 2, col_index+1)
#         ax.scatter(augmented_X[:, 0], Y)
#         x = augmented_X[:, 0]
#         # Draw the fitted line
#         x_fit = np.linspace(min(augmented_X[:, 0]), max(augmented_X[:, 0]), 100)
#         y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
#         ax.plot(x_fit, y_fit, color='red', label='Fitted Line')
#         for n, label in enumerate(labels):
#             ax.text(x[n], Y[n], label, ha='right', fontsize=7)
#         ax.set_xlabel(f"{feature_names[col_index]}")
#         ax.set_ylabel(f"Average Firing Rate")
#         ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")
#
#     results_df = pd.DataFrame(all_stat_params, columns=['Date', 'Round', 'Neuron', 'Feature Name', 'R-squared',
#                                                         'p-value', 'coefficients'])
#     all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
#     all_results_df.to_excel('/home/connorlab/Documents/GitHub/Julie/all_cells_linear_regression/single_feature_results_from_partition.xlsx', index=False)
#     plt.show()
#
#
#
# ''' Looking at each neuron two features at a time'''
# # date = "2023-10-04"
# # round_number = 3
# # second_set_of_time_windowed_cells = spike_rate_analysis.get_average_spike_rates_for_each_monkey(date, round_number)
# # spike_rates_zombies = second_set_of_time_windowed_cells[[col for col in zombies if col in second_set_of_time_windowed_cells.columns]]
# # print(spike_rates_zombies.head())
# # labels = spike_rates_zombies.columns.tolist()
# '''
# # fit 2 features at a time
# num_features = X.shape[1]
# feature_combinations = list(combinations(range(num_features), 2))
#
# for index, row in spike_rates_zombies.iterrows():
#     print(f"{date} round number {round_number} : performing linear regression for {index}")
#     Y = np.array(row.values).reshape(-1, 1)
#     fig = plt.figure(figsize=(12, 14))
#     fig.suptitle(f"{date} Round #{round_number}: {index}")
#     for idx, (feat1_idx, feat2_idx) in enumerate(feature_combinations):
#         # Extract features
#         X_subset = X[:, [feat1_idx, feat2_idx]]
#         X_subset_adj = np.hstack((X_subset, np.ones((X_subset.shape[0], 1))))
#         # Fit the model
#         model = OLS(Y, X_subset_adj)
#         results = model.fit()
#         formatted_coeffs = ", ".join([f"{param:.2f}" for param in results.params])
#
#         ax = fig.add_subplot(2, 3, idx+1, projection='3d')
#         ax.scatter(X_subset_adj[:, 0], X_subset_adj[:, 1], Y)
#         # Plot the regression plane
#         x_surf = np.linspace(X_subset_adj[:, 0].min(), X_subset_adj[:, 0].max(), 100)
#         y_surf = np.linspace(X_subset_adj[:, 1].min(), X_subset_adj[:, 1].max(), 100)
#         x_surf, y_surf = np.meshgrid(x_surf, y_surf)
#         exog = np.column_stack((x_surf.flatten(), y_surf.flatten(), np.ones(x_surf.flatten().shape)))
#         out = results.predict(exog)
#         ax.plot_surface(x_surf, y_surf, out.reshape(x_surf.shape), color='None', alpha=0.5)
#         ax.text(0.05, 0.05, 0.05, f"R-squared: {results.rsquared:.2f} Coeff: {formatted_coeffs}", transform=ax.transAxes,
#                 verticalalignment='bottom', horizontalalignment='left', fontsize=8, color='blue')
#     plt.show()
# '''
#
# ''' Looking at individual trial '''
# # second_set_of_time_windowed_cells = spike_rate_analysis.get_spike_rates_for_each_trial("2023-10-04", 3)
# # spike_rates_zombies = second_set_of_time_windowed_cells[[col for col in zombies if col in second_set_of_time_windowed_cells.columns]]
# # print(spike_rates_zombies.head())
# #
# # for index, row in spike_rates_zombies.iterrows():
# #     repeated_X_rows = []
# #     spike_rate_list = []
# #     row_index = 0
# #     print(f"for {index}, linear regression computed")
# #     for column_name, value in row.items():
# #         print(f"\t{len(value)} spike rates for {column_name} and {value.count(0)} of them are zeros")
# #         spike_rate_list.extend(value)
# #         repeated_X_rows.extend([X[row_index]] * len(value))
# #         row_index += 1
# #     Y = np.array(spike_rate_list).reshape(-1, 1)
# #     final_X = np.array(repeated_X_rows)
# #
# #     model = OLS(Y, final_X)
# #     results = model.fit()
# #     print(results.summary())
