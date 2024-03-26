import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from sklearn.linear_model import LinearRegression
import spike_rate_analysis
from excel_data_reader import ExcelDataReader
from monkey_names import Monkey

excel_data_reader = ExcelDataReader(file_name='feature_df_submissive.xlsx')
beh = excel_data_reader.get_first_sheet()
beh = beh.iloc[:, 11:] # only extract the beh columns

Sm_arrow = beh.sum(axis=1) / (beh.shape[1] - 1)
Sarrow_m = beh.sum(axis=0) / (beh.shape[0] - 1)

sum_Sm_arrow = Sm_arrow.sum() # these two values should be the same
sum_Sarrow_m = Sarrow_m.sum() # these two values should be the same

Sm_arrow_values = Sm_arrow.values.reshape(-1,1)
Sarrow_m_values = Sarrow_m.values.reshape(-1,1)

Sm_arrow_n = Sm_arrow_values * Sarrow_m_values.T

temp = Sm_arrow_n / sum_Sm_arrow
final = beh.values - temp
np.fill_diagonal(final, 0)
beh_final = pd.DataFrame(final, columns=beh.columns)
print(beh_final) # this table is RSm->n
# beh_final.to_excel('submissive_adjusted.xlsx', index=False)

# Get 81G
attraction_to_submission_81G = beh_final.iloc[:, 6]
submission_by_81G = pd.Series(beh_final.iloc[6, :].values)
general_submission = Sm_arrow
general_attraction_to_submission = pd.Series(Sarrow_m.values)

combined_df = pd.concat([attraction_to_submission_81G, submission_by_81G, general_submission, general_attraction_to_submission], axis=1,
                        keys=['attraction_to_submission_81G', 'submission_by_81G', 'general_submission', 'general_attraction_to_submission'])
combined_df.reset_index(drop=True, inplace=True)
numeric_array = combined_df.values
# Append a column of 1s to the array to represent the last column
numeric_array_with_ones = np.hstack((numeric_array, np.ones((numeric_array.shape[0], 1))))
# Compute pseudoinverse of the resulting array
X = numeric_array_with_ones


zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
zombies_df = pd.DataFrame({'Focal Name': zombies})

# spike_rates = spike_rate_analysis.compute_overall_average_spike_rates_for_each_round("2023-09-29", 2)
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
#
# lr = LinearRegression(fit_intercept=False)
# lr.fit(X, Y)
#
# print('coeff')
# print(lr.coef_)
# model = OLS(Y, X)
# results = model.fit()
# print(results.summary())


# Get individual spikes for individual stimuli
sp_rate_each = spike_rate_analysis.get_raw_spike_rates_for_each_stimulus("2023-09-29", 2)
sp_rate_zombies = sp_rate_each[zombies]
print(sp_rate_zombies)
