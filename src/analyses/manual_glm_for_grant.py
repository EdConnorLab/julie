import pandas as pd
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, Lasso
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from excel_data_reader import ExcelDataReader
from initial_4feature_lin_reg import construct_feature_matrix_from_behavior_data, \
    get_metadata_for_ANOVA_passed_cells_time_windowed, compute_average_spike_rates_for_list_of_cells_with_time_windows
from monkey_names import Zombies
import social_data_processor

'''
Perform LassoCV (since there are only 9 data points, rather than doing Lasso) 
and pick out the non-zero coefficients and then use these for glm fit

Date Created: 07-02-2024
Last Modified: 07-02-2024

'''
# Get all dates and rounds from metadata
time_windowed_cells = get_metadata_for_ANOVA_passed_cells_time_windowed()

# Spike Rate
spike_rates = compute_average_spike_rates_for_list_of_cells_with_time_windows(time_windowed_cells)
zombies = [member.value for name, member in Zombies.__members__.items()]
zombies_columns = [col for col in zombies if col in spike_rates.columns]
zombies_spike_rates = spike_rates[zombies_columns]

# Get behavioral data
agon_beh_df = ExcelDataReader(file_name="feature_df_agonism.xlsx").get_first_sheet().iloc[:, 1:]
sub_beh_df = ExcelDataReader(file_name="feature_df_submission.xlsx").get_first_sheet().iloc[:, 1:]
aff_beh_df = ExcelDataReader(file_name="feature_df_affiliation.xlsx").get_first_sheet().iloc[:, 1:]
agon_beh = agon_beh_df.to_numpy()
sub_beh = sub_beh_df.to_numpy()
aff_beh = aff_beh_df.to_numpy()


# get general behaviors separately
X0 = np.column_stack([aff_beh[:, 3], aff_beh[3, :].T, agon_beh[7, :].T])
X1 = np.column_stack([sub_beh[:, 0], sub_beh[:, 3], sub_beh[5, :].T, agon_beh[:, 5], agon_beh[:, 9]])
X2 = np.column_stack([aff_beh[:, 8], aff_beh[8, :].T, sub_beh[8, :].T, agon_beh[:, 8], agon_beh[8, :].T])
X3 = np.column_stack([aff_beh[:, 4], aff_beh[4, :].T, sub_beh[:, 5], sub_beh[:, 8], agon_beh[2, :].T, agon_beh[3, :].T])
X4 = np.column_stack([aff_beh[:, 2], aff_beh[:, 6], aff_beh[2, :].T, aff_beh[6, :].T, agon_beh[5, :].T])
X5 = np.column_stack([aff_beh[:, 2], aff_beh[:, 6], aff_beh[2, :].T, aff_beh[6, :].T, agon_beh[5, :].T])
X6 = np.column_stack([aff_beh[:, 6], aff_beh[6, :].T, agon_beh[5, :].T])
X7 = np.column_stack([sub_beh[0, :].T, sub_beh[1, :].T, sub_beh[2, :].T, sub_beh[3, :].T, sub_beh[6, :].T, sub_beh[7, :].T, sub_beh[8, :].T, agon_beh[:, 2]])
X8 = np.column_stack([agon_beh[2, :].T, agon_beh[5, :].T])
X9 = np.column_stack([aff_beh[:, 0], aff_beh[:, 5], aff_beh[0, :].T, aff_beh[5, :].T])
X10 = np.column_stack([aff_beh[5, :].T])
X12 = np.column_stack([aff_beh[:, 2], aff_beh[:, 6], aff_beh[2, :].T, aff_beh[6, :].T])
X13 = np.column_stack([aff_beh[:, 8], aff_beh[2, :].T, aff_beh[8, :].T, sub_beh[:, 1], agon_beh[0, :].T, agon_beh[1, :].T, agon_beh[4, :].T, agon_beh[8, :].T])
X14 = np.column_stack([aff_beh[:, 7], aff_beh[7, :].T, sub_beh[1, :].T])
X15 = np.column_stack([aff_beh[:, 4], aff_beh[4, :].T, sub_beh[:, 3], sub_beh[:, 5], sub_beh[:, 7], sub_beh[:, 8], sub_beh[:, 9], sub_beh[5, :].T, agon_beh[2, :].T, agon_beh[3, :].T])
X16 = np.column_stack([aff_beh[:, 3], aff_beh[3, :].T, sub_beh[:, 3], sub_beh[:, 6], sub_beh[:, 8], sub_beh[:, 9], agon_beh[3, :].T, agon_beh[6, :].T])
X17 = np.column_stack([sub_beh[1, :].T, sub_beh[2, :].T, sub_beh[3, :].T, sub_beh[6, :].T, sub_beh[7, :].T, agon_beh[:, 2], agon_beh[:, 6]])
X18 = np.column_stack([aff_beh[9, :].T, sub_beh[1, :].T, sub_beh[6, :].T, sub_beh[8, :].T, agon_beh[:, 8]])
X19 = np.column_stack([sub_beh[:, 9]])
X21 = np.column_stack([aff_beh[:, 0], aff_beh[:, 5], aff_beh[:, 9], aff_beh[0, :].T, aff_beh[9, :].T, sub_beh[5, :].T, agon_beh[9, :].T])
X22 = np.column_stack([aff_beh[:, 0], aff_beh[:, 5], aff_beh[:, 9], aff_beh[0, :].T, aff_beh[5, :].T, aff_beh[9, :].T, sub_beh[5, :].T, agon_beh[9, :].T])
X23 = np.column_stack([aff_beh[:, 0], aff_beh[:, 4], aff_beh[4, :].T, aff_beh[5, :].T, aff_beh[9, :].T, sub_beh[:, 5], sub_beh[:, 7], sub_beh[:, 8], sub_beh[:, 9], sub_beh[5, :].T, agon_beh[2, :].T, agon_beh[3, :].T])

X_dict = {
    "X0": X0, "X1": X1,  "X2": X2,  "X3": X3, "X4": X4, "X5": X5,  "X6": X6, "X7": X7, "X8": X8, "X9": X9, "X10": X10,
    "X12": X12, "X13": X13, "X14": X14, "X15": X15, "X16": X16, "X17": X17, "X18": X18, "X19": X19, "X21": X21,
    "X22": X22, "X23": X23
    }

# X = StandardScaler().fit_transform(X)
# Get all dates and rounds from metadata
# first_set_of_time_windowed_cells = get_metadata_for_ANOVA_passed_cells_time_windowed()

# Names of features assuming you have them as a list or from a DataFrame
# feature_names = ['General Agonism', 'General Attraction to Agonism', 'General Submission',
#     'General Attraction to Submission', 'General Affiliation', 'General Attraction to Affiliation']

all_results = {}
for key, X in X_dict.items():
    # Extract the row index from the key (assuming the key format is "X<row_index>")
    row_index = int(key[1:])
    X = np.delete(X, 6, axis=0)
    X = StandardScaler().fit_transform(X)
    # Extract the corresponding y values from zombies_spike_rates
    y = zombies_spike_rates.iloc[row_index, :].values.flatten()

    # Apply LassoCV
    lasso_cv = LassoCV(cv=5, max_iter=10000)
    lasso_cv.fit(X, y)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)

    # Plot the MSE vs. Alpha
    plt.plot(lasso_cv.alphas_, mean_mse, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE vs. Alpha in LassoCV for {key}')
    plt.xscale('log')
    plt.show()

    # Apply Lasso with the best alpha
    lasso = Lasso(alpha=lasso_cv.alpha_)
    lasso.fit(X, y)
    non_zero_mask = lasso.coef_ != 0

    num_of_non_zero_coeff = sum(non_zero_mask)

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(f"---------------{num_of_non_zero_coeff} non-zero coefficients for {key}--------------------")
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!!!!!!!!!!!!!!!!!!! results for {key}")
    print(np.where(non_zero_mask)[0])
    # Select the non-zero coefficients
    X_selected = X[:, non_zero_mask]
    X_adj = sm.add_constant(X_selected)
    X_aug = sm.add_constant(X)
    # Fit the GLM model
    model = sm.GLM(y, X_adj, family=sm.families.Gaussian())
    # model = sm.GLM(y, X_aug, family=sm.families.Gaussian())
    results = model.fit()
    pvalues = np.round(results.pvalues, 4)
    params = np.round(results.params, 4)
    aic = np.round(results.aic, 4)
    bic = np.round(results.bic, 4)
    stat_params_to_be_saved = [aic, bic, pvalues, params]
    all_results[key] = stat_params_to_be_saved
    # Print the summary of results
    print(results.summary())
    # print(results.summary2())
#
all_results_df = pd.DataFrame(all_results)
all_results_df.T.to_csv('lasso_glm_all_results.csv')
'''
original code 

for index, row in zombies_spike_rates.iterrows():
    y = row.values.flatten()
    lasso_cv = LassoCV(cv=5, max_iter=9000, tol=0.001)
    lasso_cv.fit(X, y)
    # print(lasso_cv.coef_)
    # print(lasso_cv.intercept_)
    # print(lasso_cv.alpha_)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    plt.plot(lasso_cv.alphas_, mean_mse, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Alpha in LassoCV')
    plt.xscale('log')  # Since alphas can vary on a large scale
    plt.show()
    lasso = Lasso(alpha=lasso_cv.alpha_)
    lasso.fit(X, y)
    # Get the mask of non-zero coefficients
    non_zero_mask = lasso.coef_ != 0
    num_of_non_zero_coeff = sum(non_zero_mask)
    print("----------------------------------------------------------------------------------------------")
    print(f"-------------------------------{num_of_non_zero_coeff}---------------------------------------")
    print("----------------------------------------------------------------------------------------------")
    # important_features = [name for name, m in zip(general_feature_names, non_zero_mask) if m]
    # print("Important features:", important_features)
    X_selected = X[:, non_zero_mask]
    X_adj = sm.add_constant(X_selected)
    model = sm.GLM(y, X_adj, family=sm.families.Gaussian())
    results = model.fit()
    print(results.summary())
'''