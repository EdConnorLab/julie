from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import spike_count
from sklearn.preprocessing import StandardScaler
from excel_data_reader import ExcelDataReader

"""
Regressors
    1. Regressor group 1: genealogy

    2. Regressor group 2: frequency of behaviors (affiliative, agonistic, submissive)
    standardize behavioral matrix (subtract mean and divide by standard deviation)

Response
    1. Entire trial (2 seconds)
    2. Time windowed (based on visual inspection)
    3. Set time window -- 300 ms to 1000 ms

"""

# Get genealogy matrix
excel_data_reader = ExcelDataReader(file_name='genealogy_matrix.xlsx')
genealogy_data = excel_data_reader.get_first_sheet()
genealogy_data['Focal Name'] = genealogy_data['Focal Name'].astype(str)  # To convert 7124 into string
X = genealogy_data  # no need to standardize the genealogy data

spikes = spike_count.get_spike_count_for_each_trial("2023-10-04", 1)
Y = spikes.iloc[0, :]

lasso = LassoCV(cv=10).fit(X, Y)
betas = lasso.coef_



# X_scaled = sm.add_constant(X_scaled)
# poisson_model = sm.GLM(Y, X_scaled, family=sm.families.Poisson()).fit()
# print(poisson_model.summary())

np.random.seed(42)
# Create a synthetic dataset
n = 100
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
y = np.random.poisson(lam=np.exp(0.5 * x1 + 0.3 * x2))
data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
print(data.head())


# Define the model formula
formula = 'y ~ x1 + x2'

# Fit the Poisson GLM
model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()

# Print the model summary
print(model.summary())
