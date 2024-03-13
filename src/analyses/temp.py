import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from social_data_reader import read_social_data_and_validate
from spike_rate_analysis import get_raw_spike_rates_for_each_stimulus

df = get_raw_spike_rates_for_each_stimulus("12-11-2023", 1)
feat = pd.read_csv('genealogy_features.csv', index_col=0)

for i_channel in range(len(df)):
    X, y = [], []

    channel_data = df.iloc[i_channel, :]
# for m in channel_data.keys():
#     data.extend(channel_data[m])
#     monkey.extend(np.tile(m, (10, 1)))

    for m in channel_data.keys():
        if m in feat['Focal Name'].unique():
            y.append(np.mean(channel_data[m]))
            X.append(feat.loc[feat['Focal Name'] == m, :].values[0][1:])
    X, y = np.array(X), np.array(y)
    X = X.astype(np.float32)
    y = y.reshape(-1, 1)

    model = OLS(y, X[:, :9])
    results = model.fit()
    print(results.summary())
