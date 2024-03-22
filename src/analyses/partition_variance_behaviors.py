import numpy as np
import pandas as pd

from excel_data_reader import ExcelDataReader

excel_data_reader = ExcelDataReader(file_name='feature_df_submissive.xlsx')
beh = excel_data_reader.get_first_sheet()
beh = beh.iloc[:, 11:] # only extract the beh columns

n_entries = beh.shape[0] * beh.shape[1] - beh.shape[0]

Savg = beh.values.flatten().sum() / n_entries
m_arrow = beh.sum(axis=1) / (beh.shape[1] - 1) - Savg
arrow_m = beh.sum(axis=0) / (beh.shape[0] - 1)  - Savg

adjusted_columns = beh.sub(m_arrow, axis=0) # subtract a column
adjusted_rows = adjusted_columns.sub(arrow_m, axis=1) # subtract a row
adjusted_beh = adjusted_rows - Savg

numbers = adjusted_beh.values
np.fill_diagonal(numbers, 0)
beh_final = pd.DataFrame(numbers, columns=adjusted_rows.columns)
beh_final.to_excel('submissive_adjusted.xlsx', index=False)
print(beh_final)