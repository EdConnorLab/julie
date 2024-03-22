import numpy as np
import pandas as pd

from excel_data_reader import ExcelDataReader

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
print(beh_final)
# beh_final.to_excel('submissive_adjusted.xlsx', index=False)
