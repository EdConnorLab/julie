import os
from pathlib import Path

import pandas as pd

current_dir = os.getcwd()
raw_data_file_name = 'Cortana_Recording_Metadata.xlsx'
file_path = Path(current_dir).parent.parent.parent / 'resources' / raw_data_file_name
xl = pd.ExcelFile(file_path)
sheet_names = xl.sheet_names
print(sheet_names)
channels = xl.parse('Channels')
locations = xl.parse('Location')