import os
from pathlib import Path

import pandas as pd


class ExcelDataReader:

    def __init__(self, file_name=None):
        self.file_name = file_name
        self.resource_dir_path = '/home/connorlab/Documents/GitHub/Julie/resources/'
        if self.file_name is not None:
            self.file_path = Path(os.path.join(self.resource_dir_path, self.file_name))
        else:
            self.file_path = None

        self.xl = pd.ExcelFile(self.file_path) if self.file_path is not None else None

    def get_raw_data(self):
        return self.xl

    def get_sheet_by_name(self, sheet_name):
        return self.xl.parse(sheet_name)

    def get_first_sheet(self):
        sheet_names = self.xl.sheet_names
        return self.xl.parse(sheet_names[0])

    def get_last_sheet(self):
        sheet_names = self.xl.sheet_names
        return self.xl.parse(sheet_names[-1])