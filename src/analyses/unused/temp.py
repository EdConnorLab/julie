from datahandler import InputDataManager
from recording_metadata_reader import RecordingMetadataReader
import pandas as pd

'''
metadata_reader = RecordingMetadataReader()
raw_metadata = metadata_reader.get_raw_data()
first_list = raw_metadata.parse("Cells_fromEd")
second_list = raw_metadata.parse("BestFrans_Cells")
unique_pairs_first_list = first_list[['Date', 'Round No.']].drop_duplicates()
unique_pairs_second_list = second_list[['Date', 'Round No.']].drop_duplicates()

combined_pairs = pd.concat([unique_pairs_first_list, unique_pairs_second_list], ignore_index=True)
combined_pairs.to_csv("/home/connorlab/Documents/GitHub/Julie/checkpoint/all_rounds_to_be_scanned.csv")
'''


data_manager = InputDataManager("/home/connorlab/Documents/IntanData/Cortana/2023-09-26/230926_round1_copy") # create input data manager with intan_file_directory
data_manager.read_data()


