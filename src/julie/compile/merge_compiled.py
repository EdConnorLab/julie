import os
import pickle

import pandas as pd


def main():
    round_one = ["1696954960635002_231010_122242_round1_1.pk1",
                 "1696957097095212_231010_125817_round1_2.pk1",
                 ]
    round_two = ["1696888890257195_231009_180130_round2_1.pk1",
                 "1696891705894032_231009_184826_round2_2.pk1"]
    round_three = ["1700934337327831_231125_124537_round3_1.pk1",
                   "1700936534141338_231125_132214_round3_2.pk1"]
    round_four = [".pk1",
                  "1698436475645557_231027_155436.pk1"]
    round_five = ["1695326404335201_230921_160004.pk1"]

    experiment_data_filenames = round_three

    experiment_names = [experiment_data_filename.split(".")[0] for experiment_data_filename in
                        experiment_data_filenames]

    file_paths = ["/home/connorlab/Documents/GitHub/Julie/compiled/%s" % experiment_data_filename for
                  experiment_data_filename in experiment_data_filenames]

    data = add_pickled_dataframes(file_paths)
    print("Combined Dataframe number of trials:", len(data))
    combined_filename = "&".join(experiment_names) + ".pk1"
    save_dir = "/home/connorlab/Documents/GitHub/Julie/compiled/"
    save_path = os.path.join(save_dir, combined_filename)
    data.to_pickle(save_path)


def add_pickled_dataframes(paths):
    result = None

    for path in paths:
        with open(path, 'rb') as file:
            data = pickle.load(file)
            if result is None:
                result = data
            else:
                # Concatenate the DataFrames vertically (later ones below earlier ones)
                result = pd.concat([result, data], axis=0, ignore_index=True)

    return result


if __name__ == '__main__':
    main()
