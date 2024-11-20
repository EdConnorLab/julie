import pickle
import pandas as pd

if __name__ == "__main__":
    experiment_data_filename = "230926_round1.pkl"
    experiment_name = experiment_data_filename.split(".")[0]
    file_path = "/home/r2_allen/git/EStimShape/EStimShapeAnalysis/compiled/%s" % experiment_data_filename
    raw_data = pd.read_pickle(file_path)
    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(raw_data)
    zombies = df[df['MonkeyGroup'] == 'Zombies']
    zombies = zombies.drop(['SpikeTimes', 'MonkeyId'], axis = 1)
    sorted_df = zombies.sort_values(by=['MonkeyName','TaskField'])
    print(sorted_df)
    sorted_df.to_csv("/home/r2_allen/git/EStimShape/EStimShapeAnalysis/compiled/%s.csv" % experiment_name, index=False)