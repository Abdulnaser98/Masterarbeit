import os
import pandas as pd
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
import config as conf
import utils as utl

path_to_working_dir = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/'
path_to_data_folder = path_to_working_dir + 'data/'
path_to_active_learning_folder = path_to_data_folder + 'active_learning/'
path_to_results_folder = path_to_results_folder = path_to_data_folder + 'results/'



def apply_active_learning(file_name):
    sim_vec_file_to_process = []
    preformance = []

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(path_to_active_learning_folder) if file.endswith('.csv')]

    for csv_file_name in csv_files:
        csv_file_path = os.path.join(path_to_active_learning_folder, csv_file_name)
        df = pd.read_csv(csv_file_path)
        threshold = 0.8
        # Calculate the absoulte difference between 'probabilties_0' and 'probabilties_1'
        df['AbsDiff'] = abs(df['probabilties_0'] - df['probabilties_1'])
        df['signifikant_diff'] = (df['AbsDiff'] < threshold).astype(int)
        # Change the value of 'pred' based on conditions
        mask = (df['signifikant_diff'] == 1) & (df['is_match'] != df['pred'])
        df.loc[mask, 'pred'] = df['is_match']
        accuracy = accuracy_score(df['is_match'], df['pred'])
        sim_vec_file_to_process.append(csv_file_name)
        preformance.append(accuracy)

    # Create a DataFrame
    data = {'sim_vec_file': sim_vec_file_to_process, 'performance': preformance}

    df_active_learning = pd.DataFrame(data)

    df_active_learning.to_csv(os.path.join(path_to_results_folder,(file_name + 'active_learning.csv')))
    utl.delete_files_from_folder(path_to_active_learning_folder)