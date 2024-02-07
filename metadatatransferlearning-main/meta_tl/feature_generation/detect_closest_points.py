import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_kernels



n_components = 100  # Number of components in the approximation
gamma = 1.0  # RBF kernel parameter




def detect_closest_points():
    # specify the path of the data folder
    path_to_data_folder = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/data/'

    # Specify the folder path containing the csv files
    path_to_sim_vector_folder =  path_to_data_folder + 'sim_dataframes/'

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(path_to_sim_vector_folder) if file.endswith('.csv')]

    # Columns to extracte from feature extraction
    to_consider_columns = [ 'MPN_Liste_TruncateBegin20','EAN_Liste_TruncateBegin20','Produktname_dic3',
                            'Modell_Liste_3g','Digital_zoom_NumMaxProz30','optischer_zoom_NumMaxProz30', 'Breite_NumMaxProz30',
                            'Höhe_NumMaxProz30', 'Gewicht_NumMaxProz30', 'Sensortyp_Jaccard3']


    first_file = []
    second_file = []
    mmd_values = []

    for csv_file_1 in tqdm(csv_files, desc="Processing items"):
        file_path_1 = os.path.join(path_to_sim_vector_folder, csv_file_1)
        file_df_1 = pd.read_csv(file_path_1)
        file_df_1 = file_df_1[to_consider_columns]
        file_df_1 = file_df_1.apply(pd.to_numeric, errors = 'coerce')
        file_df_1.fillna(2,inplace=True)

        for csv_file_2 in csv_files:
            file_path_2 = os.path.join(path_to_sim_vector_folder, csv_file_2)
            if file_path_1 != file_path_2:
                first_file.append(csv_file_1)
                second_file.append(csv_file_2)
                file_df_2 = pd.read_csv(file_path_2)
                file_df_2 = file_df_2[to_consider_columns]
                file_df_2 = file_df_2.apply(pd.to_numeric, errors = 'coerce')
                file_df_2.fillna(2,inplace=True)

                rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)

                # Transform the datasets using the RBFSampler
                X_rbf_dist1 = rbf_sampler.fit_transform(file_df_1)
                X_rbf_dist2 = rbf_sampler.transform(file_df_2)

                # Compute the MMD using the RBF kernel
                kernel_matrix_dist1 = pairwise_kernels(X_rbf_dist1, metric="rbf", gamma=gamma)
                kernel_matrix_dist2 = pairwise_kernels(X_rbf_dist2, metric="rbf", gamma=gamma)

                mmd = np.mean(kernel_matrix_dist1) + np.mean(kernel_matrix_dist2) - 2 * np.mean(pairwise_kernels(X_rbf_dist1, X_rbf_dist2, metric="rbf", gamma=gamma))
                mmd_values.append(mmd)



    df = pd.DataFrame({'first_file': first_file, 'second_file': second_file, 'mmd_values':mmd_values})

    # Find the index associated with the minimum value in the third column
    min_indices = df.groupby('first_file')['mmd_values'].idxmin()


    # Retrieve the corresponding values from the second column "second_file"
    result = df.loc[min_indices , ['first_file', 'second_file','mmd_values']]


    return result


