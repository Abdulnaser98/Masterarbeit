import os
import sys
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler



sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
import config as conf

sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/processes_data')
import utils as utl

path_to_working_dir = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/'
path_to_data_folder = path_to_working_dir + 'data/'
path_to_feature_vectors_folder = path_to_data_folder + 'feature_vectors/'
path_to_sim_vector_folder =  path_to_data_folder + 'sim_dataframes/'


def prepare_dataframe(csv_file,to_consider_columns):
    file_path = os.path.join(path_to_sim_vector_folder, csv_file)
    df = pd.read_csv(file_path)
    df = df[to_consider_columns]
    # Convert "/" into Nan
    df = df.apply(pd.to_numeric,errors= 'coerce')
    return df


def generate_data_representation_initial_features(csv_files,nan_replacement_value,to_consider_columns):
    utl.delete_files_from_folder(path_to_feature_vectors_folder,files_to_del='feature_vectors_init_features.csv')
    # Create an empty list to store feature vectors
    feature_vectors_list = []
    # Iterate through each CSV file and generate statistics
    for csv_file in csv_files:
        df = prepare_dataframe(csv_file,to_consider_columns)
        # Calculate statistics
        # The number of columns that have at least one non-Nan Value
        non_nan_columns_count = np.sum(df.notnull().any())
        # The percentage of the Nan values in each column
        nan_percentage = ((df.isnull().sum() / len(df)) * 100).round(2)
        column_min = df.min().round(2)
        column_mean = df.mean().round(2)
        column_max = df.max().round(2)
        column_median = df.median().round(2)


        # Create a Series for the names of the sim-vectors
        file_name_series = pd.Series([csv_file], index=['file_name'])

        # Create a Series for non_nan_columns_count
        non_nan_columns_count_series = pd.Series([non_nan_columns_count], index=['non_nan_columns_count'])

        # Concatenate statistics into a feature vector
        feature_vector = pd.concat([file_name_series,non_nan_columns_count_series,nan_percentage,column_min, column_mean, column_max, column_median], axis=0)

        # Replace NaN values in the feature vector with a specific numerical value
        feature_vector.fillna(nan_replacement_value, inplace=True)

        # Convert the feature vector to a list and append it to the feature vectors list
        feature_vectors_list.append(feature_vector.tolist())

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(feature_vectors_list)
    # Assign names to columns
    column_names = [f"feature_{i+1}" for i in range(df.shape[1])]
    df.columns = column_names
    df.to_csv(os.path.join(path_to_feature_vectors_folder,'feature_vectors_init_features.csv'))

    return df,df['feature_1']



def generate_data_representation_better_features(csv_files,nan_replacement_value,to_consider_columns):
    utl.delete_files_from_folder(path_to_feature_vectors_folder,files_to_del='feature_vectors_better_features.csv')
    # Create an empty list to store feature vectors
    feature_vectors_list = []
    # Iterate through each CSV file and generate statistics
    for csv_file in csv_files:
        df = prepare_dataframe(csv_file,to_consider_columns)
        # Calculate statistics
        # The number of columns that have at least one non-Nan Value
        #non_nan_columns_count = np.sum(df.notnull().any())
        # The percentage of the Nan values in each column
        nan_percentage = ((df.isnull().sum() / len(df)) * 100).round(2)
        column_std_deviation = df.std().round(2)
        column_median = df.median().round(2)
        column_skewness = df.skew().round(2)
        column_kurtosis = df.kurt().round(2)


        # Create a Series for the names of the sim-vectors
        file_name_series = pd.Series([csv_file], index=['file_name'])

        # Create a Series for non_nan_columns_count
        #non_nan_columns_count_series = pd.Series([non_nan_columns_count], index=['non_nan_columns_count'])

        # Concatenate statistics into a feature vector
        feature_vector = pd.concat([file_name_series,nan_percentage,column_std_deviation, column_median, column_skewness, column_kurtosis], axis=0)

        # Replace NaN values in the feature vector with a specific numerical value
        feature_vector.fillna(nan_replacement_value, inplace=True)

        # Convert the feature vector to a list and append it to the feature vectors list
        feature_vectors_list.append(feature_vector.tolist())


    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(feature_vectors_list)
    # Assign names to columns
    column_names = [f"feature_{i+1}" for i in range(df.shape[1])]
    df.columns = column_names
    df.to_csv(os.path.join(path_to_feature_vectors_folder, 'feature_vectors_better_features.csv'))

    return df,df['feature_1']


def generate_data_representations_combined_features(csv_files,nan_replacement_value,to_consider_columns):
    utl.delete_files_from_folder(path_to_feature_vectors_folder,files_to_del='feature_vectors_combined_features.csv')
    # Create an empty list to store feature vectors
    feature_vectors_list = []
    # Iterate through each CSV file and generate statistics
    for csv_file in csv_files:
        df = prepare_dataframe(csv_file,to_consider_columns)
        # Calculate statistics
        # The number of columns that have at least one non-Nan Value
        non_nan_columns_count = np.sum(df.notnull().any())
        # The percentage of the Nan values in each column
        nan_percentage = ((df.isnull().sum() / len(df)) * 100).round(2)
        column_min = df.min().round(2)
        column_mean = df.mean().round(2)
        column_max = df.max().round(2)
        column_median = df.median().round(2)

        column_std_deviation = df.std().round(2)
        column_skewness = df.skew().round(2)
        column_kurtosis = df.kurt().round(2)

        # Create a Series for the names of the sim-vectors
        file_name_series = pd.Series([csv_file], index=['file_name'])

        # Create a Series for non_nan_columns_count
        non_nan_columns_count_series = pd.Series([non_nan_columns_count], index=['non_nan_columns_count'])

        # Concatenate statistics into a feature vector
        feature_vector = pd.concat([file_name_series,non_nan_columns_count_series,nan_percentage,column_min, column_mean, column_max, column_median], axis=0)

        # Replace NaN values in the feature vector with a specific numerical value
        feature_vector.fillna(nan_replacement_value, inplace=True)

        # Convert the feature vector to a list and append it to the feature vectors list
        feature_vectors_list.append(feature_vector.tolist())

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(feature_vectors_list)

    # Assign names to columns
    column_names = [f"feature_{i+1}" for i in range(df.shape[1])]
    df.columns = column_names
    df.to_csv(os.path.join(path_to_feature_vectors_folder,'feature_vectors_combined_features.csv'))

    return df,df['feature_1']


def generate_data_representations_mean_embeddings(csv_files,nan_replacement_value,to_consider_columns):
    utl.delete_files_from_folder(path_to_feature_vectors_folder,files_to_del='feature_vectors_mean_embeddings_features.csv')
    # Create an empty list to store feature vectors
    feature_vectors_list = []
    # Iterate through each CSV file and generate statistics
    for csv_file in csv_files:
        df = prepare_dataframe(csv_file,to_consider_columns)
        # Replace NaN values in the feature vector with a specific numerical value
        df.fillna(nan_replacement_value, inplace=True)
        df = df.values
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        df_rbf = rbf_feature.fit_transform(df)
        # Calculate MMD
        mmd = df_rbf.mean(axis=0)
        # Convert the mmd np-array into a list
        mmd = mmd.tolist()
        # Insert the compared_resource name at the first index of the list
        mmd.insert(0, csv_file)
        # Convert the feature vector to a list and append it to the feature vectors list
        feature_vectors_list.append(mmd)


    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(feature_vectors_list)
    # Assign names to columns
    column_names = [f"feature_{i+1}" for i in range(101)]
    df.columns = column_names
    df.to_csv(os.path.join(path_to_feature_vectors_folder,'feature_vectors_mean_embeddings_features.csv'))
    return df,df['feature_1']





def generate_meta_data_representations():
    print("We are now in the generate_meta_data_representations")
    # Get a list of all CSV files in sim_vector_folder the folder
    csv_files = [file for file in os.listdir(path_to_sim_vector_folder) if file.endswith('.csv')]

    # Value to exclude from feature extraction
    nan_replacement_value = 99999

    # Columns to extracte from feature extraction
    to_consider_columns = ['MPN_Liste_TruncateBegin20','EAN_Liste_TruncateBegin20','Produktname_dic3',
                           'Modell_Liste_3g','Digital_zoom_NumMaxProz30','optischer_zoom_NumMaxProz30', 'Breite_NumMaxProz30',
                           'Höhe_NumMaxProz30', 'Gewicht_NumMaxProz30', 'Sensortyp_Jaccard3']

    print(f"the value of the conf.meta_data_representation is {conf.meta_data_representation}")

    if conf.meta_data_representation == 1:
         return generate_data_representation_initial_features(csv_files,nan_replacement_value,to_consider_columns)
    elif conf.meta_data_representation == 2:
         return generate_data_representation_better_features(csv_files,nan_replacement_value,to_consider_columns)
    elif conf.meta_data_representation == 3:
         return generate_data_representations_combined_features(csv_files,nan_replacement_value,to_consider_columns)
    elif conf.meta_data_representation == 4:
         print("we are now calling the function generate_data_representations_mean_embeddings()")
         return generate_data_representations_mean_embeddings(csv_files,nan_replacement_value,to_consider_columns)
