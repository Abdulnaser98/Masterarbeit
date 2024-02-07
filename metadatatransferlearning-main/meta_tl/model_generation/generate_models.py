import sys

sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/processes_data')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')

import pickle
import pandas as pd
import utils as utl
import config as conf

import xgboost as xgb

path_to_working_dir = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/'
path_to_data_folder = path_to_working_dir + 'data/'
path_to_models_dir = path_to_data_folder + 'models/'
path_to_sim_vector_folder =  path_to_data_folder + 'sim_dataframes/'

def generate_models(cluster_infos):
    # Remvoe already created models
    #utl.delete_files_from_folder(path_to_models_dir)
    cluster_max_row_file = {}
    # 1. Extract the file with the most number of rows
    for cluster,files in cluster_infos.items():
        print(f"We are now processing the cluster {cluster}")
        # Initialize variable to keep track of the file with the most rows
        max_rows = 0
        max_rows_file = ""

        # Iterate through the CSV files and find the one with the most rows
        for file in files:
            try:
                # Read the CSV_file
                df = pd.read_csv(path_to_sim_vector_folder + file)

                # Get the number of rows in the dataframe
                num_rows = len(df)

                # compare the number of rows with the current maximum
                if num_rows > max_rows:
                    max_rows = num_rows
                    max_rows_file = file


            except Exception as e:
                print(f"Error reading {file}: {e}")

        print("cluster")
        print(cluster)
        print("max_file")
        print(max_rows_file)

        # 2. Build an XGBoost model and train it on the data of the dataframe with the most number of rows
        sim_vector_file_with_highest_number = pd.read_csv(path_to_sim_vector_folder + max_rows_file)

        print("shape")
        print(sim_vector_file_with_highest_number.shape)
        print("count values is:")
        print(sim_vector_file_with_highest_number['is_match'].value_counts())
        # Drop the specified columns using the drop() function
        sim_vector_file_with_highest_number.drop(columns=['record_compared_1','record_compared_2','Modell_no_Liste_TruncateBegin20','Unnamed: 0','recId','recId.1'], axis=1, inplace=True)

        # Replace "/" with 9999 in 'is_match' column
        sim_vector_file_with_highest_number.replace('/', 9999, inplace=True)

        # Convert all columns to numerical data types
        sim_vector_file_with_highest_number = sim_vector_file_with_highest_number.apply(pd.to_numeric, errors='coerce')

        if conf.training_instance_based == True:
            sim_vector_file_with_highest_number['Produktname_dic3'] = sim_vector_file_with_highest_number['Produktname_dic3'].round(2)
            sim_vector_file_with_highest_number['Modell_Liste_3g'] = sim_vector_file_with_highest_number['Modell_Liste_3g'].round(2)
            columns_to_select = [col for col in sim_vector_file_with_highest_number.columns if col != 'is_match']
            sim_vector_file_with_highest_number['closest_match_ratio'] = sim_vector_file_with_highest_number.apply(lambda row: utl.calcualte_ratio(row,sim_vector_file_with_highest_number,columns_to_select), axis=1)
            sim_vector_file_with_highest_number = sim_vector_file_with_highest_number.loc[sim_vector_file_with_highest_number['closest_match_ratio'] > 0.5]
            sim_vector_file_with_highest_number.drop(columns=['closest_match_ratio'], axis=1, inplace=True)


        # Assuming the last column is the target variable and the rest are features
        X = sim_vector_file_with_highest_number.iloc[:, :-1] # Features (all columns except the last one)
        y = sim_vector_file_with_highest_number.iloc[: , -1] # Taregt variable (is_match)

        # Create an XGBoost classifier
        model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

        # Train the model in the taining data
        model.fit(X, y)

        # Specify the full path for saving the model
        #model_path = path_to_modles_folder + 'Cluster_' + str(cluster) + '.model'

        # Save the XGBoost model to the specified path
        #model.save_model(model_path)
        file_name = path_to_models_dir + "Cluster_" +str(cluster) + '.pkl'

        # save
        pickle.dump(model, open(file_name, "wb"))
        print(f"we saved the model {file_name}")
        cluster_max_row_file['Cluster_'+str(cluster)+'.pkl'] = max_rows_file

    return cluster_max_row_file

