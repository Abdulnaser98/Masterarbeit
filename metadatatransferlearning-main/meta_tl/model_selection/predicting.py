import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
import config as conf
import utils as utl




path_to_working_dir = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/'
path_to_data_folder = path_to_working_dir + 'data/'
path_to_models_dir = path_to_data_folder + 'models/'
path_to_sim_vector_folder =  path_to_data_folder + 'sim_dataframes/'
active_learning_path = path_to_data_folder + 'active_learning/'
path_to_results_folder = path_to_data_folder + 'results/'

meta_data_rep_mapping = {
    1:'generate_data_representation_initial_features',
    2:'generate_data_representation_better_features',
    3:'generate_data_representations_combined_features',
    4:'generate_data_representations_mean_embeddings'
}

def models_training(cluster_strings,cluster_max_row_file):
    # Delete all files in the Active Learning
    utl.delete_files_from_folder(active_learning_path)

    # List all files in the directory
    model_files = [f for f in os.listdir(path_to_models_dir) if f.endswith('.pkl')]

    # Load each XGBoost model
    loaded_models = {}
    sim_vec_file_to_process = []
    trained_models = []
    performance = []

    for model_file in model_files:
        # Construct the full path for the model
        model_path = os.path.join(path_to_models_dir, model_file)
        # Load the XGBoost model
        xgb_model_loaded = pickle.load(open(model_path,"rb"))

        # Append the loaded model to the list associated with the model name
        if model_file not in loaded_models:
            loaded_models[model_file] = []
        loaded_models[model_file].append(xgb_model_loaded)


    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(path_to_sim_vector_folder) if file.endswith('.csv')]


    for file_name in csv_files:
       #print("The file to process is {}".format(file_name))
       # find the first key associated with the specified value
       first_matching_key = [key for key,value in cluster_strings.items() if file_name in value]
       sim_cluster_name = ('Cluster_'+str(first_matching_key[0])+'.pkl').replace('[','').replace(']','')
       #print("The cluster name is {}".format(sim_cluster_name))
       full_path = os.path.join(path_to_sim_vector_folder, file_name)
       sim_vec_file = pd.read_csv(full_path)
       sim_vec_file.drop(columns=['record_compared_1','record_compared_2','Modell_no_Liste_TruncateBegin20','Unnamed: 0','recId','recId.1'], axis=1, inplace=True)
       # Replace "/" with 9999 in 'is_match' column
       sim_vec_file.replace('/', 9999, inplace=True)
       # Convert all columns to numerical data types
       sim_vec_file = sim_vec_file.apply(pd.to_numeric, errors='coerce')
       X_test = sim_vec_file.iloc[:, :-1] # Features (all columns except the last one)
       y_test = sim_vec_file.iloc[: , -1] # Taregt variable (is_match)


       # Print cluster number and strings belonging to each other
       for model_name, model in loaded_models.items():
           print("model name is")
           print(model_name)
           print("sim_cluster_name is")
           print(sim_cluster_name)
           if model_name == sim_cluster_name:
              sim_vec_file_to_process.append(file_name)
              trained_models.append("Original cluster model " + cluster_max_row_file.get(model_name))
              predictions = model[0].predict(X_test)
              class_probs = model[0].predict_proba(X_test)
              sim_vec_file['pred'] = predictions
              sim_vec_file[['probabilties_0', 'probabilties_1']] = class_probs
              #accuracy = accuracy_score(y_test, predictions)
              accuracy = f1_score(y_test, predictions)
              performance.append(accuracy)
              sim_vec_file.to_csv(active_learning_path + file_name + model_name + ".csv")

           #else:
           #  predictions = model[0].predict(X_test)
           #   class_probs = model[0].predict_proba(X_test)
           #   sim_vec_file['pred'] = predictions
           #   sim_vec_file[['probabilties_0', 'probabilties_1']] = class_probs
           #   accuracy = accuracy_score(y_test, predictions)
           #   sim_vec_file_to_process.append(file_name)
           #   trained_models.append(cluster_max_row_file.get(model_name))
           #   performance.append(accuracy)





    # Create a DataFrame
    data = {'sim_vec_file':sim_vec_file_to_process, 'model':trained_models, 'performance': performance}

    df = pd.DataFrame(data)
    print(df.head())

    file_name = ""

    if conf.data_duplicated:
        file_name = "with_duplicated_data_"
    else:
        file_name = "without_duplicated_data_"

    #file_name = file_name + meta_data_rep_mapping[conf.meta_data_representation] + "_"

    if conf.training_instance_based:
        file_name = file_name + 'instance_based.csv'
    else:
        file_name = file_name + 'not_instance_based.csv'

    df.to_csv(os.path.join(path_to_results_folder + file_name))

    return file_name



