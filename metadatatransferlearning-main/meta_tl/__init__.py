# import libraries
import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/processes_data')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/feature_generation/')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/clustering/')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/model_generation/')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/model_selection/')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/record_linkage/')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/active_learning/')


import os
import utils as utl
import data_cleaning as dc
import detect_closest_points as dcp
import k_means_clus as km_clus
import mmd_values_points_clustering as mmd_clustering
import generate_models as models_generator
import generate_feature_vectors as feature_gen
import predicting as pred
import record_linkage_main as rec_linkage_main
import generate_gold_truth_data as gt_data_extractor
import config as conf
import pandas as pd
import active_learning as active_learning



# Pathes to the data sources
path_to_working_dir = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/'
path_to_data_folder = path_to_working_dir + 'data/'
path_to_sim_vector_folder =  path_to_data_folder + 'sim_dataframes/'
models_directory = path_to_working_dir + 'models/'
path_to_results_folder = path_to_data_folder + 'results/'
path_to_feature_vectors_folder = path_to_data_folder + 'feature_vectors/'
active_learning_path = path_to_data_folder + 'active_learning/'

"""
for data_duplicated in [True, False]:
    for training_instance_based in [True,False]:
        conf.data_duplicated = data_duplicated
        conf.training_instance_based = training_instance_based
        #=============================================== Data cleaning and preparation ==================================================================
        #1) Clean the data directory
        utl.delete_files_from_folder(path_to_data_folder + 'cleaned_data/')

        #2) Prepare and clean the data and save it in the cleaned_data directory
        dc.prepare_and_clean_data(to_duplicate=False)

        #=============================================== Record Linkage =================================================================================
        #3) Generate record linkage sim vectors and save them in the sim_vector_folder
        rec_linkage_main.record_linkage_main()
        #=============================================== Generate Ground truth data =====================================================================
        #4) Extract the ground truth data and add them as new column "is_match" in the sim vectors
        gt_data_extractor.ground_truth_data_extractor()
        #=============================================== Detect the closest points =======================================================================
        #5) Detect for each data point the point that has the most similair distribution to it
        closest_points = dcp.detect_closest_points()
        #=============================================== Clustering ======================================================================================
        #6) Cluster the points according to their distance from eahc other (Transivity and merge)
        clusters_elements = mmd_clustering.mmd_points_clustering(closest_points)
        #=============================================== Models generation and training ==================================================================
        #7) Models generator
        cluster_max_row_file = models_generator.generate_models(clusters_elements)
        # ============================================== Transfer Learning ===============================================================================
        #8) Transfer Learning on other datasets
        file_name = pred.models_training(clusters_elements,cluster_max_row_file)
        # ============================================== Active Learning  ===============================================================================
        active_learning.apply_active_learning(file_name)


"""


config_typ = []
min = []
median = []
mean = []
max = []
csv_files = [file for file in os.listdir(path_to_results_folder) if file.endswith('.csv')]
for filename in csv_files:
    file_path = os.path.join(path_to_results_folder, filename)
    df = pd.read_csv(file_path)
    config_typ.append(filename)
    min.append(df['performance'].min())
    median.append(df['performance'].median())
    mean.append(df['performance'].mean())
    max.append(df['performance'].max())
# Create a dataframe
data = {'Config_typ': config_typ, 'min':min, 'median':median,
            'mean':mean, 'max': max}

df = pd.DataFrame(data)

# Sort the DataFrame based on the 'mean' Column
sorted_df = df.sort_values(by='mean',ascending=False)

sorted_df.to_csv(os.path.join(path_to_results_folder,'main_results.csv'))






