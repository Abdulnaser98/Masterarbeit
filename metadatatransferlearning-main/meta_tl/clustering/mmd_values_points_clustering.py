import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
import config as conf
from utils import *
from tqdm import tqdm


def mmd_points_clustering(mmd_values):

    cluster_lists = []
    print(mmd_values.head())
    # Iterate over the rows using iterrows()
    for index, row in tqdm(mmd_values.iterrows(), desc="Processing items"):
        if is_element_in_dict(cluster_lists, row['first_file'])[0] == True:
            continue
        else:
            elm_list = []
            elm_list.append(row['first_file'])
            closest_elm = mmd_values.loc[mmd_values['first_file'] == row['first_file'],'second_file'].values[0]
            while((closest_elm not in elm_list) & (is_element_in_dict(cluster_lists,closest_elm)[0]==False)):
                    elm_list.append(closest_elm)
                    closest_elm = mmd_values.loc[mmd_values['first_file']== closest_elm , 'second_file'].values[0]

            if closest_elm in elm_list:
                cluster_lists.append(elm_list)

            elif(is_element_in_dict(cluster_lists,closest_elm)[0]==True):
                    new_set = list(set(is_element_in_dict(cluster_lists,closest_elm)[1]).union(elm_list))
                    cluster_lists.append(new_set)
                    cluster_lists.remove(is_element_in_dict(cluster_lists,closest_elm)[1])
    result_dict = {f'Cluster_{index}':sublist for index, sublist in enumerate(cluster_lists)}
    print(f"The result dict is {result_dict} ")

    return result_dict