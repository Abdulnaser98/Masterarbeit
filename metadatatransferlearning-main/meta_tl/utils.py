import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances



# Function to convert mixed units to pounds and remove "lbs" suffix
def convert_to_pounds(weight_str):
    if pd.isna(weight_str):
        return weight_str  # Return NaN as is
    if 'oz' or 'Oz' in weight_str:
        ounces = float(weight_str.split(' ')[0])  # Extract the number of ounces
        pounds = ounces / 16.0  # Convert ounces to pounds (1 pound = 16 ounces)
        return f'{pounds:.2f}'  # Remove "lbs" suffix
    elif 'lbs' in weight_str:
        return weight_str.split(' ')[0]  # Remove "lbs" suffix
    else:
        return 'Invalid'


# Function to consolidate weight columns
def consolidate_weights(row):
    for col in row.index:
        if not pd.isna(row[col]) and 'weight' in col:
            return row[col]
    return np.nan


# Function to convert non-numeric characters to numeric and coerce to NaN if errors occur
def clean_numeric_columns(df, columns):
    df[columns] = df[columns].replace(r'[^0-9.]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    return df


def consolidate_dimensions(row):
    for col in row.index:
        if not pd.isna(row[col]) and 'dimensions' in col:
            return row[col]
    return np.nan


def convert_to_grams(weight_str):
    if pd.isna(weight_str):
        return weight_str # Retrun NaN as is
    if 'oz' in weight_str or 'Oz' in weight_str:
        ounces = float(weight_str.split(' ')[0]) # Extract the number of ounces
        return ounces * 28.3495 # convert ounces to grams
    elif 'lb' in weight_str or 'pounds' in weight_str:
        pounds = float(weight_str.split(' ')[0])
        return  pounds * 453.592  # convert ounces to grams

    elif 'Kg' in weight_str or 'kg' in weight_str:
        kgrams = float(weight_str.split(' ')[0])
        return kgrams * 1000
    else:
        processed_string = re.sub(r'gr', '', weight_str)  # Removes the word "gr" as a whole word
        processed_string = processed_string.replace('"', '')  # Removes double quotes ("")
        processed_string = processed_string.replace('\\','') # Replace backslah (\) with empty string in the 'extracted_weight' column
        return processed_string

def extract_dimensions(dimensions_str, dimension,split_num):
    if pd.isna(dimensions_str):
        return dimensions_str
    splitt_array = dimensions_str.replace('″', '').strip().split('x')
    if 'cm' in dimensions_str:
        if dimension == 'width':
            width = splitt_array[split_num]
            return width
        if dimension == 'height':
            width = splitt_array[split_num]
            return width
        if dimension == 'depth':
            width = splitt_array[split_num]
            return width
    elif 'mm' in dimensions_str:
        if len(splitt_array) >= 3:
             if dimension == 'width':
                width = splitt_array[split_num] + ' mm'
                return width
             if dimension == 'height':
                width = splitt_array[split_num] + ' mm'
                return width
             if dimension == 'depth':
                width = splitt_array[split_num] + ' mm'
                return width
    else:
        return 'Invalid'


def extract_resolutions(resolution_str, resolution_dim):
    if pd.isna(resolution_str):
        return resolution_str
    splitt_array = resolution_str.split('x')
    if resolution_dim == 'x':
        return splitt_array[0]
    else:
        return splitt_array[1]




def convert_mm_to_cm(dimension_str):
    dimension_str = str(dimension_str)
    if pd.isna(dimension_str):
        return dimension_str
    elif 'cm' in dimension_str:
        dimension_str_splitted = dimension_str.split(' ')
        return dimension_str_splitted[0]
    elif not any(char.isdigit() for char in dimension_str):
        return 0
    else:
        numeric_part = re.search(r'\d+', dimension_str)

        return float(numeric_part.group()) / 10.0


def convert_to_cm(dim_str,unit=None):
    # Conversion factor from inches to centimeters
    cm_per_inch = 2.54

    if not pd.isna(dim_str):
        dim_str = dim_str.strip()

    if pd.isna(dim_str):
        return dim_str


    elif 'mm' in dim_str:
        dimension_str_splitted = dim_str.split(' ')
        return float(dimension_str_splitted[0]) / 10.0

    elif 'in' in dim_str or unit=='inch':
        # Convert inches to centimeters
        cm = float(dim_str.split(' ')[0]) * cm_per_inch
        return cm
    elif 'cm' in dim_str:
        dimension_str_splitted = dim_str.split(' ')
        return dimension_str_splitted[0]

    else:
        processed_string = re.sub(r'[^\d.]', '', dim_str)
        return processed_string





# Define a function to extract values for 'd,' 'w,' and 'h' from flexible patterns
def extract_values(text):
    if pd.isna(text):
        return text

    values = {'d': None, 'w': None, 'h': None}

    # Find all numeric values and labels for 'd,' 'w,' and 'h'
    matches = re.findall(r'(\d+\.\d+|\d+)[^\d]*(d|w|h)', text)

    for value, label in matches:
        values[label] = value

    return values


# Function to conditionally assign values based on specific strings
def assign_value(text):
    if pd.isna(text):
        return text
    if 'CMOS' in text:
        return 'CMOS'
    elif 'MOS' in text:
        return 'MOS'
    elif 'CCD' in text:
        return 'CCD'
    else:
        return math.nan



def check_and_create_columns(df, column_list):
    missing_columns = [col for col in column_list if col not in df.columns]
    for col in missing_columns:

        df[col] = "/"
    return df

def del_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")


def delete_files_from_folder(folder_path,files_to_del='ALL'):
    if files_to_del == 'ALL':
        # List all files in the folder
        files = os.listdir(folder_path)

        files_count = len(files)

        # Check whether the folder contains any files
        if not files:
            print(f"There are no files in the {folder_path} to be deleted!!!")
            return


        # Iterate through the files and delete them
        for file in files:
            file_path = os.path.join(folder_path, file)
            del_file(file_path)

        print(f"All Data in the {folder_path} have been deleted successfully")

    else:
        full_file_path = os.path.join(folder_path, files_to_del)

        # Check if the specified file exists
        if not os.path.isfile(full_file_path):
            print(f"The file {full_file_path} does not exist!")
            return

        del_file(full_file_path)


def get_best_number_of_clusters_using_silhouette_score(data):
    silh_scores = []
    data_array = data.iloc[:, 1:].values

    list_k = list(range(2,18))

    for k in list_k:
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(data_array)
        silhouette_avg = silhouette_score(data_array, clusters)
        silh_scores.append(silhouette_avg)

    # Get the index of the highest number
    index_of_highest_number = silh_scores.index(max(silh_scores)) + 1
    highest_number = max(silh_scores)
    print("The best number of clusters is {} with an average silhouette score of {}".format(index_of_highest_number,highest_number))

    # Plot silh_scores against k
    plt.figure(figsize=(6,6))
    plt.plot(list_k, silh_scores, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Avg Silhouette Scores')

    return index_of_highest_number, highest_number



def get_best_number_of_clusters_using_wcss_score(data):
    wcss = []
    data_array = data.iloc[:, 1:].values
    for k in range(1, 20):  # You can adjust the range of k as needed
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_array)
        wcss.append(kmeans.inertia_)

    # Get the index of the highest number
    index_of_highest_number = wcss.index(min(wcss)) + 1
    highest_number = min(wcss)
    print("The best number of clusters is {} with an wcss score of {}".format(index_of_highest_number,highest_number))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 20), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

    return index_of_highest_number,highest_number

# Calculate the Euclidean distance between rows
def calculate_distances(row,df, columns_to_consider):
    distances = euclidean_distances([row[columns_to_consider]],df[columns_to_consider])
    return distances.flatten()


# Extract the closest 7 rows and calculate the ratio of 'is_match' values
def calcualte_ratio(row, df,columns_to_consider):
    #print("The following row has the value of 'is_match' {}".format(row['is_match']))
    distances = calculate_distances(row,df,columns_to_consider)
    closest_indices = np.argsort(distances)[1:10] # Exclude the row itself
    closest_rows = df.iloc[closest_indices]
    # Count the occurrences of the same 'is_match' label
    match_count = (closest_rows['is_match'] == row['is_match']).sum()

    # Calculate the ratio
    ratio = match_count / len(closest_rows)
    return ratio




def is_element_in_dict(cluster_lists,element_to_check):
    for inner_list in cluster_lists:
        if element_to_check in inner_list:
            return True,inner_list
    return False,[]

def pricedekho_extract_weight(text):
    if not pd.isna(text):
        match = re.search(r'Weight\s*\\n\s*(\d+)', text)
        if match:
            return match.group(1)
        else:
            return None

def pricedekho_extract_weight(text):
    if not pd.isna(text):
        match = re.search(r'Weight\s*\\n\s*(\d+)', text)
        if match:
            return match.group(1)
        else:
            return None

def pricedekho_extract_optical_zoom(text):
    if not pd.isna(text):
        match = re.search(r'Optical Zoom\s*\\n\s*(\d+)', text)
        if match:
            return match.group(1)
        else:
            return None

def pricedekho_extract_digital_zoom(text):
    if not pd.isna(text):
        match = re.search(r'Digital Zoom\s*\\n\s*(\d+)', text)
        if match:
            return match.group(1)
        else:
            return None