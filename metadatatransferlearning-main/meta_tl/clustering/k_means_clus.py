import numpy as np
from sklearn.cluster import KMeans


def predict_clusters(df,num_of_clusters,feature_vectors_list):
    # Number of clusters
    n_clusters = num_of_clusters

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df.iloc[:, 1:].values)


    # Organize strings based on their cluster assignments
    cluster_strings = {}
    for i, cluster in enumerate(kmeans.labels_):
        string = feature_vectors_list[i]
        if cluster not in cluster_strings:
            cluster_strings[cluster] = []
        cluster_strings[cluster].append(string)



    # Print cluster number and strings belonging to each other
    for cluster, strings in cluster_strings.items():
        print(f"Cluster Number: {cluster}")
        print(f"Elements: {', '.join(strings)}")
        print()

    clusters = kmeans.labels_

    # Add the cluster assignments to the DataFrame
    df['Cluster'] = clusters

    # Count the occurrences of each cluster
    cluster_counts = np.bincount(clusters)

    # Print the cluster counts
    print("\nNumber of elements in each cluster:")
    for cluster_num, count in enumerate(cluster_counts):
        print(f"Cluster {cluster_num}: {count} elements")

    return cluster_strings