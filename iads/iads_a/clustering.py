from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from . import kmoyennes as km

def normalisation(data):
    return km.normalisation(data).values

def centroide(matrix):
    if len(matrix.shape) == 1:  # conversion : 1-D array => 2-D array
        matrix = matrix.reshape(1,-1)
    return matrix.mean(axis=0)

def dist_groupes(first, second):
    return km.dist_vect(centroide(first), centroide(second))

def initialise(matrix):
    return {k:example for k, example in enumerate(matrix)}

def fusionne(partition):
    new_key = -1

    # keys of the closest clusters in current partition
    min_k1, min_k2 = -1, -1
    min_dist = float('inf')

    # key of the merged groups
    max_key = -1

    for k1, k2 in combinations(partition.keys(), 2):
        # distance for each pair of clusters
        dist = dist_groupes(partition[k1], partition[k2])
        if dist < min_dist:
            min_k1 = k1
            min_k2 = k2
            min_dist = dist

        if max(k1, k2) > max_key:
            max_key = max(k1, k2)

    new_partition = dict(partition)
    new_key = 1 + max_key

    # update new partition
    new_partition[new_key] = np.vstack( (partition[min_k1], partition[min_k2]) )
    new_partition.pop(min_k1)
    new_partition.pop(min_k2)

    return new_partition, min_k1, min_k2, min_dist

def agglomerate_clusters(initial_clusters):
    # initialisation
    current_clusters = initial_clusters
    merge_matrix = []

    # while there is at least 2 sets to be merged
    while len(current_clusters) >= 2:
        # print the current number of clusters as feedback
        if len(current_clusters) % 5000 == 0:
            print(len(current_clusters))

        # merge the two closest clusters
        new_clusters, k1, k2, dist_min = fusionne(current_clusters)
        # update the merge matrix
        if len(merge_matrix) == 0:
            merge_matrix = [k1, k2, dist_min, 2]
        else:
            merge_matrix = np.vstack( [merge_matrix, [k1, k2, dist_min, 2]] )
        # update the cluster partition
        current_clusters = new_clusters

    return merge_matrix

def plot_hierarchical_clustering(merge_matrix, labels):
    import scipy.cluster.hierarchy

    # Paramètre de la fenêtre d'affichage:
    plt.figure(figsize=(80, 500)) # taille : largeur x hauteur
    plt.title('Dendrogram', fontsize=25)
    plt.xlabel('Example', fontsize=25)
    plt.ylabel('Distance', fontsize=25)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Construction du dendrogramme à partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(
        merge_matrix,
        leaf_font_size=30.,  # taille des caractères de l'axe des X
        labels=labels,
        orientation='right',
        link_color_func=lambda k: colors[k%len(colors)]
    )

    # Affichage du résultat obtenu:
    plt.show()

def clustering_hierarchique(data, labels=None):
    matrix = normalisation(data)
    initial_clusters = initialise(matrix)
    merge_matrix = agglomerate_clusters(initial_clusters)
    plot_hierarchical_clustering(merge_matrix, labels)
    return merge_matrix
