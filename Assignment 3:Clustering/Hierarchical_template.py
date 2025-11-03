
# coding: utf-8

import sys
from matplotlib import pyplot as plt
import numpy as np
import copy
import csv


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return np.array(dataMat)


def merge_cluster(distance_matrix, cluster_candidate, T):
    ''' Merge two closest clusters according to min distances
    1. Find the smallest entry in the distance matrix—suppose the entry 
        is i-th row and j-th column
    2. Merge the clusters that correspond to the i-th row and j-th column 
        of the distance matrix as a new cluster with index T

    Parameters:
    ------------
    distance_matrix : 2-D array
        distance matrix
    cluster_candidate : dictionary
        key is the cluster id, value is point ids in the cluster
    T: int
        current cluster index

    Returns:
    ------------
    cluster_candidate: dictionary
        upadted cluster dictionary after merging two clusters
        key is the cluster id, value is point ids in the cluster
    merge_list : list of tuples
        records the two old clusters' id and points that have just been merged.
        [(cluster_one_id, point_ids_in_cluster_one), 
         (cluster_two_id, point_ids_in_cluster_two)]
    '''
    merge_list = []

    # TODO
    # print(f'distance_matrix:{distance_matrix}, \n\ncluster_candidate:{cluster_candidate}, \n\nT:{T}')

    # 1. Find the smallest entry in the distance matrix—suppose the entry 
    #     is i-th row and j-th column
    min_dist = np.min(distance_matrix)
    # print(f'min_dist:{min_dist}') ## 0.14
    i, j = np.where(distance_matrix == min_dist)
    # print(f'distance matrix:\n{distance_matrix}\n\ni:{i}, j:{j}') ## i:[0 1], j:[1 0]. this equals to (0, 1) and (1, 0)
    index_i = i[0]
    index_j = j[0]
    # print(index_i)


    # 2. Merge the clusters that correspond to the i-th row and j-th column 
    #     of the distance matrix as a new cluster with index T
    # print(cluster_candidate) ## {1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5]}. key is the cluster id, value is point ids in the cluster
    cluster_id_1, cluster_id_2 = -1, -1
    for cluster_id, point_ids in cluster_candidate.items():
        if index_i in point_ids:
            cluster_id_1 = cluster_id
        if index_j in point_ids:
            cluster_id_2 = cluster_id
    # print(f'cluster_id_1:{cluster_id_1}, cluster_id_2:{cluster_id_2}') ## cluster_id_1:1, cluster_id_2:2

    if cluster_id_1 != cluster_id_2: # check if they are the same cluster
        # Merge clusters
        new_cluster_points = cluster_candidate[cluster_id_1] + cluster_candidate[cluster_id_2]
        # print(f'new_cluster_points:{new_cluster_points}') ## new_cluster_points:[0, 1]
        cluster_candidate[T] = new_cluster_points

        merge_list.append((cluster_id_1, cluster_candidate[cluster_id_1]))
        merge_list.append((cluster_id_2, cluster_candidate[cluster_id_2]))

        del cluster_candidate[cluster_id_1]
        del cluster_candidate[cluster_id_2]
    
        # print(cluster_candidate) ## {3: [2], 4: [3], 5: [4], 6: [5], 7: [0, 1]}
        # print(merge_list) ## [(1, [0]), (2, [1])]


    return cluster_candidate, merge_list


def update_distance(distance_matrix, cluster_candidate, merge_list):
    ''' Update the distantce matrix
    
    Parameters:
    ------------
    distance_matrix : 2-D array
        distance matrix
    cluster_candidate : dictionary
        key is the updated cluster id, value is a list of point ids in the cluster
    merge_list : list of tuples
        records the two old clusters' id and points that have just been merged.
        [(cluster_one_id, point_ids_in_cluster_one), 
         (cluster_two_id, point_ids_in_cluster_two)]

    Returns:
    ------------
    distance_matrix: 2-D array
        updated distance matrix       
    '''
    
    # TODO
    # update the distance matrix according to the merged cluster
    # check empty
    if not merge_list:
        return distance_matrix

    cluster_one_id, points_in_cluster_one = merge_list[0] ## 1, [0]
    cluster_two_id, points_in_cluster_two = merge_list[1] ## 2, [1]

     # For each data point not in the merged cluster, update its distance to the merged cluster
    merged_points_1 = points_in_cluster_one + points_in_cluster_two
    merged_points_2 = points_in_cluster_two + points_in_cluster_one

    # print(distance_matrix)

    # For each pair of data points between the the s-th and t-th clusters (one
    # point in the s-th cluster, and one point in the t-th cluster), change their
    # distance to a big number.
    for i in range(distance_matrix.shape[0]):
        if i not in merged_points_1:
            min_dist = float('inf')
            for p_merged in merged_points_1:
                min_dist = min(min_dist, distance_matrix[i, p_merged], distance_matrix[p_merged, i])
            for p_merged in merged_points_1:
                distance_matrix[i, p_merged] = min_dist
                distance_matrix[p_merged, i] = min_dist
    
    for p1 in merged_points_1:
        for p2 in merged_points_1:
            if p1 != p2:
                distance_matrix[p1,p2] = 100000

    return distance_matrix

def agglomerative_with_min(data, cluster_number):
    """
    agglomerative clustering algorithm with min link

    Parameters:
    ------------
    data : 2-D array
        each row represents an observation and 
        each column represents an attribute

    cluster_number : int
        number of clusters

    Returns:
    ------------
    clusterAssment: list
        assigned cluster id for each data point
    """
    cluster_candidate = {}
    N = len(data)
    # initialize cluster, each sample is a single cluster at the beginning
    for i in range(N):
        cluster_candidate[i+1] = [i]  #key: cluser id; value: point ids in the cluster
    # print(data) ## [[ 0.88845  0.96682  0.93679  0.81723  0.88242] ...]
    # print(cluster_candidate) ## {1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5]}
    
    # initialize distance matrix
    distance_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j == i: # or j<=i
                distance_matrix[i,j] = 100000 # set large value for self-distance
            else:
                distance_matrix[i,j] = np.sqrt(np.sum((data[i]-data[j])**2))
    
    # hiearchical clustering loop
    # print(f"N={N}, cluster_number={cluster_number}") ## N=6, cluster_number=1
    T = N + 1 #cluster index
    for i in range(N-cluster_number):
        cluster_candidate, merge_list = merge_cluster(distance_matrix, cluster_candidate, T)
        distance_matrix   = update_distance(distance_matrix, cluster_candidate, merge_list )
        print('%d-th merging: %d, %d, %d'% (i, merge_list[0][0], merge_list[1][0], T))
        T += 1
        # print(cluster_candidate)


    # assign new cluster id to each data point 
    clusterAssment = [-1] * N
    for cluster_index, cluster in enumerate(cluster_candidate.values()):
        for c in cluster:
            clusterAssment[c] = cluster_index
    # print (clusterAssment)
    return clusterAssment


def saveData(save_filename, data, clusterAssment):
    clusterAssment = np.array(clusterAssment, dtype = object)[:,None]
    data_cluster = np.concatenate((data, clusterAssment), 1)
    data_cluster = data_cluster.tolist()

    with open(save_filename, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data_cluster)
    f.close()



if __name__ == '__main__':
    if len(sys.argv) == 3:
        data_filename = sys.argv[1]
        cluster_number = int(sys.argv[2])
    else:
        data_filename = 'Example.csv'
        cluster_number = 1

    save_filename = data_filename.replace('.csv', '_hc_cluster.csv')

    data = loadDataSet(data_filename)

    clusterAssment = agglomerative_with_min(data, cluster_number)

    saveData(save_filename, data, clusterAssment)

    ## python Hierarchical_template.py Example.csv 1
    ## python Hierarchical_template.py Utilities.csv 1