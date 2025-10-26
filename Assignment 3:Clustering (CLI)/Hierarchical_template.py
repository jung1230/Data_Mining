
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
    # Find the smallest entry in the distance matrix
    i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

    cluster_one_id, cluster_two_id = -1, -1
    for cluster_id, points in cluster_candidate.items():
        if i in points:
            cluster_one_id = cluster_id
        if j in points:
            cluster_two_id = cluster_id
    
    if cluster_one_id != cluster_two_id:
        # Merge clusters
        new_cluster_points = cluster_candidate[cluster_one_id] + cluster_candidate[cluster_two_id]
        cluster_candidate[T] = new_cluster_points
        
        merge_list.append((cluster_one_id, cluster_candidate[cluster_one_id]))
        merge_list.append((cluster_two_id, cluster_candidate[cluster_two_id]))
        
        del cluster_candidate[cluster_one_id]
        del cluster_candidate[cluster_two_id]

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
    if not merge_list:
        return distance_matrix

    cluster_one_points = merge_list[0][1]
    cluster_two_points = merge_list[1][1]

    merged_points = cluster_one_points + cluster_two_points

    for i in range(distance_matrix.shape[0]):
        if i not in merged_points:
            min_dist = float('inf')
            for p_merged in merged_points:
                min_dist = min(min_dist, distance_matrix[i, p_merged], distance_matrix[p_merged, i])
            for p_merged in merged_points:
                distance_matrix[i, p_merged] = min_dist
                distance_matrix[p_merged, i] = min_dist
    
    for p1 in merged_points:
        for p2 in merged_points:
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

    # initialize distance matrix
    distance_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j == i: # or j<=i
                distance_matrix[i,j] = 100000
            else:
                distance_matrix[i,j] = np.sqrt(np.sum((data[i]-data[j])**2))
    
    # hiearchical clustering loop
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
        data_filename = 'Utilities.csv'
        cluster_number = 1

    save_filename = data_filename.replace('.csv', '_hc_cluster.csv')

    data = loadDataSet(data_filename)

    clusterAssment = agglomerative_with_min(data, cluster_number)

    saveData(save_filename, data, clusterAssment)
