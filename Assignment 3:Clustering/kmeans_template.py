
# coding: utf-8

import sys
from numpy import *
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
    return np.asmatrix(dataMat)


def loadCenterSet(fileName):      #general function to parse tab -delimited floats
    centerMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        centerMat.append(fltLine)
    return np.asmatrix(centerMat)


def assignCluster(dataSet, k, centroids):
    '''For each data point, assign it to the closest centroid
    Inputs:
        dataSet: each row represents an observation and 
                 each column represents an attribute
        k:  number of clusters
        centroids: initial centroids or centroids of last iteration
    Output:
        clusterAssment: list
            assigned cluster id for each data point
    '''
    #TODO
    clusterAssment = [0] * len(dataSet)
    # print(centroids) ## [[4.4 3.  1.3 0.2][5.9 3.  5.1 1.8][4.  3.  4.  1.2]]
    # print(dataSet) ## [[5.4 3.9 1.7 0.4][6.4 3.1 5.5 1.8]....]

    for dataIndex, data in enumerate(dataSet):
        # print(data) ## [[5.4 3.9 1.7 0.4]]
        minDist = float('inf')
        for centroidsIndex, centroid in enumerate(centroids):
            # calculate Euclidean distance between xi and each of the K centroids
            dist = np.linalg.norm(data - centroid, 2) # 2 for Euclidean distance(L2 norm)
            # print(dist)
            if dist < minDist:
                minDist = dist

                # Assign xi to the cluster whose centroid is the closest to xi
                clusterAssment[dataIndex] = centroidsIndex
    # print(clusterAssment) ## [0, 1, 0, 0, 0, 0...]

    return clusterAssment


def getCentroid(dataSet, k, clusterAssment):
    '''recalculate centroids
    Input: 
        dataSet: each row represents an observation and 
            each column represents an attribute
        k:  number of clusters
        clusterAssment: list
            assigned cluster id for each data point
    Output:
        centroids: cluster centroids
    '''
    
    #TODO
    # print(dataSet.shape) ## (150, 4)
    centroids = np.zeros((k, dataSet.shape[1])) ## [[0. 0. 0. 0.][0. 0. 0. 0.][0. 0. 0. 0.]]
    # Calculate its centroid as the mean of all the objects in that cluster
    for centroidsIndex in range(k):
        #  get all data assigned to this cluster
        dataInCluster = []
        for dataIndex, cluster in enumerate(clusterAssment):
            if cluster == centroidsIndex:
                # print(dataSet[dataIndex].A1) # turn it into 1D from 2d
                dataInCluster.append(dataSet[dataIndex].A1)
        # print(np.array(dataInCluster)) ## [[5.1 3.5 1.4 0.2][4.9 3.  1.4 0.2]...]

        # calculate the mean
        centroids[centroidsIndex] = np.mean(np.array(dataInCluster), axis=0)
        # print(centroids[centroidsIndex]) ## [5.006 3.428 1.462 0.246]...
    
    return centroids


def kMeans(dataSet, T, k, centroids):
    '''
    Input:
        dataSet: each row represents an observation and 
                each column represents an attribute
        T:  number of iterations
        k:  number of clusters
        centroids: initial centroids
    Output:
        centroids: final cluster centroids
        clusterAssment: list
            assigned cluster id for each data point
    '''
    clusterAssment = [0] * len(dataSet)
    pre_clusters  = [1] * len(dataSet) # previous cluster assignment, check convergence

    i=1
    while i < T and list(pre_clusters) != list(clusterAssment):
        pre_clusters = copy.deepcopy(clusterAssment) 
        clusterAssment = assignCluster(dataSet, k, centroids )
        centroids      = getCentroid(dataSet, k, clusterAssment)
        i=i+1

    return centroids, clusterAssment


def saveData(save_filename, data, clusterAssment):
    clusterAssment = np.array(clusterAssment, dtype = object)[:,None]
    data_cluster = np.concatenate((data, clusterAssment), 1)
    data_cluster = data_cluster.tolist()

    with open(save_filename, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data_cluster)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        data_filename = sys.argv[1]
        centroid_filename = sys.argv[2]
        k = int(sys.argv[3])
    else:
        data_filename = 'Iris.csv'
        centroid_filename = 'Iris_Initial_Centroids.csv'
        k = 3

    save_filename = data_filename.replace('.csv', '_kmeans_cluster.csv')

    data = loadDataSet(data_filename)
    centroids = loadCenterSet(centroid_filename)
    centroids, clusterAssment = kMeans(data, 7, k, centroids ) # for iris, T=12. For yeast gene, T=7
    print(centroids)
    saveData(save_filename, data, clusterAssment)


    ### Example: python kmeans_template.py Iris.csv Iris_Initial_Centroids.csv
    ### Example: python kmeans_template.py YeastGene.csv YeastGene_Initial_Centroids.csv 6