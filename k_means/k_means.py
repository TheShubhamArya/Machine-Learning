# Shubham Arya 1001650536
# Machine learning CSE 4309 - 001 
# Implement k-means clustering
# Run command for your terminal: 
# python3 k_means.py [data_file] [K] [initialization] 

import sys
import math
import numpy as np
import random

# calculates the error measure for all the clusters
def error_measure(num, mean):
    summation = 0
    for i in range(len(num)):
        summation += (num[i] - mean[i])**2
    return math.sqrt(summation)

def group(vals):
    values = set(map(lambda x: x[1], vals))
    return [[y[0] for y in vals if y[1] == x] for x in values]

# opens files and parse data into array. By default, the cluster numbers are put by round robin.
def open_file_and_parse_data(filepath, k):
    array = []
    with open(filepath, 'r') as file:
        clusterNum = 0
        for line in file:
            clusterNum += 1
            line_split = line.split()
            temp = [float(x) for x in line_split]
            array.append([temp,clusterNum]) # initalizes cluster with round robin value by default
            clusterNum = clusterNum % k
    return array 

# function that performs k_means 
def k_means(data_file, k, initialization):
    # open file   parse into data
    data = open_file_and_parse_data(data_file, k)
    if initialization == "random":
        for i in range(len(data)):
            data[i][1] = random.randint(1, k)

    grouped = group(data)
    cluster_means = []
    for i in range(k):
        cluster_means.append(np.mean(grouped[i], axis=0))

    same_cluster = False
    # Continues classification into clusters until the same cluster is repeatedly successively
    while(not same_cluster):
        new_cluster = []
        for i in range(len(data)):
            smallest_distance = 9999999 # selecting largest possible value so the smallest distance can be detected
            cluster_number = 1
            for x in range(len(cluster_means)):
                distance = error_measure(data[i][0], cluster_means[x])
                if distance < smallest_distance:
                    cluster_number = x + 1
                    smallest_distance = distance
            new_cluster.append([data[i][0], cluster_number])

        # check if clusters are same
        same_cluster = True
        for i in range(len(data)):
            if data[i][1] != new_cluster[i][1]:
                same_cluster = False
                break

        data = new_cluster.copy()
        grouped = group(data)
            
        # calculating new arithmetic mean of the new cluster
        for i in range(k):
            cluster_means[i] = np.mean(grouped[i], axis=0)

    # prints the output of k-means
    for i in range(len(data)):
        if len(data[i][0]) == 1:
            print("%10.4f --> cluster %d" % (data[i][0][0], data[i][1]))
        else:
            print("(%10.4f, %10.4f) --> cluster %d" % (data[i][0][0], data[i][0][1], data[i][1]))

# main function
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect arguments. The run command should be: python3 k_means.py [data_file] [K] [initialization] ")
    else:
        k_means(sys.argv[1],int(sys.argv[2]),sys.argv[3])
        