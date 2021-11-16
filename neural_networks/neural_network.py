# Shubham Arya 1001650536
# Machine learning CSE 4309 - 001 
# Implement neural networks, using the backpropagation algorithm
# Run command for your terminal: 
# python3 neural_network.py [path of training file] [path of test file] [layers] [units_per_layer] [rounds]

import sys
import math
import numpy as np
import random

wordToNumberDictionary = dict()
numberToWordDictionary = dict()

def neural_network(training_file, test_file, layers, unitsPerLayer, rounds):
    # TRAINING STAGE
    # open files and parse data into a 2D array and normalize the data
    trainData = open_file_and_parse_data(training_file)
    testData = open_file_and_parse_data(test_file)
    trainData, testData = normalize(trainData, testData)

    classes = get_classes(trainData)
    K = len(classes) # len of class
    D = len(trainData[0])-1  # number of attributes
    # intialize J
    J = np.zeros(layers, dtype=int)
    for i in range(len(J)):
        J[i] = unitsPerLayer
    J[0] = int(D)
    J[-1] = int(K)

    # initalize t
    t = np.zeros((len(trainData), K))
    for i in range(len(trainData)):
        t[i] = classes[int(trainData[i][-1])]

    # initialize the weights to uniform distribution
    maxVal = max(K, D, unitsPerLayer)
    b = initialize_weights(np.zeros((layers, maxVal)))
    w = initialize_weights(np.zeros((layers, maxVal, maxVal)))

    for round in range(rounds):  
        for n in range(len(trainData)):  # for each input in a round
            # Step 1: Initialize input layer
            z = np.zeros((layers,), dtype=np.ndarray) # sets number of rows
            z[0] = trainData[n][:-1]  # set number of columns for input layer

            # Step 2: Compute outputs
            a = np.ndarray((layers,), dtype=np.ndarray)
            for l in range(1, layers):
                a[l] = np.zeros(J[l])
                z[l] = np.zeros(J[l])
                for i in range(J[l]):
                    weighted_sum = 0
                    for j in range(J[l-1]):
                        weighted_sum += (w[l][i][j] * z[l-1][j])
                    a[l][i] = b[l][i] + weighted_sum
                    z[l][i] = sigmoid(a[l][i])

            # Step 3: Compute New delta Vals
            delta = np.zeros((layers,), dtype=np.ndarray)
            for i in range(layers):
                delta[i] = np.zeros(J[i])
            L = layers - 1
            for i in range(J[-1]):
                delta[L][i] = (z[L][i] - t[n][i]) * z[L][i] * (1-z[L][i])
            for l in range(L-1, 0, -1):
                delta[l] = np.zeros(J[l])
                for i in range(J[l]):
                    summation = 0
                    for k in range(J[l+1]):
                        summation += delta[l+1][k] * w[l+1][k][i]
                    delta[l][i] = summation * z[l][i] * (1-z[l][i])

            # Step 4: Update Weights
            n = calculate_learning_rate(round)
            for l in range(1, layers):
                for i in range(J[l]):
                    b[l][i] = b[l][i] - n * delta[l][i]
                    for j in range(J[l-1]):
                        w[l][i][j] = w[l][i][j] - (n * delta[l][i] * z[l-1][j])

 
    # TESTING STAGE
    total_correct = 0
    for n in range(len(testData)):
        # Step 1: Initialize input layer
        z = np.zeros((layers,), dtype=np.ndarray) # sets number of rows
        z[0] = testData[n][:-1]  # set number of columns for input layer

        # Step 2: Compute outputs
        a = np.ndarray((layers,), dtype=np.ndarray)
        for l in range(1, layers):
            a[l] = np.zeros(J[l])
            z[l] = np.zeros(J[l])
            for i in range(J[l]):
                weighted_sum = 0
                for j in range(J[l-1]):
                    weighted_sum += w[l][i][j] * z[l-1][j]
                a[l][i] = b[l][i] + weighted_sum
                z[l][i] = sigmoid(a[l][i])

        predict_class = -999999 # A class value which will never be encountered
        predict_value = 0
        accuracy = 0
        for i in range(J[-1]):
            if z[layers-1][i] >= predict_value:
                predict_value = z[layers-1][i]
                predict_class = sorted(classes.keys())[i]

        if testData[n][-1] == predict_class:
            accuracy = 1
    
        total_correct += accuracy
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % ((n+1), numberToWordDictionary[predict_class], numberToWordDictionary[int(testData[n][-1])], accuracy))
    print("classification accuracy=%6.4f" % (total_correct/len(testData)))
        

# Opens the file and parse the contents of the file into a 2D array depending on its type. The last column contains class labels that are strings. I map each label to an integer value using dictionary.
def open_file_and_parse_data(filepath):
    array = []
    with open(filepath, 'r') as file:
        for line in file:
            line_split = line.split()
            temp = line_split[0:-1]
            class_label = line_split[-1]
            if class_label not in wordToNumberDictionary:
                wordToNumberDictionary[class_label] = len(wordToNumberDictionary)
                numberToWordDictionary[wordToNumberDictionary[class_label]] = class_label
            temp.append(wordToNumberDictionary[class_label])
            array.append(temp)
    return np.array(array).astype(np.float64)

# normalize all attribute values, by dividing them with the MAXIMUM ABSOLUTE value over all attributes over all training objects for that dataset
def normalize(train, test):
    maximum = 0
    for i in range(len(train)):
        if max(train[i][0:-1]) > maximum:
            maximum = max(train[i][0:-1])
    for i in range(len(train)):
        train[i][0:-1] /= maximum
    for i in range(len(test)):
        test[i][0:-1] /= maximum
    return train, test

def get_classes(data):
    classes = dict()
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes[data[i][-1]] = []
    count = 0
    for i in sorted(classes.keys()):
        temp = np.zeros(len(classes))
        temp[count] = 1
        count += 1
        classes[i] = temp
    return classes

# initializes weights unfiformly between -0.05 and 0.05
def initialize_weights(weights):
    for x in np.nditer(weights, op_flags=['readwrite']):
        x[...] = random.uniform(-0.05, 0.05)
    return weights

def calculate_learning_rate(r):
    return 0.98**(r-1)

# formula for sigmoid function with input a[l][i]
def sigmoid(a_li):
    return 1/(1+math.exp(-a_li))

# main function that sends the command line arguments to the linear regression function.
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Error: Incorrect format!\nThe run command is: python3 neural_network.py [path of training file] [path of test file] [layers] [units_per_layer] [rounds]")
    else:
        args = sys.argv
        neural_network(args[1], args[2], int(args[3]), int(args[4]), int(args[5]))
