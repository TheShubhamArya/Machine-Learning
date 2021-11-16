# Shubham Arya 1001650536
# Machine learning CSE 4309 - 001 
# Implement decision trees and decision forests.
# Run command for your terminal: 
# python3 decision_tree.py [path of training file] [path of test file] [option] [pruning_thr]

import sys
import math
import numpy as np
import random

class Tree:
    def __init__(self, attribute=-1, threshold=-1, left=None, right=None, data=-1, gain=0):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.data = data
        self.gain = gain

# DTL_TopLevel is the top level function for decision tree learning. Makes the first call to the DTL function
def DTL_TopLevel(examples, attributes):
    default = get_class_distribution(examples)[1]
    return DTL(examples, attributes, default)

# DTL builds the entire tree, and it's recursive calls build each individual subtree, and each individual leaf node.
def DTL(examples, attributes, default):
    if len(examples) < pruning_thr:
        return Tree(data=default)
    elif 1 in get_class_distribution(examples):
        return Tree(data=get_class_distribution(examples))
    else:
        best_attribute, best_threshold, gain = choose_attribute(examples, attributes)
        tree = Tree(best_attribute, best_threshold, gain=gain)
        examples_left = [x for x in examples if x[best_attribute] < best_threshold]
        examples_right = [x for x in examples if x[best_attribute] >= best_threshold]
        tree.left = DTL(examples_left, attributes, get_class_distribution(examples))
        tree.right = DTL(examples_right, attributes, get_class_distribution(examples))
        return tree

# Search for best combination of attribute and threshold happens in this fucntion
def choose_attribute(examples, attributes):
    if option == "optimized":
        max_gain = best_attribute = best_threshold = -1
        for A in attributes:
            attribute_values = [x[A] for x in examples]
            L = min(attribute_values)
            M = max(attribute_values)
            for K in range(1, 51):
                threshold = L + K*(M-L)/51
                gain = information_gain(examples, A, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = A
                    best_threshold = threshold
        return (best_attribute, best_threshold, max_gain)
    elif option == "randomized":
        max_gain = best_threshold = -1
        A = random.choice(attributes)
        attribute_values = [x[A] for x in examples]
        L = min(attribute_values)
        M = max(attribute_values)
        for K in range(1, 51):
            threshold = L + K*(M-L)/51
            gain = information_gain(examples, A, threshold)
            if gain > max_gain:
                max_gain = gain
                best_threshold = threshold
        return (A, best_threshold, max_gain)

# gets the class label from the train data file
def get_classes(examples):
    classes = []
    counter = 0
    for i in range(len(examples)):
        if examples[i][-1] not in classes:
            classes.append(examples[i][-1])
    temp = dict.fromkeys(classes, 0)
    for i in sorted(temp.keys()):
        temp[i] = counter
        counter += 1
    return temp

# returns the probability distribution of each class label in the training set.
def get_class_distribution(examples):
    distribution_array = np.zeros(len(distribution))
    for i in range(len(examples)):
        distribution_array[distribution[examples[i][-1]]] += 1
    for i in range(len(distribution_array)):
        if len(examples) > 0:
            distribution_array[i] /= len(examples)
    return distribution_array

# Calculates the information gain of the data using the formula I(E,L) = H(E) - sum((Ki/K)H(Ei))
def information_gain(examples, attr, threshold):
    H_E = H_E1 = H_E2 = 0
    examples_left = []
    examples_right = []
    for i in examples:
        if i[attr] < threshold:
            examples_left.append(i)
        else:
            examples_right.append(i)
    dist = get_class_distribution(examples)
    left = get_class_distribution(examples_left)
    right = get_class_distribution(examples_right)
    for num in dist:
        if num > 0:
            H_E -= (num * np.log2(num))
    for num in left:
        if num > 0:
            H_E1 -= (num * np.log2(num))
    for num in right:
        if num > 0:
            H_E2 -= (num * np.log2(num))
    K = len(examples)
    K1 = len(examples_left)
    K2 = len(examples_right)
    final_entropy = H_E - ((K1 / K) * H_E1) - ((K2 / K) * H_E2)
    return final_entropy

# This function predicts the probability of a class label depending on the tree computed
def predict(tree, test_data):
    if tree.left == None and tree.right == None:
        return tree.data
    if test_data[tree.attribute] < tree.threshold:
        return predict(tree.left, test_data)
    else:
        return predict(tree.right, test_data)

# Every node is printed in breath first order with left children before right children
def breath_first_order(root, i, node_number):
    if not root:
        return
    queue = []
    queue.append(root)
    while queue:
        current_node = queue.pop(0)
        print("tree=%2d, node=%3d,feature=%2d, thr=%6.2f, gain=%f"%(i+1,node_number,current_node.attribute, current_node.threshold, current_node.gain))
        node_number += 1
        if current_node.left:
            queue.append(current_node.left)
        if current_node.right:
            queue.append(current_node.right)


# Opens the file and parse the contents of the file into a 2D array.
def open_file_and_parse_data(filepath):
    array = []
    with open(filepath, 'r') as file:
        for line in file:
            line_split = line.split()
            array.append(line_split)
    return np.array(array).astype(np.float64)

"""
Main function starts here
"""

if len(sys.argv) != 5:
    print("Incorrect run command! Try: python3 decision_tree.py [path of training file] [path of test file] [option] [pruning_thr].")
    exit(0)

option = sys.argv[3]
pruning_thr = int(sys.argv[4])

# open files and parse data into a 2D array
trainData = open_file_and_parse_data(sys.argv[1])
testData = open_file_and_parse_data(sys.argv[2])
attributes = range(len(trainData[0][:-1]))

# TRAINING PHASE
distribution = get_classes(trainData)
trees = []
if option == "optimized" or option == "randomized":
    trees.append(DTL_TopLevel(trainData, attributes))
else:
    if option == "forest3":
        option = "randomized"
        for i in range(3):
            trees.append(DTL_TopLevel(trainData, attributes))
    elif option == "forest15":
        option = "randomized"
        for i in range(15):
            trees.append(DTL_TopLevel(trainData, attributes))
    else:
        print("Invalid option. Enter: python3 decision_tree.py [path of training file] [path of test file] [optimized | randomized | forest3 | forest15] [pruning_thr]")
        exit(0)
# prints the tree that was generated during the training phase
for i in range(len(trees)):
    breath_first_order(trees[i], i, node_number=1)
            
# TESTING PHASE
total_correct = 0
for n in range(len(testData)):
    accuracy = 0
    distance = []
    for i in range(len(trees)):
        distance.append(predict(trees[i], testData[n]))
    predicted_class_index = np.argmax(distance)
    predicted_class = predicted_class_index
    if predicted_class_index > len(distance[0]):
        predicted_class = predicted_class_index % len(distance[0])
    if predicted_class == testData[n][-1]:
        accuracy = 1
    total_correct += accuracy
    print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f"%(n+1, predicted_class, int(testData[n][-1]), accuracy))
print("classification accuracy= %6.4f" % (total_correct/len(testData)))
