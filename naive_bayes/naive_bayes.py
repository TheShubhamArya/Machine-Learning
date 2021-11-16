# Shubham Arya 1001650536
# Machine learning CSE 4309 - 001 
# Implement Naive Bayes classifiers based on Gaussians.
# Run command for your terminal: 
# python3 naive_bayes.py [path of training file] [path of test file]

import sys
import enum
import math

# 2D array that will store the all the values
training_file_2D_array = []
test_file_2D_array = []

# Enum created to tell what type of file is opened
class File_Type(enum.Enum):
    training_file = 1
    test_file = 2

def naive_bayes(training_file, test_file):
    # open files and parse data into a 2D array
    open_file_and_parse_data(training_file, File_Type.training_file)
    open_file_and_parse_data(test_file, File_Type.test_file)

    # gets summary of the training file by calculating the mean and the stdev
    summary = get_summary_by_class(training_file_2D_array)
    display_output_of_training_phase(summary)

    # predict for the test file
    predictions = get_predictions(summary, test_file_2D_array)
    calculate_classification_accurancy(test_file_2D_array, predictions)

def display_output_of_training_phase(summary):
    for label in sorted(summary.keys()):
	    for (i,row) in enumerate(summary[label]):
		    print("class ",label,", attribute ",i+1, ", mean = %0.2f" % row[0],", std = %0.2f" % row[1])

# Returns a dictionary with class label as key, and a list under it with mean and stdev for all the attributes for that key.
def get_summary_by_class(data):
    dictionary = dict()
    # This will split data by class into dictionary. So all the data with similar class label will be grouped together under one key in the dictionary.
    for i in range(len(data)):
        values = data[i]
        class_value = values[-1] 
        if class_value not in dictionary:
            dictionary[class_value] = list()
        # This splits the values from the first row to the second last row that has the attributes.
        # The last row is not selected as that only has the class labels.
        dictionary[class_value].append(values[0:-1])
    summaries = dict() 
    # summaries[i][0] - mean, summaries[i][1] - stdev, summaries[1][2] - len
    for class_label, values in dictionary.items():
        summaries[class_label] = [(get_mean(attr), get_stdev(attr), len(attr)) for attr in zip(*values)]
    return summaries

# This function opens the file and parses the contents of the file into a 2D arrat depending on its type.
def open_file_and_parse_data(filepath, type):
    file = open(filepath, "r")
    for (i,line) in enumerate(file.readlines()):
        if type == File_Type.training_file:
            training_file_2D_array.append([])
        else:
            test_file_2D_array.append([])
        for string in line.split():
            value = float(string)
            if type == File_Type.training_file:
                training_file_2D_array[i].append(value)
            else:
                test_file_2D_array[i].append(value)

# This function returns the probaility density function for each attribute of a class label
def gaussian_formula(x, mean, stdev):
    exponent = math.exp(-(((x-mean)**2) / (2 * (stdev ** 2))))
    probability_density = (1 / (stdev*(math.sqrt(2 * math.pi))))*exponent
    return probability_density

# Calculates probabilities for each class using the bayes classifier.
def calculate_class_probabilities(summaries, row):
    p_Cj_given_x = dict()           # p(Cj|x) - probability that will be returned
    p_Cj = dict()                   # p(Cj) - probability of selecting 1 class out of total class
    p_x_given_Cj = dict()           # p(x|Cj) - probability density alculated by the gaussian formula
    p_x_given_Cj_into_p_Cj = dict() # p(x|Cj) * p(Cj)
    p_x = 0                         # Sum of all the p(x|Cj) * p(Cj)
    total_rows = len(training_file_2D_array)

    for class_label, class_summaries in summaries.items():
        p_x_given_Cj[class_label] = 1
        p_Cj[class_label] = summaries[class_label][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            p_x_given_Cj[class_label] *= gaussian_formula(row[i], class_summaries[i][0], class_summaries[i][1])

        p_x_given_Cj_into_p_Cj[class_label] = p_x_given_Cj[class_label] * p_Cj[class_label]
        p_x += (p_x_given_Cj[class_label] * p_Cj[class_label])
    
    for class_label, p_x_given_Cj_into_p_Cj in p_x_given_Cj_into_p_Cj.items():
        p_Cj_given_x[class_label] = p_x_given_Cj_into_p_Cj / p_x
    return p_Cj_given_x

# Returns a set of predictions for each row in the test set
def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)): # for each row in the test set
        probability_of_all_labels = calculate_class_probabilities(summaries, test_set[i])
        best_label = None       # Label of the class
        best_probability = 0    # Highest probability among all class labels
        count_ties = 0          # Counts the number of times their is a tie in probability
        for class_label, probability in probability_of_all_labels.items():
            if best_label == None or probability >= best_probability:
                if best_probability == probability and best_probability != 0:
                    count_ties += 1
                best_probability = probability
                best_label = class_label
        predictions.append([best_label, best_probability, count_ties])
    return predictions

def calculate_classification_accurancy(test_set, predictions):
    correct = 0 # Keeping count to calculate overall accuracy
    # predictions[i][0]- best_label
    # predictions[i][1]- best probabilty
    # predictions[i][2]- count ties
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i][0]:
            correct += 1
            accuracy = 1
            if predictions[i][2] > 0:
                accuracy = 1 / predictions[i][2]
            print("ID= %5d" % (i+1),", predicted= %3d" % predictions[i][0],", probability= %.4f" % predictions[i][1],", true= %3d" % test_set[i][-1],", accuracy= %4.2f" % accuracy)
        else:
            print("ID= %5d" % (i+1),", predicted= %3d" % predictions[i][0],", probability= %.4f" % predictions[i][1],", true= %3d" % test_set[i][-1],", accuracy= %4.2f" % 0)
    accuracy = correct / float(len(test_set))
    print("classification accuracy is %6.4f" % accuracy)

# function that returns mean of the values
def get_mean(numbers):
    return sum(numbers) / float(len(numbers))

# function that returns the standard deviation
def get_stdev(numbers):
    delta_x_square = 0
    avg = get_mean(numbers)
    for number in numbers:
        delta_x_square += (float(number) - float(avg))**2
    covariance = (delta_x_square / (len(numbers)-1))
    stdev = covariance ** (0.5)
    if stdev < 0.01:
        stdev = 0.01
    return stdev

# main function that sends the command line arguments to the naive_bayes function.
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: Incorrect format! The run command is: python3 naive_bayes.py [path of training file] [path of test file]")
        exit()
    else:
        args = sys.argv
        naive_bayes(args[1], args[2])