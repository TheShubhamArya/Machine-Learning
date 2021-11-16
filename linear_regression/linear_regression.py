# Shubham Arya 1001650536
# Machine learning CSE 4309 - 001 
# Implement linear regression using batch learning regularized least-squares approach
# Run command for your terminal: 
# python3 linear_regression.py [path of training file] [path of test file] [degree] [lambda]

import sys
import math
import numpy as np

def linear_regression(training_file, test_file, degree, lamda):
    # TRAINING STAGE
    # open files and parse data into a 2D array
    train = open_file_and_parse_data(training_file)
    test = open_file_and_parse_data(test_file)

    phi = get_phi_values(train, degree)
    I = np.array(np.identity(len(phi[0])))
    t = train.T[len(train[0])-1]

    # w = (lambda*I + (phi^(-1))phi)^(-1) phi^(T) t
    phi_into_inverse_phi = np.dot(phi.T, phi)
    w = np.dot(np.dot(np.linalg.pinv(lamda*I + phi_into_inverse_phi), phi.T), t)

    # printing weights
    for i in range(len(w)):
        print("w%d=%.4f"%(i,w[i]))

    # TESTING STAGE
    phi = get_phi_values(test, degree)

    for i in range(len(test)):
        output = float(sum(w.T*phi[i]))
        target = test[i][-1]
        error = float((target - output)**2)
        print("ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f" % ((i+1),output,target,error))
        
# Returns a polynomial basis function phi
def get_phi_values(array, degree):
    rows = len(array)
    columns = len(array[0])
    phi = []
    for i in range(rows): # This loops through each row in the file
        temp = [1]
        for j in range(columns-1): # This loops through each column (attribute) in the file
            for k in range(degree): # For each attribute, generate a phi value based on the polynomial basis function
                temp.append(array[i][j]**(k+1))
        phi.append(temp)
    return np.array(phi).astype(np.float)

# Opens the file and parses the contents of the file into a 2D array depending on its type.
def open_file_and_parse_data(filepath):
    array = []
    with open(filepath, 'r') as file:
        for line in file:
            array.append(line.split())
    return np.array(array).astype(np.float)

# Main function that sends the command line arguments to the linear regression function.
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error: Incorrect format!\nThe run command is: python3 linear_regression.py [path of training file] [path of test file] [degree] [lambda]")
    else:
        args = sys.argv
        linear_regression(args[1], args[2], int(args[3]), float(args[4]))
