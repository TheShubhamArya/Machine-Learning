# Shubham Arya 1001650536
# Machine learning CSE 4309 - 001 
# Implement the value iteration algorithm
# Run command for your terminal: 
# python3 value_iteration.py [environment_file] [non_terminal_reward] [gamma] [K]

import sys
import math
import numpy as np
import random
import csv

# opens files and parse data into array using a csv reader.
def open_file_and_parse_data(filepath):
    array = []
    with open(filepath) as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            array.append(row)
    return array 

# returns the non terminal reward for each state
def R(state):
    if state == '.':
        return non_terminal_reward
    elif state == 'X':
        return 0
    else:
        return float(state)

directions = ["Up","Down","Left","Right"]

# computes the values of p(s'|s,a) for bellmans equation
def get_p_of_s2_given_s1_for_a(sprime, s, a, N):
    value = 0
    if a == "Up":
        for direction in directions: #goes through all directions except the direction opposite to current action.
            if direction != "Down": # these new directions are used to get newState and probability for the new state.
                newState, probability = get_state_probability(s,direction,a)
                value = add_update_value(newState,s,sprime,N, value, probability)

    elif a == "Left":
        for direction in directions:
            if direction != "Right":
                newState, probability = get_state_probability(s,direction,a)
                value = add_update_value(newState,s,sprime,N, value, probability)

    elif a == "Down":
        for direction in directions:
            if direction != "Up":
                newState, probability = get_state_probability(s,direction,a)
                value = add_update_value(newState,s,sprime,N, value, probability)

    elif a == "Right":
        for direction in directions:
            if direction != "Left":
                newState, probability = get_state_probability(s,direction,a)
                value = add_update_value(newState,s,sprime,N, value, probability)
    return value

# A function that gets the new state depending on the direction and returns a probability
def get_state_probability(s,c,a) :
    probability = 0.1
    if a == c:
        probability = 0.8
    if c == "Up":
        newState = (s[0]-1, s[1])
    elif c == "Left":
        newState = (s[0], s[1]-1)
    elif c == "Right":
        newState = (s[0], s[1]+1)
    elif c == "Down":
        newState = (s[0]+1, s[1])
    return newState, probability

# this adds and updates the value for a part in bellman equation if the s' = s
def add_update_value(newState,s,sprime,N, value, probability):
    if not valid(newState, N):
        newState = s
    if newState == sprime:
        value += probability
    return value

# checks the validity of new state when the directions are changed. So when +1 is added to a row or a column, this function checks whether that state exists 
def valid(s, N):
    if s[0] >= 0 and s[0] < len(N):
        if s[1] >= 0 and s[1] < len(N[1]):
            if N[s[0]][s[1]] == 'X':
                return False
            return True
    return False

# gets the intermediate value for the Belman equation which is compared with the max value. 
def find_max(state, U, N, action,S):
    sum = 0
    for i in range(N[0]):
        for j in range(N[1]):
            p_of_s2_given_s1_for_a = get_p_of_s2_given_s1_for_a((i, j), state, action, S)
            sum += (p_of_s2_given_s1_for_a * U[i][j])
    return sum

# function that performs the value iteration algorithm. This computes the utility of each state for a Markov Decision Process.
def value_iteration(S,A,K):
    N = len(S), len(S[0]) # N holds the size of the set S
    U2 = np.zeros(N)      # U2 is U prime, initialized with the size of N with 0 values
    policy = [ [ 'x' for i in range(len(S[0])) ] for j in range(len(S)) ]
    # print(policy)
    for _ in range(K):
        U = U2.copy()
        # i loops over rows, j loops over columns. Together it loops over each state s in S
        for i in range(N[0]): 
            for j in range(N[1]):
                s = S[i][j]
                if S[i][j] == 'X': # for blocked states
                    U2[i][j] = 0
                    policy[i][j] = 'X'
                elif S[i][j] != '.': # for terminal states
                    U2[i][j] = float(s)
                    policy[i][j] = 'o'
                else: # for non terminal states
                    max_value = 0
                    for a in A: # for each possible action from a set of actions
                        value = find_max((i,j),U,N,a,S)
                        max_value = max(value, max_value)
                        if value == max_value:
                            policy[i][j] = A[a]
                    U2[i][j] = R(s) + gamma * max_value # Bellman update step as these values are from the bellman equation
    return U, policy

# main function
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Incorrect arguments. The run command should be: python3 value_iteration.py [environment_file] [non_terminal_reward] [gamma] [K] ")
    else:
        environment_file = sys.argv[1]
        non_terminal_reward = float(sys.argv[2])
        gamma = float(sys.argv[3])
        K = int(sys.argv[4])
        A = {"Up": "^",
             "Right": ">",
             "Down": "v",
             "Left": "<"}

        # open file and parse into S (Set of states)
        S = open_file_and_parse_data(environment_file)
        # value iteration returns the utilities and policy for the environment
        utilities, policy = value_iteration(S,A,K)
        # Prints the output
        print("utilities:")
        for i in range(len(utilities)):
            for j in range(len(utilities[0])):
                print("%6.3f"%(utilities[i][j]), end=" ")
            print("")
        print("\npolicy:")
        for i in range(len(policy)):
            for j in range(len(policy[0])):
                print("%6s"%(policy[i][j]), end=" ")
            print("")
        