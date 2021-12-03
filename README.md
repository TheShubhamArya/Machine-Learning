# Machine-Learning
Here, I go through over what happens in each code very briefly whle also attaching the links to lecture notes for each part. I have also attached a PDF of the course textbook called "Pattern Recognition and Machine Learning" which provides more information on the logic of the code.

## Note:
Some of the codes take a very long time (5-10mins depending on your computer) to run since they are written from scratch. 

## Run commands and data files
Each python file will have the run commands commented at the start to run the file. Most folders have some training and test files to run the code. However, if you need more datasets, you can find it [here](https://athitsos.utasites.cloud/courses/cse4309_fall2021/assignments/uci_datasets/).

## Naive Bayes Classifier
The Naive Bayes classifier is a proabablistic machine learning model that differentiates different objects based on their features. This is done using the Bayes theorem which finds out the probability of an object given the features. This classifier is called "naive" because it assumes that each input variable is independent.
#### [Slides for Bayes Classifier and Generative Methods](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/04_bayes_classifiers.pdf)


## Linear Regression
Linear regression is a linear model which means that there is a linear relationship between the input variable and the output variable. Once we have the model ready, making predictions only requires to solve the equation for specific set of input values.
#### [Slides for linear model regression](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/07_linear_regression.pdf)

## Neural Networks
Neural netowrks is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics human brain,like neurons. A “neuron” in a neural network is a mathematical function that collects and classifies information according to a specific architecture. The network bears a strong resemblance to statistical methods such as curve fitting and regression analysis. A neural network contains interconnected nodes called perceptrons. The perceptron feeds the signal produced by a multiple linear regression into an activation function. In a multi-layered perceptron (MLP), perceptrons are arranged in interconnected layers. The input layer collects input patterns. The output layer has classifications or output signals to which input patterns may map. Hidden layers fine-tune the input weightings until the neural network’s margin of error is minimal.
#### Slides for Neural Network
##### [Part 1: Introduction](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/09a_neural_networks.pdf)

##### [Part 2: Training Perceptrons and Handling Multiclass problems](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/09b_neural_networks.pdf)

##### [Part 3: Backpropagation](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/09c_neural_networks.pdf)

## Decision Tree
Decision trees can be used to represent decison making by branching down based on specific conditions. 
#### Slides for Decision Trees
##### [Part 1: Basic Definitions](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/11a_decision_trees.pdf)

##### [Part 2: Practical Issues](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/11b_decision_trees.pdf)

## K-Means
This is a unsupervised learning algorithm. It make inferences from datasets using only input vectors without referring to known, or labelled, outcomes. The objective is to group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset. This algorithm starts by randomly selecting centroids which are used as the beginning points. After that, through iteration, these centroids points are optimized and when there is no change in the centroids position or when desired iterations are achieved, we have our cluster.
#### [Slides for K-means and clustering](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/17_clustering.pdf)

## Value iteration algorithms
Value iteration algorithm allows us to numerically calculate the values of the states of Markov decision processes, with known transition probabilities and rewards. A better source to read more is linked [here](https://towardsdatascience.com/the-value-iteration-algorithm-4714f113f7c5).
#### Markov Decision Processes
##### [Part 1: Basic Definitions](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/18a_mdp.pdf)

##### [Part 2: Utilities of states and the Bellman Equation](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/18b_mdp.pdf)

##### [Part 3: Utilities of states and the Bellman Equation](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/18c_mdp.pdf)

#### [Reinforcement Learning](https://athitsos.utasites.cloud/courses/cse4309_fall2021/lectures/19_rl.pdf)

## Learning Outcomes
After successfully taking this course, I am familiar with standard approaches to machine learning, know the pros and cons of these approaches, able to implement basic machine learning methods, and able to apply basic machine learning methods to real world problems.
