# Sadashiva, Sowjanya
# 1001_898_874
# 2021_09_26
# Assignment_01_01

import numpy as np
import random

class SingleLayerNN(object):
    def __init__(self, input_dimensions = 2, number_of_nodes = 4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions + 1)
        self.initialize_weights()
    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)          
            self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions + 1)

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """        
        
        if self.weights.shape != W.shape:
           return -1
        else:
            self.weights = W
        

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights
    
    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """        
        array_of_ones = np.ones((1,X.shape[1]))
        input_array = np.concatenate((array_of_ones,X)).transpose()
        target = np.zeros([self.number_of_nodes,X.shape[1]], int).transpose()
        for (number_of_sample_data, actual_output) in zip(input_array, target):
            for nodes in range(self.number_of_nodes):
                hardlimit_a = None
                wxb = 0
                for (array_of_inputs, weights) in zip(number_of_sample_data, self.weights[nodes]):
                    wxb += (array_of_inputs * weights)               
                if wxb >= 0: hardlimit_a = 1
                else: hardlimit_a = 0
                actual_output[nodes] = hardlimit_a
        return(target.transpose())
    
    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        actual = self.predict(X).transpose()        
        target_output = Y.transpose()
        array_of_ones = np.ones((1,X.shape[1]))
        input_array = np.concatenate((array_of_ones,X),axis= 0).transpose()                   
        for epoch in range(num_epochs):            
            for (array_input, target_output, target_output_row) in zip(input_array, target_output, range(len(target_output))):
                actual_output = actual[target_output_row]
                for nodes in range(self.number_of_nodes):
                    error = target_output[nodes] - actual_output[nodes]
                    for (X, weight, weightmatrix) in zip(array_input, self.weights[nodes],range(len(self.weights[nodes]))):
                        self.weights[nodes][weightmatrix] = weight + (alpha * error * X)                
       
    

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """           
        target_output = Y.transpose()
        actual_output = self.predict(X).transpose()
        error_count = 0
        no_of_input_samples = 0
        for (target, actual) in zip(target_output, actual_output):
            no_of_input_samples += 1
            for nodes in range(self.number_of_nodes):
                if target[nodes]!= actual[nodes]:
                    error_count += 1
                    break
        percent_error = (error_count / no_of_input_samples) * 100
        return(percent_error)


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())
