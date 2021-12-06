import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function        
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)      
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
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
    
    def Hard_limit(self, net):
        net[net >= 0] = 1
        net[net < 0] = 0
        return net

    def Linear(self, net):
        return net

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        weights = self.get_weights()
        net = np.dot(weights, X)

        if self.transfer_function == "Hard_limit":
            actual_value = self.Hard_limit(net)
        elif self.transfer_function == "Linear":
            actual_value = self.Linear(net)
        return actual_value
        
        
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        input_values = X
        target = y
        pseudo_inverse = np.linalg.pinv(input_values)
        self.weights = np.dot(target,pseudo_inverse) 

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
         
        learning = learning.lower()
        for epoch in range(num_epochs):
             for epoch_number in range(num_epochs):
                    for batch_wise in range(0, X.shape[1], batch_size):
                        if((batch_wise+batch_size) <= X.shape[1]):
                            if learning == "filtered":
                                alpha_t_pt = alpha * np.dot(y[:,batch_wise:batch_wise+batch_size],X[:,batch_wise:batch_wise+batch_size].T)
                                self.weights = (1 - gamma) * self.weights + alpha_t_pt
                            elif learning == "delta":
                                error = y[:,batch_wise:batch_wise+batch_size] - self.predict(X[:,batch_wise:batch_wise+batch_size])
                                self.weights += alpha * np.dot(error,X[:,batch_wise:batch_wise+batch_size].T)
                            elif learning == "Supervised_hebb":
                                self.weights += alpha * np.dot(self.predict(X[:,batch_wise:batch_wise+batch_size]),X[:,batch_wise:batch_wise+batch_size].T)
         
            
    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        predicted = self.predict(X)
        difference_of_array = np.subtract(predicted, y)
        square_of_array = np.square(difference_of_array)
        mean_squared_array = square_of_array.mean()
        return mean_squared_array
