# Jain, Mayank Ambalal
# 1001-761-066
# 2020-10-11
# Assignment-02-01

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
        if seed!=None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if self.weights.shape!=W.shape:
            return -1
        self.weights = W

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        if self.transfer_function.lower() == "hard_limit" :
            y_hat = np.dot(self.weights,X)
            y_hat = np.where(y_hat>=0,1,0)
            return y_hat
        else:
            y_hat = np.dot(self.weights,X)
            return y_hat

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        self.pseudo_inv = np.linalg.pinv(X)
        self.weights = np.dot(y,self.pseudo_inv) 

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
        for i in range(num_epochs):
            bse = 0  
            for j in range(X.shape[1]//batch_size):
                x_bs = X[:,bse:bse+batch_size]
                y_bs = y[:,bse:bse+batch_size]
                diff = np.subtract(y_bs, self.predict(x_bs))
                bse+=batch_size

                if learning.lower() == "filtered":
                    self.weights = (1-gamma)*self.weights + alpha*(np.dot(y_bs, x_bs.T))
                elif learning.lower() == "delta":
                    self.weights = self.weights + alpha*(np.dot(diff, x_bs.T))

                else:
                    self.weights = self.weights + alpha*(np.dot(self.predict(x_bs), x_bs.T))


    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        mean_squared_error = np.square(np.subtract(y,self.predict(X))).mean()
        return mean_squared_error
