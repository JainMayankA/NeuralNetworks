# Jain, Mayank Ambalal
# 1001-761-066
# 2020_10_25
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        if not self.weights:
            self.weights.append(tf.Variable(np.random.randn(self.input_dimension,num_nodes)))
        else:
            self.weights.append(tf.Variable(np.random.randn(len(self.weights[-1][1]),num_nodes)))
        self.biases.append(tf.Variable(np.random.randn(num_nodes)))
        self.activations.append(transfer_function.lower())

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights
        

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
         
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_hat))


    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        input_layer = X
        for i in range(len(self.weights)):
            predicted = tf.matmul(input_layer,self.weights[i])+(self.biases[i])
            if self.activations[i] == "linear":
                predicted = predicted
            elif self.activations[i] == "sigmoid":
                predicted = tf.nn.sigmoid(predicted)
            else:
                predicted = tf.nn.relu(predicted)
            input_layer = predicted
        return input_layer


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """  
        for i in range(num_epochs):
            bs = 0
            for j in range (X_train.shape[1]//batch_size):
                bs_x = X_train[bs:bs+batch_size]
                bs_y = y_train[bs:bs+batch_size]
                bs+=batch_size
                with tf.GradientTape() as tape:
                    pred = self.predict(bs_x)
                    loss = self.calculate_loss(bs_y, pred)
                    dloss_dw, dloss_db = tape.gradient(loss, [self.weights, self.biases])
                for l in range(len(self.weights)):
                    self.weights[l].assign_sub(alpha*dloss_dw[l])
                    self.biases[l].assign_sub(alpha*dloss_db[l])
        

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """ 
        preds = np.argmax(self.predict(X), axis = 1)
        err = np.sum(np.where(preds == y,0,1))
        return (err/len(preds))


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        x = np.argmax(self.predict(X),axis = 1)
        return tf.math.confusion_matrix(y,x)


