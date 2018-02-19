############################################################
##
## Restricted Boltzmann Machine for calculating ground
## wavefunction of fermionic quantum systems.
##
## By Heesoo Kim
##
## Updated: 02/19/2018
##
## IN PROGRESS
##
############################################################

import numpy as np
import tensorflow as tf
import os
import csv

class RBM(object):

    def __init__(self, num_visible, num_hidden, learning_rate, ID):

        self.num_visible = num_visible  # Number of visible units
        self.num_hidden = num_hidden  # Number of hidden units
        self.learning_rate = learning_rate

        self.ID = ID

        self.weights = None  # Neural network weights
        self.visible_bias = None # Neural network visible bias
        self.hidden_bias = None # Neural network hidden_bias

        self.visible_units = None
        self.hidden_units = None

        self.weights_history = None
        self.visible_bias_history = None
        self.hidden_bias_history = None


    def fit(self):

        _create_directories()
        _initialize_weights_biases()


    def _create_directories(self):

        dir_path = os.getcwd()
        if dir_path[-1] is not '/':
            dir_path = dir_path + '/'
        history_dir = '%shistorical_parameters' % dir_path
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            print 'Directory for storing histories has been created.'
        self.weights_history = '%s/weights_%s.csv' % (history_dir, self.ID)
        self.visible_bias_history = '%s/visible_bias_%s.csv' % (history_dir, self.ID)
        self.hidden_bias_history = '%s/hidden_bias_%s.csv' % (history_dir, self.ID)


    def _initialize_weights_biases_states(self):

        # Initialize weights to random values chosen from zero-mean Gaussian
        # distribution with a standard deviation of 0.01.

        # Number of weights are number of visible x number of hidden
        # Weights are a matrix of v x h
        # Weights[i][j] is the weights connecting ith visible to jth hidden
        self.weights = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=(self.num_visible, self.num_hidden))

        # Initialize bias parameters of both visible and hidden as 0
        self.visible_bias = np.zeros(self.num_visible, dtype=float)
        self.hidden_bias = np.zeros(self.num_hidden, dtype=float)

        # Initialize hidden states
        self.hidden_units = np.random.randint(2, size=self.num_hidden)


    def _training_step(self):

        initial_expectation = _get_expectation()
        _hidden_units_update()
        _visible_units_update()
        reconstructed_expectation = _get_expectation()
        _weights_update(initial_expectation, reconstructed_expectation)

        _save_parameters()


    def _save_parameters(self):

        with open(self.visible_bias_history, 'a') as file:
            file.writerow(self.visible_bias)
            file.close()
        with open(self.hidden_bias_history, 'a') as file:
            file.writerow(self.hidden_bias)
            file.close()


    def _hidden_units_update(self):

        new_hidden_units = np.zeros(self.num_hidden, dtype=float)

        for h in range(self.num_hidden):

            current_weights = self.weights[:, h]
            total_input = np.dot(
                current_weights, self.visible_units) + self.hidden_bias[h]
            probability = _sigmoid(total_input)
            if probability is not None:
                new_hidden_units[h] = np.random.binomial(n=1, p=probability)
            else:
                new_hidden_units[h] = self.hidden_units[h]

        self.hidden_units = new_hidden_units


    def _visible_units_update(self):

        new_visible_units = np.zeros(self.num_visible, dtype=float)

        for v in range(self.num_visible):

            current_weights = self.weights[v, :]
            total_input = np.dot(
                current_weights, self.hidden_units) + self.visible_bias[v]
            probability = _sigmoid(total_input)
            new_visible_units[v] = probability

        self.visible_units = new_visible_units


    def _sigmoid(self, total_input):

        try:
            return 1 / (1 + np.exp(-total_input))
        except:
            print 'Sigmoid function failed. Returning None.'
            return None


    def _weights_update(self, initial, recon):

        weights_update = self.learning_rate * (initial - recon)
        self.weights += weights_update


    def calculate_wavefunction(self, input_data):

        reshape = np.reshape(input_data, self.num_visible)

        weight_sum = 0
        for v in range (self.num_visible):
            for h in range (self.num_hidden):
                weight_sum += self.weights[v][h] * input_data[v] * hidden_units[h]

        visible_sum = np.dot(reshape, self.visible_bias)
        hidden_sum = np.dot(self.hidden_state, self.hidden_bias)

        return np.sum(np.exp(weight_sum + visible_sum + hidden_sum))



