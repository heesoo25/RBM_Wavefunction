##################################
##
## Restricted Boltzmann Machine implementation in Python to
## calculate ground wavefunction of fermionic quantum
## systems
##
## By Heesoo Kim
##
## Updated: 03/11/2018
##
## IN PROGRESS
##
############################################################

import numpy as np
import tables
import os
import time

class RBM(object):

    def __init__(self, lattice_size, num_hidden, h=10, history=False,
        learning_rate=0.01, ID=''):

        self.lattice_size = lattice_size
        self.num_visible = lattice_size * lattice_size  # Number of visible units
        self.num_hidden = num_hidden  # Number of hidden units
        self.learning_rate = learning_rate

        self.ID = ID
        self.h = h

        self.history = history

        self.weights = None  # Neural network weights
        self.visible_bias = None # Neural network visible bias
        self.hidden_bias = None # Neural network hidden_bias

        self.energy = None

        self.visible_units = None
        self.hidden_units = None

        self.weights_history = None
        self.visible_bias_history = None
        self.hidden_bias_history = None


    def fit(self, num_train_steps, history_frequency=1):
        if type(history_frequency) is not int:
            raise TypeError('history_frequency must be an integer.')
        if history_frequency < 0 or history_frequency == 0:
            raise ValueError('history_frequency must be a positive integer.')
        self._create_directories()
        self._initialize()
        self._train(num_train_steps, history_frequency)


    def _train(self, num_train_steps, history_frequency):
        reps = int(num_train_steps / history_frequency)
        for _ in range(reps):
            for _ in range(history_frequency):
                self._training_step()
            if self.history:
                self._save_parameters()
            print 'The free energy of the system is: %s' % self.energy


    def _create_directories(self):

        time_now = time.strftime('%y, %m, %d, %H, %M, %S').split(', ')
        date = ''.join(time_now[:3])
        self.ID += '%s-%s' % (date, ''.join(time_now[3:]))

        dir_path = os.getcwd()
        if dir_path[-1] is not '/':
            dir_path = dir_path + '/'
        history_dir = '%shistorical_parameters' % dir_path
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            print 'Directory for storing histories has been created.'
        today_dir = '%s/%s' % (history_dir, date)
        if not os.path.exists(today_dir):
            os.makedirs(today_dir)
            print 'Directory for storing today\'s histories has been created.'

        weights_dir = '%s/weights_%s.h5' % (today_dir, self.ID)
        weights_file = tables.open_file(
            weights_dir,
            mode='w',
            title='Weights and Biases History')
        weights_atom = tables.FloatAtom(shape=(self.num_visible, self.num_hidden))
        visible_bias_atom = tables.FloatAtom(shape=(self.num_visible,))
        hidden_bias_atom = tables.FloatAtom(shape=(self.num_hidden,))
        self.weights_history = weights_file.create_earray(
            weights_file.root,
            'weights',
            title='Historical weights for trial %s' % self.ID,
            atom=weights_atom,
            shape=(0,))
        self.visible_bias_history = weights_file.create_earray(
            weights_file.root,
            'visible_bias',
            title='Historical visible biases for trial %s' % self.ID,
            atom=visible_bias_atom,
            shape=(0,))
        self.hidden_bias_history = weights_file.create_earray(
            weights_file.root,
            'hidden_bias',
            title='Historical hidden biases for trial %s' % self.ID,
            atom=hidden_bias_atom,
            shape=(0,))


    def _initialize(self):

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
        self.hidden_units = 2 * np.random.randint(2, size=self.num_hidden) - \
            np.ones(self.num_hidden)
        self.visible_units = 2 * np.random.randint(2, size=self.num_visible) - \
            np.ones(self.num_visible)

        print self.hidden_units

        self.energy = self._current_free_energy()


    def _training_step(self):

        initial_energy = self.energy
        self._hidden_units_update()
        self._visible_units_update()
        self.energy = self._current_free_energy()
        self._weights_update(initial_energy, self.energy)


    def _save_parameters(self):

        self.weights_history.append([self.weights])
        self.visible_bias_history.append([self.visible_bias])
        self.hidden_bias_history.append([self.hidden_bias])


    def _hidden_units_update(self):

        new_hidden_units = np.zeros(self.num_hidden, dtype=float)

        for h in range(self.num_hidden):

            current_weights = self.weights[:, h]
            total_input = np.dot(
                current_weights, self.visible_units) + self.hidden_bias[h]
            probability = self._sigmoid(total_input)
            if probability is not None:
                new_hidden_units[h] = np.random.binomial(n=1, p=probability)
            else:
                new_hidden_units[h] = self.hidden_units[h]

        self.hidden_units = new_hidden_units * 2 - np.ones(
            self.num_hidden, dtype=float)
        print self.hidden_units


    def _visible_units_update(self):

        new_visible_units = np.zeros(self.num_visible, dtype=float)

        for v in range(self.num_visible):

            current_weights = self.weights[v, :]
            total_input = np.dot(
                current_weights, self.hidden_units) + self.visible_bias[v]
            probability = self._sigmoid(total_input)
            if probability is not None:
                new_visible_units[v] = np.random.binomial(n=1, p=probability)
            else:
                new_visible_units[v] = self.visible_units[v]

        self.visible_units = new_visible_units * 2 - np.ones(
            self.num_visible, dtype=float)


    def _sigmoid(self, total_input):

        try:
            return 1 / (1 + np.exp(-total_input))
        except:
            print 'Sigmoid function failed. Returning None.'
            return None


    def _weights_update(self, initial, recon):

        weights_update = self.learning_rate * (initial - recon)
        self.weights += weights_update


    def _current_free_energy(self):
        lattice = self.lattice_size
        config = self.visible_units.reshape((lattice, lattice))
        energy = np.sum(np.multiply(config[1:, :], config[:lattice - 1, :])) + \
            np.sum(np.multiply(config[:, 1:], config[:, :lattice - 1])) + \
            np.sum(np.multiply(config[0, :], config[lattice - 1, :])) + \
            np.sum(np.multiply(config[:, 0], config[:, lattice - 1]))
        return -energy - self.h * np.sum(self.visible_units)

    
    def get_visible_bias(self):
        return self.visible_bias


    def get_hidden_bias(self):
        return self.hidden_bias


    def get_weights(self):
        return self.weights


rbm = RBM(70, 2, history=False)
rbm.fit(10)
