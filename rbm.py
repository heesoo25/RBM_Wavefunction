###############################################################################
##
## Restricted Boltzmann Machine implementation in Python for calculating the
## ground wavefunction of fermionic quantum systems
##
## By Heesoo Kim
##
## Updated: 03/11/2018
##
## IN PROGRESS
##
## This code is written in Python 2.7
## This code will automatically create necessary directories to store data.
##
###############################################################################

import numpy as np
import tables
import os
import time


class RBM(object):
    """This class constructs a Restricted Boltzmann Machine configured to
    optimize the network weights to minimize the free energy of the 2D Ising
    Model. As the energy function is a stand-alone method in this class, it
    can be substituted with various other energy functions associated with
    different quantum systems.

    Parameters
    ----------
    lattice_size : int
        The square lattice size of the Ising model. The number of
        visible units is calculated by squaring the latice_size.

    num_hidden : int
        The number of hidden units in the Restricted Boltzmann
        Machine. The general rule of the thumb for choosing the
        number of hidden units for a given number n of visible
        units is some number between log(n) and ln(n).

    h
        Strength of the external magnetic field. This value can
        be any real number (positive, zero, or negative). If not
        provided, the default value is 10.

    record_history : Boolean
        Determines whether or not the historical data of visible
        unit biases, hidden unit biases, and network weights is
        saved. The data is saved for True and the data is not saved
        for False. If not provided, the default value is False.

    learning_rate : float
        The learning rate of the weights update. A larger
        learning_rate results in a large update increments for the
        weights, and a smaller learning rate results in a smaller
        update increments for the weights. If not provided, the
        default value is 0.01.

    ID : str
        If desired, more identifiers can be provided for a given
        configuration of the model. Must contain characters
        appropriate for a file path. If not provided, the default
        value is empty. During initializiation method, this value
        will be concatenated with a string describing the date and
        the time that the model is run.

    Examples
    --------
    See below for a small example of the use of the 'RBM' class.
        
        from nqs_rbm import RBM
        
        rbm = RBM(100, 5, record_history=True)
        rbm.fit(5)
        print 'Visible bias weights: %s' % rbm.get_visible_bias()
        print 'Hidden bias weights: %s' % rbm.get_hidden_bias()
        print 'Network weights: %s' % rbm.get_network_weights()

    The output for the previous script will print three complex-valued
    matrices, the first one of size (100,), the second of size (5,) and the
    third of size (100, 5). The code will also generate an HDF5 file named
    'weights_ID.h5' in the folder /historical_parameters/date/.
    """


    def __init__(self, lattice_size, num_hidden, h=10, record_history=False,
                 learning_rate=0.01, ID=''):
        """Initialize the class RBM."""
        self.lattice_size = lattice_size
        self.num_visible = lattice_size*lattice_size
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate

        self.ID = ID
        self.h = h
        self.energy = None  # Free energy of the system

        self.record_history = record_history
        self.weights = None  # Neural network weights
        self.visible_bias = None # Neural network visible bias
        self.hidden_bias = None # Neural network hidden_bias

        self.visible_units = None
        self.hidden_units = None


    def fit(self, num_train_steps, history_frequency=1):
        """Fit the model by creating appropriate directories, initialize
        variables associated with this model, and train the model.
        """
        if type(history_frequency) is not int:
            raise TypeError('history_frequency must be an integer.')
        if history_frequency < 0 or history_frequency == 0:
            raise ValueError('history_frequency must be a positive integer.')
        self._create_directories()
        self._initialize()
        self._train(num_train_steps, history_frequency)


    def _train(self, num_train_steps, history_frequency):
        """Train the model by taking a given number of training steps.
        If the model is set to record history, save the weights history
        accoriding to the given history frequency. Print the free energy
        of the system as the model trains.
        """
        reps = int(num_train_steps/history_frequency)
        for _ in range(reps):
            for _ in range(history_frequency):
                self._training_step()
            if self.record_history:
                self._save_parameters()
            print 'The free energy of the system is: %s' % self.energy

    def _training_step(self):
        """Update the hidden units and the visible units configurations, and
        the network weights.
        """
        initial_energy = self.energy
        self._hidden_units_update()
        self._visible_units_update()
        self.energy = self._current_free_energy()
        self.weights += self.learning_rate * (initial_energy - self.energy)

    def _hidden_units_update(self):
        """Update the hidden units configuration.

        For the hth hidden unit, generate a probability by taking the
        sigmoid of the sum of the jth hidden bias with the dot product
        of the hth row of the weights and the visible units. Take the
        Poisson distribution to get the new value for the hidden unit.
        """
        new_hidden_units = np.zeros(self.num_hidden, dtype=int)
        for h in range(self.num_hidden):
            current_weights = self.weights[:, h]
            total_input = np.dot(current_weights, self.visible_units)\
                            + self.hidden_bias[h]
            probability = self._sigmoid(total_input)
            if probability is not None:
                new_hidden_units[h] = np.random.binomial(
                                        n=1, p=np.absolute(probability))
            else:
                new_hidden_units[h] = self.hidden_units[h]
        self.hidden_units = new_hidden_units * 2 - np.ones(
            self.num_hidden, dtype=int)

    def _visible_units_update(self):
        """Update the visible units configuration.

        For the vth visible unit, generate a probability by taking the
        sigmoid of the sum of the vth visible bias with the dot product
        of the vth row of the weights and the hidden units. Take the
        Poisson distribution to get the new value for the visible unit.
        """
        new_visible_units = np.zeros(self.num_visible, dtype=int)
        for v in range(self.num_visible):
            current_weights = self.weights[v, :]
            total_input = np.dot(current_weights, self.hidden_units)\
                            + self.visible_bias[v]
            probability = self._sigmoid(total_input)
            if probability is not None:
                new_visible_units[v] = np.random.binomial(
                                        n=1, p=np.absolute(probability))
            else:
                new_visible_units[v] = self.visible_units[v]
        self.visible_units = new_visible_units * 2 - np.ones(
            self.num_visible, dtype=int)

    def _sigmoid(self, total_input):
        """Compute the sigmoid of the complex-valued scalar total_input."""
        try:
            return 1 / (1 + np.exp(-total_input))
        except:
            print 'Sigmoid function failed. Returning None.'
            return None


    def _current_free_energy(self):
        """Compute the free energy of the Ising model by adding the
        nearest neighbor energies of the configuration under periodic
        boundary conditions and adding the free energy contribution of
        the external magnetic field.
        """
        lattice = self.lattice_size
        config = self.visible_units.reshape((lattice, lattice))
        energy = np.sum(np.multiply(config[1:, :], config[:lattice - 1, :]))\
            + np.sum(np.multiply(config[:, 1:], config[:, :lattice - 1]))\
            + np.sum(np.multiply(config[0, :], config[lattice - 1, :]))\
            + np.sum(np.multiply(config[:, 0], config[:, lattice - 1]))
        return -energy - self.h * np.sum(self.visible_units)

    
    def _create_directories(self):
        """Create appropriate directories for weights histories to be
        saved. The ID associated with the weights history files is
        generated using the current time in the form YYMMDD-HHMMSS. A
        single history file weights_ID.h5 is generated per run of this
        model.
        """
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

        weights_file = tables.open_file(
            '%s/weights_%s.h5' % (today_dir, self.ID),
            mode='w',
            title='Weights and Biases History')

        weights_atom = tables.ComplexAtom(
            itemsize=16, shape=(self.num_visible, self.num_hidden))
        visible_bias_atom = tables.ComplexAtom(
            itemsize=16, shape=(self.num_visible,))
        hidden_bias_atom = tables.ComplexAtom(
            itemsize=16, shape=(self.num_hidden,))

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
        """Initialize network weights and record the free energy of the
        initial configuration of the visible units.

        The network weights is complex-valued matrix of shape (number
        of visible units, number of hidden units) with real parts
        sampled from a zero-mean Gaussian distribution with a standard
        deviation of 0.01, and with imaginary parts set to 0.
        The visible unit baises and the hidden unit biases are set to
        zeros of complex data type.
        The elements of the visible units and the hidden units are
        valued at 1 or -1 randomly sampled with equal probability.
        """
        self.weights = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=(self.num_visible, self.num_hidden)).astype(complex)
        self.visible_bias = np.zeros(self.num_visible, dtype=complex)
        self.hidden_bias = np.zeros(self.num_hidden, dtype=complex)

        self.hidden_units = 2 * np.random.randint(2, size=self.num_hidden)\
                            - np.ones(self.num_hidden, dtype=int)
        self.visible_units = 2 * np.random.randint(2, size=self.num_visible)\
                            - np.ones(self.num_visible, dtype=int)
        self.energy = self._current_free_energy()

    def _save_parameters(self):
        """Append the current network weights, visible unit bias
        weights, and hidden unit bias weights to their respective
        history file.
        """
        self.weights_history.append([self.weights])
        self.visible_bias_history.append([self.visible_bias])
        self.hidden_bias_history.append([self.hidden_bias])


    def get_visible_bias(self):
        """Return the current visible unit bias weights."""
        return self.visible_bias

    def get_hidden_bias(self):
        """Return the current hidden unit bias weights."""
        return self.hidden_bias

    def get_network_weights(self):
        """Return the current network weights."""
        return self.weights



rbm = RBM(70, 2, record_history=True)
rbm.fit(5)

