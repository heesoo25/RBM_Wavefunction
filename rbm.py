###############################################################################
##
## Restricted Boltzmann Machine implementation in Python for calculating the
## ground wavefunction of the Ising model.
##
## By Heesoo Kim
##
## Updated: 06/02/2018
##
## IN PROGRESS
##
## This code is written in Python 2.7
## This code will automatically create necessary directories to store data.
##
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import itertools
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
        number of hidden units for a given number n of latice
        size is some number between log(n) and ln(n).

    h
        The strength of the external magnetic field. This value
        can be any real number (positive, zero, or negative). If
        not provided, the default value is 10.

    record_history : Boolean
        Determines whether or not the historical data of visible
        unit biases, hidden unit biases, and network weights is
        saved. The data is saved for True and the data is not saved
        for False. If not provided, the default value is False.

    visualize : Boolean
        Determines whether or not the Ising model will be
        visualized and updated real time as the model trains. If
        not provided, the default value is False.

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
        
        rbm = RBM(100, 5, record_history=True, visualize=True)
        rbm.fit(5)
        print 'Visible bias weights: %s' % rbm.get_visible_bias()
        print 'Hidden bias weights: %s' % rbm.get_hidden_bias()
        print 'Network weights: %s' % rbm.get_network_weights()

    The output for the previous script will print three complex-valued
    matrices, the first one of size (100,), the second of size (5,) and the
    third of size (100, 5). The code will also generate an HDF5 file named
    'weights_ID.h5' in the folder /historical_parameters/date/.
    """


    def __init__(self, lattice_size, num_hidden, h=0, record_history=False,
                learning_rate=0.01, verbose=10, ID=''):
        """Initialize the class RBM."""
        self.lattice_size = lattice_size
        self.num_visible = lattice_size*lattice_size
        self.num_hidden = num_hidden
        self.lr = learning_rate

        self.ID = ID
        self.h = h
        self.energy = None  # Free energy of the system
        self.expectation = None

        self.hidden_prob = np.zeros(num_hidden)
        self.visible_prob = np.zeros(self.num_visible)

        self.visible_units = None  # shape = (spin_size,)
        self.hidden_units = None  # shape = (num_hidden_units,)

        self.spin_configs = None  # shape = (num configs, spin size)

        self.record_history = record_history
        self.weights = None  # shape = (spin_size, num_hidden_units)
        self.visible_bias = None # shape = (spin_size,)
        self.hidden_bias = None # shape = (num_hidden_units,)
        self.verbose = verbose


    def fit(self, num_train_steps, history_frequency=1):
        """Build directories, initialize variables, and train the model.

        Args:
            num_train_steps (int) : The total number of steps for which this
                model will be trained
            history_frequency (int) : The number of training steps every which
                the parameters will be saved.
        """
        if type(history_frequency) is not int:
            raise TypeError('history_frequency must be an integer.')
        if history_frequency < 0 or history_frequency == 0:
            raise ValueError('history_frequency must be a positive integer.')
        self._create_directories()
        self._initialize()

        for i in range(num_train_steps):
            # Single training step
            initial_energy = self.energy
            initial_expectation = self.expectation
            self._hidden_units_update()
            self._visible_units_update()
            self._weights_update()
            # Save parameters accordingly to the history frequency
            if self.record_history and i % history_frequency == 0:
                self._save_parameters()
            # Print the free energy of the system accordingly to the verbosity
            if i % self.verbose == 0:
                self.energy = self._current_free_energy()
                print 'The free energy of the system is %s.' % self.energy

    def _hidden_units_update(self):
        """Update the hidden units configuration.

        For the hth hidden unit, generate a probability by taking the sigmoid
        of the sum of the jth hidden bias with the dot product of the hth row
        of the weights and the visible units. Take the Poisson distribution to
        get the new value for the hidden unit.
        """
        new_hidden_units = np.zeros(self.num_hidden, dtype=int)
        for h in range(self.num_hidden):
            current_weights = self.weights[:, h]
            total_input = np.dot(current_weights, self.visible_units)\
                            + self.hidden_bias[h]
            probability = self._sigmoid(total_input)
            if probability is None or np.isnan(probability):
                new_hidden_units[h] = self.hidden_units[h]
                print 'Bad hidden units probability value.'
            else:
                new_hidden_units[h] = np.random.binomial(
                                        n=1, p=np.absolute(probability))
        self.hidden_units = new_hidden_units * 2 - np.ones(
            self.num_hidden, dtype=int)

    def _visible_units_update(self):
        """Update the visible units configuration.

        For the vth visible unit, generate a probability by taking the sigmoid
        of the sum of the vth visible bias with the dot product of the vth row
        of the weights and the hidden units. Take the Poisson distribution to
        get the new value for the visible unit.
        """
        new_visible_units = np.zeros(self.num_visible, dtype=int)
        for v in range(self.num_visible):
            current_weights = self.weights[v, :]
            total_input = np.dot(current_weights, self.hidden_units)\
                            + self.visible_bias[v]
            probability = self._sigmoid(total_input)
            if probability is None or np.isnan(probability):
                new_visible_units[v] = self.visible_units[v]
                print 'Bad visible units probability value.'
            else:
                new_visible_units[v] = np.random.binomial(
                                        n=1, p=np.absolute(probability))
        self.visible_units = new_visible_units * 2 - np.ones(
            self.num_visible, dtype=int)

    def _sigmoid(self, total_input):
        """Compute the sigmoid of the complex-valued scalar total_input.

        Args:
            total_input (complex) : The sigmoid function input.

        Returns:
            If the sigmoid function was successful, returns the result.
            Otherwise, returns None.
        """
        try:
            return 1 / (1 + np.exp(-total_input))
        except:
            print 'Sigmoid function failed. Returning None.'
            return None

    def _current_free_energy(self):
        """Compute the free energy of the Ising model by adding the nearest
        neighbor energies of the configuration under periodic boundary
        conditions and adding the free energy contribution of the external
        magnetic field.

        Returns:
            numpy array of shape (num_spin_configs)
        """
        wf = self._evaluate_wavefunction()
        wf_conj = self._evaluate_wavefunction(conj=True)
        hamiltonian = self._evaluate_hamiltonian()

        wf2 = np.multiply(wf, wf_conj)
        return np.dot(wf2, hamiltonian) / wf2.sum()

    def _evaluate_wavefunction(self, conj=False):
        """Evaluate the current value of the wavefunction for the current state
        of visible units (s_i), visible bias (a_i), hidden bias (b_j), and
        network weights (W_ij).
            WF = exp(sum_i[s_i * a_i]) * prod_j[F_j(S)]
        where
            F_j(S) = 2 cosh[b_j + sum_i(W_ij * s_i)]
        with the wavefunction evaluated for each of the possible spin
        configurations.

        Args:
            conj (bool) : Determines if the wavefunction will be evaluated for
                its conjugate or not.

        Returns:
            A NumPy ndarray of shape (num_spin_configs,)
        """
        if conj:
            hb = np.conj(self.hidden_bias)
            w = np.conj(self.weights)
            vb = np.conj(self.visible_bias)
        else:
            hb = self.hidden_bias
            w = self.weights
            vb = self.visible_bias
        fis = 2 * np.cosh(np.asarray([hb] * self.spin_configs.shape[0])\
                        + np.dot(self.spin_configs, w))
        exponential = np.exp(np.dot(self.spin_configs, vb))
        return np.multiply(exponential, np.prod(fis, axis=1))

    def _evaluate_hamiltonian(self):
        """Evaluates the Hamiltonian of every spin configurations.
            H = -J * sum_ij(s_i * s_j) - mu * h * sum_j(s_j)
        where J = spin interaction energy, mu = magnetic moment, h = external
        magnetic field strength, {s_i} = spins.

        Returns:
            A NumPy ndarray of shape (num_spin_configs,)
        """
        lattice = self.lattice_size
        config = self.spin_configs.reshape(
            self.spin_configs.shape[0], lattice, lattice)

        left = np.sum(
            np.multiply(config[:, 1:, :], config[:, :lattice-1, :]), axis=1)
        right = np.sum(
            np.multiply(config[:, :, 1:], config[:, :, :lattice - 1]), axis=1)
        up = np.sum(
            np.multiply(config[:, 0, :], config[:, lattice - 1, :]), axis=1)
        down = np.sum(
            np.multiply(config[:, :, 0], config[:, :, lattice - 1]), axis=1)

        energy = np.sum(left, axis=1) + np.sum(right, axis=1) + up + down
        spin_sum = np.sum(self.spin_configs, axis=1)
        return -energy - self.h * spin_sum

    def _weights_update(self):
        """Update the weights via stochastic gradient descent."""
        self.weights -= self.lr * self._weights_derivative('network')
        self.visible_bias -= self.lr * self._weights_derivative('visible')
        self.hidden_bias -= self.lr * self._weights_derivative('hidden')

    def _weights_derivative(self, weights_type):
        """Evaluates the derivative of the expectation value of the Hamiltonian
        with respect to the weights of the given type, where this partial
        derivative is of the form
            d<H>/dw = (<WF|WF>*d<WF|H|WF>/dw - <WF|H|WF>*d<WF|WF>/dw)/<WF|WF>^2 
        where WF = wavefunction, w = weights, H = Hamiltonian, and <H> =
        expectation of the Hamiltonian.

        Args:
            weights_type (str) : The type of weights with respect to which the
                partial derivative of the expectation value of the Hamiltonian
                will be evaluated.

        Returns:
            A NumPy ndarray of the same shape as that of the weights of the
            given type.
        """
        wf = self._evaluate_wavefunction()  # (num config,)
        wf_conj = self._evaluate_wavefunction(conj=True)  # (num config,)
        wf2 = np.multiply(wf, wf_conj)  # ()
        hamiltonian = self._evaluate_hamiltonian()  # (num config,)
        partial_hamiltonian = self._partial_hamiltonian(weights_type)  # weight
        partial_wf2 = self._partial_wf2(weights_type)  # weight

        if weights_type == 'network':
            ext_wf2 = np.asarray(
                [np.asarray([wf2] * self.visible_bias.shape[0])]
                * self.hidden_bias.shape[0]).T
            ext_h = np.asarray(
                [np.asarray([hamiltonian] * self.visible_bias.shape[0])]
                * self.hidden_bias.shape[0]).T
        elif weights_type == 'visible':
            ext_wf2 = np.asarray([wf2] * self.visible_bias.shape[0]).T
            ext_h = np.asarray([hamiltonian] * self.visible_bias.shape[0]).T
        elif weights_type == 'hidden':
            ext_wf2 = np.asarray([wf2] * self.hidden_bias.shape[0]).T
            ext_h = np.asarray([hamiltonian] * self.hidden_bias.shape[0]).T
        else:
            raise NotImplementedError('The given weights type is not valid.')
        ext_partial_h = np.asarray(
            [partial_hamiltonian] * self.spin_configs.shape[0])
        ext_partial_wf2 = np.asarray(
            [partial_wf2] * self.spin_configs.shape[0])
        term1 = np.multiply(ext_wf2, ext_partial_h).sum(axis=0)
        term2 = np.multiply(ext_partial_wf2, ext_h).sum(axis=0)
        return np.subtract(term1, term2) / (wf2.sum()**2)

    def _partial_wf2(self, weights_type):
        """Evaluates the partial derivative of <WF|WF> with respect to weights
        of given type.
            d<WF|WF>/dw = sum_{spin configs} [dWFC/dw * WF + WFC * dWF/dw]
        where WF = wavefunction and WFC = conjugate of the wavefunction.

        Args:
            weights_type (str) : The type of weights with respect to which the
                partial of the Hamiltonian will be evaluated.

        Returns:
            A NumPy ndarray of the same shape as that of the weights of given
            type.
        """
        partial_wf_conj = self._partial_wavefunction(weights_type, conj=True)
        partial_wf = self._partial_wavefunction(weights_type)
        wf_conj = self._evaluate_wavefunction(conj=True)
        wf = self._evaluate_wavefunction()
        return np.add(np.dot(partial_wf_conj, wf), np.dot(partial_wf, wf_conj))

    def _partial_hamiltonian(self, weights_type):
        """Evaluates the partial derivative of the expected value of the
        Hamiltonian with respect to weights of given type.
            d<WF|H|WF>/dw = sum_{spins configs} [
                                    dWFC/dw * H * WF + WFC * H * dWF/dw]
        where <WF|H|WF> = expected value of the Hamiltonian, H = Hamiltonian,
        WF = wavefunction, and WFC = conjugate of the wavefunction.

        Args:
            weights_type (str) : The type of weights with respect to which the
                partial of the Hamiltonian will be evaluated,

        Returns:
            A NumPy ndarray of the same shape as that of the given weights
            type.
        """
        partial_wf_conj = self._partial_wavefunction(weights_type, conj=True)
        partial_wf = self._partial_wavefunction(weights_type)
        hamiltonian = self._evaluate_hamiltonian()
        wf = self._evaluate_wavefunction()
        wf_conj = self._evaluate_wavefunction(conj=True)
        # term1 = sum_{spin configs} [dWFC/dw * WF]
        term1 = np.dot(partial_wf_conj, np.multiply(hamiltonian, wf))
        # term2 = sum_{spin configs} [WFC * dWF/dw]
        term2 = np.dot(partial_wf, np.multiply(wf_conj, hamiltonian))
        return np.add(term1, term2)

    def _partial_wavefunction(self, weights_type, conj=False):
        """ Evaluates the partial wavefunction with respect to weights of given
        type. For visible bias {a_i}, hidden bias {b_j}, spins {s_i}, network
        weights {W_ij}, and wavefunction WF, the partials are
            dWF/da_m = s_m * WF
            dWF/db_n = WF / F_n * 2 sinh [b_n + sum_k(W_kn * s_k)]
            dWF/dW_mn = s_m * dWF/db_n
        where F_n = 2 cosh [b_n + sum_k(W_kn * s_k)]

        Args:
            weights_type (str) : The type of weights (network, visible bias,
                and hidden bias) for which the partial wavefunction will be
                calculated.
            conj (bool) : Describes whether the partial wavefunction is taken
                for the conjugate wavefunction bra as opposed to ket.

        Returns:
            An NumPy ndarray of shape (weights_shape, num_spin_configs). The
            array values are of the partial wavefunction for each spin
            configurations wih respect to each weights unit.
        """
        if conj:
            wf = self._evaluate_wavefunction(conj=True)
        else:
            wf = self._evaluate_wavefunction()
        if weights_type == 'visible':
            # dWF/da_m = s_m * WF
            extended_wf = np.asarray([wf] * self.visible_bias.shape[0])
            extended_vunit = np.asarray(
                [self.visible_units] * self.spin_configs.shape[0]).T
            partial_vbias = np.multiply(extended_wf, extended_vunit)
            return partial_vbias  # shape = (visible, spin_config)

        hb = self.hidden_bias
        w = self.weights
        fis = 2 * np.cosh(
            np.asarray([hb] * self.spin_configs.shape[0]) + np.dot(
                self.spin_configs, w))  # shape = (spin_config, hidden)
        fis_prod = np.asarray([np.prod(fis, axis=1)] * hb.shape[0]).T
        fis_missing = np.divide(fis_prod, fis)  # shape = (spin_config, hidden)
        sinh = 2 * np.sinh(
            np.asarray([hb] * self.spin_configs.shape[0]) + np.dot(
                self.spin_configs, w)).diagonal()
        extended_sinh = np.asarray([sinh] * self.spin_configs.shape[0]).T
        # dWF/db_n = WF / F_n * 2 sinh [b_n + sum_k(W_kn * s_k)]
        partial_hbias = np.multiply(fis_missing.T, extended_sinh)

        if weights_type == 'hidden':
            return partial_hbias  # shape = (hidden, spin_config)
        elif weights_type == 'network':
            # dWF/dW_mn = s_m * dWF/db_n
            extended_partial_hbias = np.asarray(
                [partial_hbias] * self.visible_bias.shape[0])
            extended_vunit = np.asarray(
                [self.visible_units] * self.hidden_bias.shape[0])
            extended_vunit = np.asarray(
                [extended_vunit] * self.spin_configs.shape[0]).T
            partial_network = np.multiply(
                extended_partial_hbias, extended_vunit)
            return partial_network  # shape = (visible, hidden, spin_config)
        else:
            raise NotImplementedError('The entered weights type is not valid')
    
    def _create_directories(self):
        """Create appropriate directories for weights histories to be saved.
        The ID associated with the weights history files is generated using
        the current time in the form YYMMDD-HHMMSS. A single history file
        named 'weights_ID.h5' is generated per run of this model.
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

        The network weights is complex-valued matrix of shape (visible, hidden)
        with real parts sampled from a zero-mean Gaussian distribution with a
        standard deviation of 0.01, and with imaginary parts set to 0.
        The visible unit biases and the hidden unit biases are set to zeros of
        complex data type.
        The elements of the visible units and the hidden units are valued at 1
        or at -1 randomly sampled with equal probability.
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
        self.spin_configs = np.asarray(
            map(list, itertools.product([-1, 1], repeat=self.lattice_size**2)))
        print 'Spin configuration created.'
        self.energy = self._current_free_energy()
        print 'Initalization complete.'

    def _save_parameters(self):
        """Append the current network weights, visible unit bias weights, and
        hidden unit bias weights to their respective history file.
        """
        self.weights_history.append([self.weights])
        self.visible_bias_history.append([self.visible_bias])
        self.hidden_bias_history.append([self.hidden_bias])


rbm = RBM(4, 2, h=0, record_history=False, verbose=5)
rbm.fit(num_train_steps=25)

