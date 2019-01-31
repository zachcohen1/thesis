
################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: CleanNetwork.py                                                      #
#  Description: A network model, written in accordance with the network        #
#               equations presented in Sussillo et. al., 2009.                 #
#                                                                              #
################################################################################

import numpy as np
import scipy
from scipy import sparse
from scipy import stats

class Network:
    """ A recurrent neural network class"""
    def __init__(self, num_neurons, p, chaotic_constant, input_num, output_num,
            gg_sparseness, dt, sigma):
        """
        Args:
            num_neurons : int - number of neurons in the network (hidden and observed)
            p : float - constant dictating starting strength of neural connections
            chaotic_constant : float - recurrent gain strength; >1 leads to chaotic activity
            input_num : int - number of sensory inputs to network
            output_num : int - number of observed neurons
            gg_sparseness : float - sparsity of network connections (for initialization)
            dt : float - timestep of network
            sigma : float - variance of stochastic noise applied to neurons

        Returns:
            None
        """
        self.num_neurons = num_neurons
        self.chaotic_constant = chaotic_constant
        self.membrane_potential = 0.5 * np.random.randn(num_neurons)
        self.input_num = input_num
        self.output_num = output_num
        scale = 1.0 / np.sqrt(p * num_neurons)
        self.connectivity_matrix = scale * chaotic_constant * \
            scipy.sparse.random(num_neurons, num_neurons,
            density=(1-gg_sparseness), data_rvs = np.random.randn).toarray()
        self.input = np.random.randn(num_neurons, input_num)
        self.dt = dt
        self.sigma = sigma

    """ Advance the network one timestep."""
    def prop(self, r, context, Conv=1, target_in_network=False):
        """
        Args:
            r : []float - firing rates of each neuron
            context : []float - sensory information + context cue
            Conv : int - scale stochastic noise (primarily used to turn off/on noise)
            target_in_network : bool - is there a behavioral target node?

        Returns:
            None
        """
        epsilon = np.random.randn(self.num_neurons) * Conv
        inp_vec = np.dot(self.input, context)
        if target_in_network:
            inp_vec[-1] = 0
        self.membrane_potential = (1-self.dt) * self.membrane_potential + \
            np.matmul(self.connectivity_matrix, (r * self.dt)) + \
            inp_vec + self.sigma * epsilon * self.dt

"""Start with 100 neurons. membrane_potential[i] is the membrane potential of
   neuron i.
# network = Network(0.001, 1000, 1.7, 10, 10, 0.9, 0.9, 0.9, 0.1, 0.2)
# network.prop(0.009)
# print(type(network.connectivity_matrix))
# print(network.membrane_potential[:]) """
