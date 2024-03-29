
################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: CleanDriver_v1.py                                                    #
#  Description: A general network driver for training and testing a            #
#               network as defined in CleanNetwork.py                          #
#                                                                              #
################################################################################

import numpy as np
import math
from scipy import sparse
import WeightUpdate as wp
from CleanNetwork import Network

class Driver:
    """ Network driver. Instantiates and provides functionality for
        training and testing a network."""
    def __init__(self, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, alpha, dt,
                 sigma, nonlinearity, target_in_network=False):
        """
        Args:
            num_neurons : int - number of (hidden & observed) neurons in the network
            p : float - connectivity density used for initial network instantiation
            chaotic_constant : float - if > 1, induces chaotic behavior in RNN when run
                                        without training
            input_num : int - dimension of input vector
            output_num : int - number of observed neurons
            alpha : float - learning rate (usually set to 1)
            dt : float - time interval (used for evolving network)
            sigma : float - variance of stochastic noise applied to neurons
            nonlinearity : []float -> []float - f: R^n -> R^n, applies a non-linear
                                                  function (r, in paper) to neuron activations
            target_in_network : bool - is a behavioral readout node in the network?

        Returns:
            None
        """
        print('Initializing Driver...')

        self.num_neurons = num_neurons
        self.chaotic_constant = chaotic_constant
        self.input_num = input_num
        self.output_num = output_num
        self.p = p
        self.target_in_network = target_in_network

        print(" > > > > > Initializing network...")
        self.network = Network(num_neurons, p, chaotic_constant, input_num,
                output_num, gg_sparseness, dt, sigma)

        if target_in_network:
            self.network.connectivity_matrix[0:-1,-1] = 0
        print(" > > > > > Network initialized.")

        self.x = self.network.membrane_potential # x(0)
        # always make the appx of the cross-correlation matrix
        # dimension of the output num, not number of network neurons
        self.f = nonlinearity
        self.P = (1/alpha) * np.identity(num_neurons) # P(0)
        self.r = self.f(self.x)
        self.zs = np.random.randn()
        self.ws = np.zeros(num_neurons)

        ## Tracking
        self.signal1 = []
        self.signal2 = []
        self.ICs = [] # initial conditions
        self.track_ics = False

        print('Driver initialized.')

    """ Train self.network using FORCE/RLS"""
    def train(self, target, vInput):
        """
        Args:
            target : [][]float - target func for each neuron
            vInput : []float - sensory evidence + context cue

        Returns:
            None
        """
        vInput = np.array(vInput)
        for i in range(len(target)):
            # propagate network
            self.network.prop(self.r, vInput, target_in_network=False)

            # update r
            self.x = self.network.membrane_potential
            self.r = self.f(self.x)

            # update zs
            self.zs = np.dot(self.ws, self.r)

            # error slippage
            err_mat = self.zs - target[i]

            # update J
            Pr = np.dot(self.P, self.r)
            c = 1 / (1 + np.dot(self.r, Pr))
            self.P = self.P - c * np.outer(Pr, Pr)
            j_delta = c * err_mat * np.outer(np.ones(len(Pr)), Pr)
            self.ws = self.ws - (err_mat * c * Pr)

            # uddate internal connectivity matrix
            connect_matrix = self.network.connectivity_matrix
            new_connect_matrix = connect_matrix - j_delta

            # no target feedback
            if self.target_in_network:
                new_connect_matrix[0:-1,(self.output_num-1)] = 0
            self.network.connectivity_matrix = new_connect_matrix

    """ Test self.network (simulate runs with various sensory evidence combinations and
        context cues)"""
    def test(self, target, vInput, twn=False, scale=0):
        """
        Args:
            target : [][]float - target func for each neuron
            vInput : []float - sensory evidence + context cue
            twn : bool - test with noise applied to sensory evidence
            scale : int - variance of stochastic noise applied to sensory evidence

        Returns:
            output : [][]float - result of simulation
        """
        vInput = np.array(vInput)
        output = []
        output_x = np.zeros((len(target), self.num_neurons))
        for i in range(len(target)):

            # remove reference problem
            vdInput = np.copy(vInput)

            # test with noise, if applicable
            if twn:
                rand1, rand2 = scale * np.random.randn(), scale * np.random.randn()
                vdInput[0] = vInput[0] + rand1 # noisy signal
                vdInput[1] = vInput[1] + rand2 # noisy signal

            # track signals for later
            #self.signal1.append(vdInput[0])
            #self.signal2.append(vdInput[1])

            # propagate z through the network
            self.network.prop(self.r, vdInput, Conv=0, target_in_network=False) # remove noise
            self.x = self.network.membrane_potential

            if self.track_ics:
                self.ICs.append((self.x, vInput))

            # update r
            self.r = self.f(self.x)

            # calculate zs
            self.zs = np.dot(self.ws, self.r)
            output.append(self.zs) # for avg trials
            output_x[i] = self.x
        return output, output_x


"""
period = 400.0
time_constant = 0.1
num_neurons = 500
chaotic_constant = 1.5
input_num = 2
output_num = 500
gg_sparseness = 0.9
gz_sparseness = 0.9
fg_sparseness = 0.9
readout_sparseness = 0.1
g_gz = 0.2
alpha = 1.0
dt = 0.1
p = 0.1

# context vectors
sins = np.array([ 1, 0 ])
coss = np.array([ 0, 1 ])
con_vecs = [sins, coss]

# target functions
test_funcs = [
    [ ((1.3/1.0) * np.sin(np.pi * i/250) + \
                          (1.3/2.0) * np.sin(2 * np.pi * i/250) + \
                          (1.3/6.0) * np.sin(3 * np.pi * i/250) + \
                          (1.3/3.0) * np.sin(4 * np.pi * i/250)) / 1.5 \
                          for i in range(int(period / dt)) ],

    [ ((1.3/1.0) * np.cos(np.pi * i/250) + \
                          (1.3/2.0) * np.cos(2 * np.pi * i/250) + \
                          (1.3/6.0) * np.cos(3 * np.pi * i/250) + \
                          (1.3/3.0) * np.cos(4 * np.pi * i/250)) / 1.5 \
                          for i in range(int(period / dt)) ],
]

test_funcs_prime_1 =  [
    test_funcs[0] for i in range(500)
]

test_funcs_prime_2 = [
    test_funcs[1] for i in range(500)
]

tests = [test_funcs_prime_1, test_funcs_prime_2]

driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
              input_num, output_num, gg_sparseness, gz_sparseness,
              fg_sparseness, readout_sparseness, g_gz, alpha, dt,
              tests, con_vecs)
"""
