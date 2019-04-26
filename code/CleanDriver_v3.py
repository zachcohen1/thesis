
################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: CleanDriver_v2.py                                                    #
#  Description: A general network driver for training and testing a            #
#               network as defined in CleanNetwork.py                          #
#                                                                              #
################################################################################

import numpy as np
import math
from scipy import sparse
from CleanNetwork import Network

class Driver:
    """ Network driver. Instantiates and provides functionality for
        training and testing a network."""
    def __init__(self, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, alpha, dt,
                 sigma, nonlinearity, target_in_network=False,
                 sparse_input_hidden=True):
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
                output_num, gg_sparseness, dt, sigma, sparse_inp_hidden=sparse_input_hidden)

        if target_in_network:
            self.network.connectivity_matrix[0:-1,-1] = 0
        print(" > > > > > Network initialized.")

        self.x = self.network.membrane_potential # x(0)
        # always make the appx of the cross-correlation matrix
        # dimension of the output num, not number of network neurons
        self.P = (1/alpha) * np.identity(num_neurons) # P(0)
        self.f = nonlinearity
        self.r = self.f(self.x)
        self.zs = 0.5 * np.random.randn(len(self.r))

        ## Tracking
        self.signal1 = []
        self.signal2 = []
        self.ICs = [] # initial conditions
        self.track_ics = False

        print('Driver initialized.')

    """ Train self.network using FORCE/RLS"""
    def train(self, target, vInput, twn=False, scale=0):
        """
        Args:
            target : [][]float - target func for each neuron
            vInput : []float - sensory evidence + context cue

        Returns:
            None
        """
        vInput = np.array(vInput)
        samps = np.zeros(len(target))
        for i in range(len(target[0])):
            # remove reference problem
            vdInput = np.copy(vInput)

            # test with noise, if applicable
            if twn:
                rand1, rand2 = scale * np.random.randn(), scale * np.random.randn()
                vdInput[0] = vInput[0] + rand1 # noisy signal
                vdInput[1] = vInput[1] + rand2 # noisy signal

            r = self.r  # record r(t-1)
            # propagate network
            self.network.prop(self.r, vdInput, target_in_network=True)

            # update r
            self.x = self.network.membrane_potential
            self.r = self.f(self.x)

            for w in range(len(target)):
                samps[w] = target[w][i]

            # add nans to hidden population
            nans = np.where(np.isnan(samps))

            # error slippage
            err_mat = samps - self.x[:self.output_num]
            err_mat[-1] = samps[-1] - \
                np.dot(self.network.connectivity_matrix[self.output_num - 1], self.r)

            # update P using r(t-1)
            Pr = np.dot(self.P, r)
            c = 1 / (1 + np.dot(r, Pr))
            self.P = self.P - c * np.outer(Pr, Pr)
            j_delta = c * np.outer(err_mat, Pr)
            if len(nans) > 0:
                j_delta[nans] = 0 # add nan neurons to hidden population

            # uddate internal connectivity matrix
            new_connect_matrix = self.network.connectivity_matrix + np.lib.pad(j_delta,
                ((0, self.num_neurons - self.output_num), (0, 0)), 'constant', constant_values=0)

            # no target feedback
            if self.target_in_network:
                new_connect_matrix[:(self.output_num - 1), (self.output_num - 1)] = 0
                new_connect_matrix[self.output_num:, (self.output_num - 1)] = 0
            self.network.connectivity_matrix = new_connect_matrix


    """ Test self.network (simulate runs with various sensory evidence combinations and
        context cues)"""
    def test(self, target, vInput, twn=False, scale=0, hidden=False):
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
        output = [[] for i in range(self.output_num)]
        output_x = [[] for i in range(self.output_num)]
        if hidden:
            output_hidden = [[] for i in range(self.num_neurons - self.output_num)]
            output_hidden_x = [[] for i in range(self.num_neurons - self.output_num)]
        for i in range(len(target[0])):
            # remove reference problem
            vdInput = np.copy(vInput)

            # test with noise, if applicable
            if twn:
                rand1, rand2 = scale * np.random.randn(), scale * np.random.randn()
                vdInput[0] = vInput[0] + rand1 # noisy signal
                vdInput[1] = vInput[1] + rand2 # noisy signal

            # track signals for later
            self.signal1.append(vdInput[0])
            self.signal2.append(vdInput[1])

            # propagate the network
            self.network.prop(self.r, vdInput, Conv=0, target_in_network=True)

            self.x = self.network.membrane_potential
            if self.track_ics:
                self.ICs.append((self.x, vInput))

            # update r
            self.r = self.f(self.x)

            for w in range(self.output_num - 1):
                if np.isnan(target[w][i]):
                    output[w].append(np.nan)
                    output_x[w].append(np.nan)
                else:
                    output[w].append(self.r[w])
                    output_x[w].append(self.x[w])
            output[-1].append(np.dot(self.network.connectivity_matrix[self.output_num - 1], self.r))

            if hidden:
                for k in range(self.num_neurons - self.output_num):
                    output_hidden[k].append(self.r[self.output_num + k])
                    output_hidden_x[k].append(self.x[self.output_num + k])
        if hidden:
            return output, output_hidden, output_x, output_hidden_x
        else:
            return output


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
