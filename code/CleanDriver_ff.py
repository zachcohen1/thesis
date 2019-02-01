################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: InternallyRecurrentDriverMOandIO.py                                  #
#  Description: A general network driver. Network architecture 3               #
#                                                                              #
################################################################################

import numpy as np
import math
from scipy import sparse
import WeightUpdate as wp
from NoisyNetworkMOandIOC import Network

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
        # instantiate a nework
        self.network = Network(num_neurons, p, chaotic_constant, input_num,
                output_num, gg_sparseness, dt, sigma)
        self.D_network = Network(num_neurons, p, chaotic_constant, (input_num + output_num),
                1, gg_sparseness, dt, sigma)

        if target_in_network:
            self.network.connectivity_matrix[0:-1,-1] = 0
        print(" > > > > > Network initialized.")

        self.x = self.network.membrane_potential # x(0)
        self.Dx = self.D_network.membrane_potential
        self.ws = 2 * (np.random.rand(num_neurons, output_num) - 0.5)
        # always make the appx of the cross-correlation matrix
        # dimension of the output num, not number of network neurons
        self.P = (1/alpha) * np.identity(num_neurons) # P(0)
        self.f = nonlinearity
        self.r = self.f(self.x)
        self.Dr = self.f(self.Dx)
        self.zs = 0.5 * np.random.randn(len(self.r))

        ## Tracking
        self.signal1 = []
        self.signal2 = []
        self.ICs = [] # initial conditions
        self.track_ics = False

        print('Driver initialized.')

    """ Train self.network using full-FORCE/RLS"""
    def train(self, target, vInput):
        vInput = np.array(vInput)
        samps = np.zeros(len(target))
        for i in range(len(target[0])):
            # gather samples
            for w in range(len(target)):
                samps[w] = target[w][i]

            self.network.prop(self.r, vInput, target_in_network=True)
            self.D_network.prop(self.Dr, np.append(vInput, samps), target_in_network=True)

            # update r
            self.x = self.network.membrane_potential
            self.Dx = self.D_network.membrane_potential
            self.r = self.f(self.x)
            self.Dr = self.f(self.Dx)

            # error slippage
            self.zs = np.dot(self.network.connectivity_matrix, self.r)
            self.D_zs = np.dot(self.D_network.connectivity_matrix, self.Dr)
            err_mat = self.zs - self.D_zs - np.dot(
                self.D_network.input[self.input_num:self.input_num+self.output_num].T,
                samps)

            # update ws, dts
            #r_trim = self.r[:self.output_num]
            Pr = np.dot(self.P, self.r)
            c = 1 / (1 + np.dot(self.r, Pr))
            j_delta = c * np.outer(err_mat, Pr)
            self.P = self.P - c * np.outer(Pr, Pr)
            del_w = c * np.outer(np.dot(self.r, self.ws) - samps, hold)

            # uddate internal connectivity matrix
            connect_matrix = self.network.connectivity_matrix
            new_connect_matrix = connect_matrix - j_delta
            new_ws = self.ws - del_w.T

            if self.target_in_network:
                new_connect_matrix[0:-1,(self.output_num-1)] = 0
                new_ws[0:-1,(self.output_num-1)] = 0
            self.network.connectivity_matrix = new_connect_matrix
            self.ws = new_ws

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
        output = [[] for i in range(self.output_num)]
        for i in range(len(target[0])):

            # remove reference problem
            vdInput = [vInput[0], vInput[1], vInput[2], vInput[3], vInput[4]]

            # test with noise, if applicable
            if twn:
                rand1, rand2 = scale * np.random.randn(), scale * np.random.randn()
                vdInput[0] = vInput[0] + rand1 # noisy signal
                vdInput[1] = vInput[1] + rand2 # noisy signal

            # track signals for later
            self.signal1.append(vdInput[0])
            self.signal2.append(vdInput[1])

            # propagate z through the network
            self.network.prop(self.r, vdInput, Conv=0, target_in_network=True) # remove noise
            self.x = self.network.membrane_potential

            if self.track_ics:
                self.ICs.append((self.x, vInput))

            # update r
            self.r = self.f(self.x)

            # calculate zs
            self.zs = np.dot(self.r, self.ws)
            for w in range(self.output_num):
                output[w].append(self.zs[w]) # for avg trials
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
