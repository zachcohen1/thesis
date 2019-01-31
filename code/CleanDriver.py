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
    """ Network driver. Run the network and see what happens. """
    def __init__(self, time_constant, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, gz_sparseness,
                 fg_sparseness, readout_sparseness, g_gz, alpha, dt,
                 sigma, target_in_network=False):

        print('Initializing Driver...')

        # simulation time constant
        self.time_constant = time_constant
        # number of neurons
        self.num_neurons = num_neurons
        # chaotic constant (>1.0)
        self.chaotic_constant = chaotic_constant
        # number of input neurons
        self.input_num = input_num
        # number of output nerons
        self.output_num = output_num
        # inter-neuronal matrix sparesness
        self.gg_sparseness = gg_sparseness
        # output connectivity matrix sparesness
        self.gz_sparseness = gz_sparseness
        # input connectivity matrix sparesness
        self.fg_sparseness = fg_sparseness
        # w-sparseness
        self.readout_sparseness = readout_sparseness
        # output scaling factor
        self.g_gz = g_gz
        # connectivity probability used for shaping
        self.p = p
        # for training
        self.target_in_network = target_in_network

        print(" > > > > > Initializing network...")
        # instantiate a nework
        self.network = Network(time_constant, num_neurons, p,
                               chaotic_constant, input_num, output_num,
                               gg_sparseness, gz_sparseness, fg_sparseness,
                               readout_sparseness, g_gz, dt,
                               output_num, sigma)

        if target_in_network:
            self.network.connectivity_matrix[0:-1,-1] = 0
        print(" > > > > > Network initialized.")
 
        self.x = self.network.membrane_potential # x(0)
        self.P = (1/alpha) * np.identity(output_num) # P(0)
        self.r = np.tanh(self.x)
        self.zs = 0.5 * np.random.randn(self.output_num)

        # samples
        #self.samps = np.array([0 for i in range(self.num_target_funcs)])

        # output vectors
        # self.ws = np.array([ np.zeros(num_neurons) for i in range(self.num_target_funcs) ])


        ## Tracking
        self.signal1 = []
        self.signal2 = []
        self.ICs = [] # initial conditions
        self.track_ics = False

        print('Driver initialized.')

    # ---------------------- Train the network -------------------------- #
    def train(self, target, vInput):

        vInput = np.array(vInput)
        for i in range(len(target[0])):

            # propagate z through the network
            self.network.prop(self.zs, self.r, vInput, target_in_network=True)

            # update r
            self.x = self.network.membrane_potential
            self.r = np.tanh(self.x)

            # update zs
            self.zs = np.dot(self.network.connectivity_matrix, self.r)
            samps = np.zeros(len(target))
            for w in range(len(target)):
                 samps[w] = target[w][i]

            # adjust z node
            if self.target_in_network:
                self.network.z_p(self.zs[-1], np.dot(self.network.connectivity_matrix[-1], self.r))

            # error slippage
            err_mat = (self.zs - samps)

            # update ws, dts
            c = 1 / (1 + np.dot(self.r, np.dot(self.P, self.r)))
            hold = np.dot(np.transpose(self.P), self.r)
            j_delta = np.array([ c * -err_mat[i] * hold for i in range(len(self.r)) ])
            self.P = wp.P_t(self.output_num, self.P, self.r)

            # uddate internal connectivity matrix
            connect_matrix = self.network.connectivity_matrix
            new_connect_matrix = connect_matrix + j_delta

            # no target feedback
            if self.target_in_network:
                new_connect_matrix[0:-1,-1] = 0
            self.network.connectivity_matrix = new_connect_matrix


    # ----------------------- Test the network -------------------------- #
    def test(self, target, vInput, twn=False, scale=0):
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
            self.network.prop(self.zs, self.r, vdInput, Conv=0, target_in_network=True) # remove noise

            # adjust z node
            # if self.target_in_network:
            #     self.network.z_p(self.zs[-1], np.dot(self.ws[-1], self.r))
            self.x = self.network.membrane_potential

            if self.track_ics:
                self.ICs.append((self.x, vInput))

            # update r
            self.r = np.tanh(self.x)

            # calculate zs
            self.zs = np.dot(self.network.connectivity_matrix, self.r)
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
