################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: ffInternallyRecurrentDriverMOandIO.py                                #
#  Description: fullFORCE training                                             #
#                                                                              #
################################################################################

import numpy as np
import math
from scipy import sparse
import WeightUpdate as wp
from NoisyNetworkMOandIOC import NoisyNetworkMOIOC
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as pca
import time
plt.style.use('seaborn-paper')

class Driver:
    """ Network driver. Run the network and see what happens. """
    def __init__(self, period, time_constant, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, gz_sparseness,
                 fg_sparseness, readout_sparseness, g_gz, alpha, dt,
                 target_functions, sigma):

        # length of simulation
        self.period = period

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

        print("Initializing network...")
        start = time.time()

        # instantiate a stationary network
        self.stat_network = NoisyNetworkMOIOC(time_constant, num_neurons, p,
                               chaotic_constant, input_num, output_num,
                               gg_sparseness, gz_sparseness, fg_sparseness,
                               readout_sparseness, g_gz, dt,
                               len(target_functions), sigma)

        # instantiate a target generating network
        self.targ_network = NoisyNetworkMOIOC(time_constant, num_neurons, p,
                               chaotic_constant, input_num, output_num,
                               gg_sparseness, gz_sparseness, fg_sparseness,
                               readout_sparseness, g_gz, dt,
                               len(target_functions), sigma)

        # target functions
        self.functions = target_functions

        # number of target functions
        self.num_target_funcs = len(target_functions)

        # alpha scale
        self.alpha = alpha

        # static network
        self.stat_x = self.network.membrane_potential # x(0)
        self.stat_P = (1/alpha) * np.identity(output_num) # P(0)
        self.stat_r = np.tanh(self.stat_x)

        # target network
        self.targ_x = self.network.membrane_potential # x(0)
        self.targ_P = (1/alpha) * np.identity(output_num) # P(0)
        self.targ_r = np.tanh(self.targ_x)

        # plot
        self.targets = [ [] for i in range(self.num_target_funcs) ]
        # plot the output of the network during training
        self.neuron_one_train = []
        # plot the output of the network during testing (w generated)
        self.neuron = [ [] for i in range(self.num_target_funcs) ]

        # computer learning error
        self.errors = [ [] for i in range(self.num_target_funcs) ]

        # ds
        self.dt_s = [ [] for i in range(self.num_target_funcs) ]

        # samples
        self.samps = np.array([0 for i in range(self.num_target_funcs)])

        # output vectors
        self.ws = [ np.zeros(num_neurons) for i in range(self.num_target_funcs) ]
        self.stat_ws = [ np.zeros(num_neurons) for i in range(self.num_target_funcs) ]
        self.stat_h = [ np.zeros(num_neurons) for i in range(self.num_target_funcs) ]

        # zs
        zs = [ 0.5 * np.random.randn() for i in range(self.num_target_funcs) ]
        self.stat_zs = np.array(zs)

        zs1 = [ 0.5 * np.random.randn() for i in range(self.num_target_funcs) ]
        self.targ_zs = np.array(zs1)

        end = time.time()
        print("Time ellapsed in network instantiation: ", end - start)

        self.T = int(period/dt)

        # x coordinates
        self.xs = np.arange(3000)

        # signal
        self.signal = []

    # ---------------------- Train the network -------------------------- #
    def train(self, target, vInput, vHint):
        # print("Beginning training...")
        # start = time.time()
        vInput = vInput
        for i in range(len(target[0])):

            # propagate z through both networks
            self.stat_network.prop(self.stat_zs, self.stat_r, vInput)
            self.targ_network.prop(self.targ_zs, self.targ_r, vInput)

            # update r
            self.stat_x = self.stat_network.membrane_potential
            self.stat_r = np.tanh(self.stat_x)

            self.targ_x = self.targ_network.membrane_potential
            self.targ_r = np.tanh(self.targ_x)

            # update zs
            for w in range(self.num_target_funcs):
                self.targ_zs[w] = np.dot(np.transpose(self.ws[w]), self.targ_r)
                self.samps[w] = target[w][i]

            # error slippage
            err_mat = (self.zs - self.samps)

            # update ws, dts
            c = wp.c(self.targ_r, self.P)

            for j in range(self.num_target_funcs):
                self.ws[j] = wp.w_t(self.output_num, self.P, self.targ_r,
                    self.ws[j], target[j][i], self.targ_zs[j], c)
                # targets[j].append(self.functions[j][i])

            # update P
            k, self.P = wp.ffP_t(self.output_num, self.P, self.r)

            # uddate internal connectivity matrix
            stat_connect_matrix = self.stat_network.connectivity_matrix
            targ_connect_matrix = self.targ_network.connectivity_matrix

            self.targ_network.connectivity_matrix =
                np.dot((np.dot(targ_connect_matrix, self.targ_r) -
                np.dot(stat_connect_matrix, self.stat_r) -
                np.dot(self.stat_ws, self.samps.T) -
                np.dot(self.stat_h, vHint.T)), k)

        end = time.time()
        # print("Time ellapsed in network training: ", end - start)

    # ----------------------- Test the network -------------------------- #
    def test(self, target, vInput, vHint, off=0, twn=False, scale=0):
        vInput = vInput
        for i in range(len(target[0])):

            # remove reference problem
            vdInput = [vInput[0], vInput[1], vInput[2], vInput[3], vInput[4]]

            # test with noise, if applicable
            if twn:
                rand1, rand2 = scale * np.random.randn(), scale * np.random.randn()
                vdInput[0] = vInput[0] + rand1 # noisy signal
                vdInput[1] = vInput[1] + rand2 # noisy signal

            if vInput[2] == 1:
                self.signal.append(vdInput[0])
            else:
                self.signal.append(vdInput[1])

            # propagate z through the network
            self.network.prop(self.zs, self.r, vdInput, Conv=0) # remove noise

            self.x = self.network.membrane_potential

            # update r
            self.r = np.tanh(self.x)

            # calculate zs
            for w in range(self.num_target_funcs):
                self.zs[w] = np.dot(np.transpose(self.ws[w]), self.r)
                self.neuron[w].append(self.zs[w]) # for avg trials
                self.errors[w].append(abs( (self.zs[w] - target[w][i])/target[w][i] ))
                self.targets[w].append(target[w][i])

        end = time.time()
        # print("Time ellapsed in network simulation: ", end - start)

    # ----------------------- Plot! -------------------------- #
    def plot(self, fig, target=None, nNumTests=1):
        errors = np.array(self.errors)

        self.neuron = np.array(self.neuron) / nNumTests

        samp = np.random.randint(0, high=200);

        # And we're done, so plot the results
        plt.figure(fig)

        plt.axis([0, 3000, -1.5, 1.5])

        plt.subplot(311)
        plt.plot(self.xs, self.targets[0], 'r--')

        plt.subplot(312)
        plt.plot(self.xs, self.neuron[samp], 'b--')

        plt.subplot(313)
        plt.plot(self.xs, errors[samp], 'g--')

        # plt.subplot(212)
        # plt.plot(self.xs, self.neuron[250], 'b--', self.xs, target, 'r--')

        plt.show()

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
