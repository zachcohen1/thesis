################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: InternallyRecurrentDriver.py                                         #
#  Description: A general network driver. Network architecture 3               #
#                                                                              #
################################################################################

import numpy as np
import math
from scipy import sparse
import WeightUpdate as wp
from NetworkMO import NetworkMO
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as pca
import time
plt.style.use('seaborn-paper')

class Driver:
    """ Network driver. Run the network and see what happens. """
    def __init__(self, period, time_constant, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, gz_sparseness,
                 fg_sparseness, readout_sparseness, g_gz, alpha, dt,
                 target_functions):

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

        # instantiate a nework
        self.network = NetworkMO(time_constant, num_neurons, p, chaotic_constant,
                               input_num, output_num, gg_sparseness,
                               gz_sparseness, fg_sparseness, readout_sparseness,
                               g_gz, dt, len(target_functions))

        # target functions
        self.functions = target_functions

        # number of target functions
        self.num_target_funcs = len(target_functions)

        # alpha scale
        self.alpha = alpha
        x = self.network.membrane_potential # x(0)
        P = (1/alpha) * np.identity(output_num) # P(0)
        r = np.tanh(x)

        # plot
        targets = [ [] for i in range(self.num_target_funcs) ]
        # plot the output of the network during training
        neuron_one_train = []
        # plot the output of the network during testing (w generated)
        self.neuron = [ [] for i in range(self.num_target_funcs) ]

        # computer learning error
        errors = [ [] for i in range(self.num_target_funcs) ]

        # ds
        dt_s = [ [] for i in range(self.num_target_funcs) ]

        # samples
        samps = np.array([0 for i in range(self.num_target_funcs)])

        # x coordinates
        xs = []

        # output vectors
        ws = [ np.zeros(num_neurons) for i in range(self.num_target_funcs) ]

        # zs
        zs = [ 0.5 * np.random.randn() for i in range(self.num_target_funcs) ]
        zs = np.array(zs)

        end = time.time()
        print("Time ellapsed in network instantiation: ", end - start)

        # ---------------------- Train the network -------------------------- #
        print("Beginning training...")
        start = time.time()
        for i in range(len(self.functions[0])):
            # propagate z through the network
            self.network.prop(zs, r)

            # update r
            x = self.network.membrane_potential
            r = np.tanh(x)

            # update zs
            for w in range(self.num_target_funcs):
                zs[w] = np.dot(np.transpose(ws[w]), r)
                samps[w] = self.functions[w][i]

            # error slippage
            err_mat = (zs - samps)

            # update ws, dts
            c = wp.c(r, P)

            for j in range(self.num_target_funcs):
                dt_s[j] = wp.dw_t(output_num, P, r, self.functions[j][i], zs[j], c)

            for j in range(self.num_target_funcs):
                ws[j] = wp.w_t(output_num, P, r, ws[j], self.functions[j][i],
                            zs[j], c)
                targets[j].append(self.functions[j][i])


            # update P
            P = wp.P_t(output_num, P, r)

            # uddate internal connectivity matrix
            connect_matrix = self.network.connectivity_matrix
            new_connect_matrix = connect_matrix + dt_s
            self.network.connectivity_matrix = new_connect_matrix

        end = time.time()
        print("Time ellapsed in network training: ", end - start)

        # ----------------------- Test the network -------------------------- #
        print("Testing network...")
        start = time.time()
        for i in range(len(self.functions[0])):
            # propagate z through the network
            self.network.prop(zs, r)

            x = self.network.membrane_potential

            # update r
            r = np.tanh(x)

            # calculate zs
            for w in range(self.num_target_funcs):
                zs[w] = np.dot(np.transpose(ws[w]), r)
                self.neuron[w].append(zs[w])
                errors[w].append(abs(self.functions[w][i] - zs[w]))

            xs.append(i)


        end = time.time()
        print("Time ellapsed in network simulation: ", end - start)
        errors = np.array(errors)

        # check learning error
        # for err in range(self.num_target_funcs):
        #     print("error in target no. ", err)
        #     print(np.mean(errors[err]))

        # And we're done, so plot the results
        # plt.figure(1)
        # plt.plot(np.arange(len(self.functions[0])), self.neuron[0])
        # plt.show()

period = 400.0
time_constant = 0.1
num_neurons = 500
chaotic_constant = 1.3
input_num = 500
output_num = 500
gg_sparseness = 0.9
gz_sparseness = 0.9
fg_sparseness = 0.9
readout_sparseness = 0.1
g_gz = 0.2
alpha = 1.0
dt = 0.1
p = 0.1
# 1000 neurons
test_funcs = np.array([
    [ ((1.3/1.0) * np.sin(    np.pi * i/250) + \
                              (1.3/2.0) * np.sin(2 * np.pi * i/250) + \
                              (1.3/6.0) * np.sin(3 * np.pi * i/250) + \
                              (1.3/3.0) * np.sin(4 * np.pi * i/250)) / 1.5 \
                              for i in range(int(period / dt)) ],
    [ ((1.3/1.0) * np.cos(    np.pi * i/250) + \
                              (1.3/2.0) * np.cos(2 * np.pi * i/250) + \
                              (1.3/6.0) * np.cos(3 * np.pi * i/250) + \
                              (1.3/3.0) * np.cos(4 * np.pi * i/250)) / 1.5 \
                              for i in range(int(period / dt)) ],
    [ ((1.3/1.0) * np.sin(    np.pi * i/250) + \
                              (1.3/2.0) * np.cos(2 * np.pi * i/250) + \
                              (1.3/6.0) * np.sin(3 * np.pi * i/250) + \
                              (1.3/3.0) * np.cos(4 * np.pi * i/250)) / 1.5 \
                              for i in range(int(period / dt)) ],
    [ ((1.3/1.0) * np.cos(    np.pi * i/250) + \
                              (1.3/2.0) * np.sin(2 * np.pi * i/250) + \
                              (1.3/6.0) * np.cos(3 * np.pi * i/250) + \
                              (1.3/3.0) * np.sin(4 * np.pi * i/250)) / 1.5 \
                              for i in range(int(period / dt)) ]
])
# mats = np.load('compressed_neu.npy')
test_funcs_prime =  [
    test_funcs[ int(np.floor(np.random.rand() * 4)) ] for i in range(500)
]

driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
              input_num, output_num, gg_sparseness, gz_sparseness,
              fg_sparseness, readout_sparseness, g_gz, alpha, dt,
              test_funcs_prime)


plt.figure(0)
plt.subplot(411)
plt.plot(np.arange(len(test_funcs[0])), test_funcs[0], linewidth=1.5, label='target')
plt.plot(np.arange(len(driv.functions[0])), driv.neuron[0], linewidth=1.0, label='network')
plt.axis([-50, 4050, -1.5, 1.5])
plt.ylabel('Neuron 1 z(t)')
plt.title('Network simulation of multiple periodic functions')
plt.legend()

plt.subplot(412)
plt.plot(np.arange(len(test_funcs[2])), test_funcs[2], linewidth=1.5, label='target')
plt.plot(np.arange(len(driv.functions[4])), driv.neuron[4], linewidth=1.0, label='network')
plt.axis([-50, 4050, -1.5, 1.5])
plt.ylabel('Neuron 4 z(t)')

plt.subplot(413)
plt.plot(np.arange(len(test_funcs[1])), test_funcs[1], linewidth=1.5, label='target')
plt.plot(np.arange(len(driv.functions[2])), driv.neuron[2], linewidth=1.0, label='network')
plt.axis([-50, 4050, -2.25, 2.25])
plt.ylabel('Neuron 2 z(t)')

plt.subplot(414)
plt.plot(np.arange(len(test_funcs[3])), test_funcs[3], linewidth=1.5, label='target')
plt.plot(np.arange(len(driv.functions[5])), driv.neuron[5], linewidth=1.0, label='network')
plt.axis([-50, 4050, -1.75, 1.75])
plt.ylabel('Neuron 5 z(t)')
plt.xlabel('Time')
plt.tight_layout()

plt.savefig('multiple_periodic_data', dpi=300)
plt.show()
