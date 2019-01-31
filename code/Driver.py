################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: Driver.py                                                            #
#  Description: A general network driver.  Network architecture 1              #
#                                                                              #
################################################################################

import numpy as np
from scipy import sparse
import WeightUpdate as wp
from Network import Network
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Driver:
    """ Network driver. Run the network and see what happens."""
    def __init__(self, period, time_constant, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, gz_sparseness,
                 fg_sparseness, readout_sparseness, g_gz, alpha, dt):

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

        # instantiate a nework
        self.network = Network(time_constant, num_neurons, p, chaotic_constant,
                               input_num, output_num, gg_sparseness,
                               gz_sparseness, fg_sparseness, readout_sparseness,
                               g_gz, dt)

        # try a complex sine wave (from Sussillo 2009)
        self.target_function = [ ((1.3/1.0) * np.sin(    np.pi * i/250) + \
                                  (1.3/2.0) * np.sin(2 * np.pi * i/250) + \
                                  (1.3/6.0) * np.sin(3 * np.pi * i/250) + \
                                  (1.3/3.0) * np.sin(4 * np.pi * i/250)) / 1.5 \
                                  for i in range(int(period / dt)) ]


        self.alpha = alpha
        x = self.network.membrane_potential # x(0)
        w_t = 0.5 * np.random.randn(num_neurons) #np.zeros(num_neurons) # w(0)
        P = (1/alpha) * np.identity(output_num) # P(0)
        z = 0.5 * np.random.randn()
        r = np.tanh(x)

        # plot
        target = [] # plot the target function
        # plot the output of the network during training
        neuron_one_train = []
        # plot the output of the network during testing (w generated)
        neuron_one_test = []
        xs = []

        # test neurons
        neuron_1 = []
        neuron_2 = []
        neuron_3 = []

        # train
        for i in range(int(period / dt)):
            # propagate z through the network
            self.network.prop(z, r)

            # update r and z
            x = self.network.membrane_potential
            r = np.tanh(x)
            z = np.dot(np.transpose(w_t), r)

            neuron_1.append(x[111])
            neuron_2.append(x[478])
            neuron_3.append(x[901])

            """
            # update w and P
            w_t = wp.w_t(output_num, P, r, w_t, self.target_function[i], z)
            P = wp.P_t(output_num, P, r)
            """
            # plot
            neuron_one_train.append(z) # plotting
            target.append(self.target_function[i]) # plotting
            xs.append(i) # plotting

        # test
        z = 0
        for i in range(int(period / dt)):
            # propagate z through the network
            self.network.prop(z, r)

            x = self.network.membrane_potential



            # update r and z
            r = np.tanh(x)
            z = np.dot(np.transpose(w_t), r)
            neuron_one_test.append(z)


        # And we're done, so plot the results
        plt.figure(1)
        plt.subplot(211)
        plt.plot(xs, neuron_1, 'b--', xs, neuron_2, 'r--', xs, neuron_3, 'g--')

        plt.subplot(212)
        plt.plot(xs, neuron_one_test, 'b--', xs, target, 'r--')
        plt.show()

period = 400.0
time_constant = 0.1
num_neurons = 1000
chaotic_constant = 1.5
input_num = 1000
output_num = 1000
gg_sparseness = 0.9
gz_sparseness = 0.9
fg_sparseness = 0.9
readout_sparseness = 0.1
g_gz = 0.2
alpha = 1.0
dt = 0.1
p = 0.1
driv = Driver(period, time_constant, num_neurons, p, chaotic_constant, input_num,
              output_num, gg_sparseness, gz_sparseness, fg_sparseness,
              readout_sparseness, g_gz, alpha, dt)
