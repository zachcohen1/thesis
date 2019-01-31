################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: Network.py                                                           #
#  Description: A network model, written in accordance with the network        #
#               equations presented in Sussillo et. al., 2009.                 #
#                                                                              #
################################################################################

import numpy as np
import scipy
from scipy import sparse
from scipy import stats

class NetworkMOIO:
    """ Construct a recurrent neural network (based on the Sussillo implementation cited
        in the README). The network is described by time_constant, or the time steps
        in the overall evolution of the network, num_neurons, or the number of neurons
        in the network, chaotic_constant, or the stabilizing weighting constant that
        allows for chaotic activity in the network.

        [...explain other input variables...]
        readout_sparseness = p_z in paper
        g_z is an output constant scaling factor
    """
    def __init__(self, time_constant, num_neurons, p, chaotic_constant,
                 input_num, output_num, gg_sparseness, gz_sparseness,
                 fg_sparseness, readout_sparseness, g_gz, dt, nNumTargFuncs):

        # evolution time constant
        self.time_constant = time_constant

        # number of neurons in the network
        self.num_neurons = num_neurons

        # chaotic constant. Sussillo recommends chaotic_constant = 1.5 to allow
        # for chaotic dynamics
        self.chaotic_constant = chaotic_constant

        # membrane potentials of each neurons
        self.membrane_potential = 0.5 * np.random.randn(num_neurons)

        # number of input neurons
        self.input_num = input_num

        # number of output neurons
        self.output_num = output_num

        # Connectivity matrix dictating synaptic strength of connections between
        # neurons in the network. (J^{GG} in the paper),
        # gg_sparseness is inter-neuron connectivitiy sparsenesss.
        scale = 1.0 / np.sqrt(p * num_neurons)
        self.connectivity_matrix = scale * chaotic_constant * \
            scipy.sparse.random(num_neurons, num_neurons,
            density=(1-gg_sparseness), data_rvs = np.random.randn).toarray()

        # Connectivity matrix dictating synaptic strength of connections between
        # neurons in the network and readout neurons
        self.feedout_matrix = sparse.random(output_num, num_neurons,
            density=(1-gz_sparseness), data_rvs = np.random.randn).toarray()

        # Connectivity matrix dictating synaptic strength of connections between
        # neurons in the network and feedin neurons
        self.feedin_matrix = sparse.random(input_num, num_neurons,
            density=(1-fg_sparseness), data_rvs = np.random.randn).toarray()

        # J^{G_z}
        self.z_matrix = np.random.uniform(-1, 1, (num_neurons, nNumTargFuncs))

        # J^{GI} - to fix: issue of all column enrties being 0
        rand_ind = np.arange(input_num)
        np.random.shuffle(rand_ind)
        self.input_con = [
            [np.random.randn() if rand_ind[j % input_num] == i else 0 for i in range(input_num)]
            for j in range(num_neurons)
        ]

        # scaling factor for z
        self.g_gz = g_gz

        # timestep
        self.dt = dt

    """ Propogate the feedback input in z through the network. This models the
        differential equation presented in Sussillo. z is the dot product of w
        and r(t) (see definitions above) """
    def prop(self, z, r, context):
        self.membrane_potential = (1-self.dt) * self.membrane_potential + \
            np.matmul(self.connectivity_matrix, (r * self.dt)) + \
            np.dot(self.input_con, (context * self.dt))

        """
        z = self.dt * z
        self.membrane_potential = (1-self.dt) * self.membrane_potential + \
            np.matmul(self.connectivity_matrix, (r * self.dt)) + \
            np.dot(self.z_matrix, np.transpose(z * self.dt))
        """

"""Start with 100 neurons. membrane_potential[i] is the membrane potential of
   neuron i.
# network = Network(0.001, 1000, 1.7, 10, 10, 0.9, 0.9, 0.9, 0.1, 0.2)
# network.prop(0.009)
# print(type(network.connectivity_matrix))
# print(network.membrane_potential[:]) """
