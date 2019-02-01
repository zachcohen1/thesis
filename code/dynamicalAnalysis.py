import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fmin, fmin_cg, minimize, check_grad

class dynamicAnalysis:
        """ Performs fixed point analysis on a given network"""
	def __init__(self, network, f):
            """
            Args:
                network : Network - an RNN on which to perform fp analysis
                f : []float -> []float - non-linearity used in experiments

            Returns:
                None
            """
	    self.network = network
	    self.context = None
	    self.trace = []
	    self.f = f  # nonlinearity

	""" Returns a tuple of the left and right eigenvalues of M """
	def decompose(self, M):
            """
            Args:
                M : [][]float - matrix to decompose

            Returns
                (lv, rv, s) : tuple - left and right eigenvectors and associated
                                      eigenvalues
            """
	    return sp.linalg.eig(M, right=True, left=True)

	""" Return the Jacobian of F(x) """
	def M(self, x):
            """
            Args:
                x : []float - activations of each neuron in the network

            Returns:
                M : [][]float - Jacobian of network at time t
            """
	    x = np.array(x, dtype=np.float64)
	    n, n = np.shape(self.network.connectivity_matrix)
	    rp = self.f(x) * (1 - self.f(x))
	    M = np.multiply(self.network.connectivity_matrix,
                    np.outer(np.ones(n), rp)) - np.eye(n)
	    return M

        """ Return the Hessian of F(x)"""
	def hess(self, x):
            """
            Args:
                x : []float - activations of each neuron in the network

            Returns:
                [][]float - Hessian of the network at time t
            """
	    self.network.membrane_potential = x
	    dx = self.p_prop(self.context)
	    n, n = np.shape(self.network.connectivity_matrix)
	    rp = self.f(x) * (1 - self.f(x))
	    h = np.multiply(np.transpose(self.network.connectivity_matrix),
                    np.outer(rp, np.ones(n))) - np.eye(n)
	    d2 = np.multiply(-2 * self.f(x), rp)
	    return np.dot(h, h) + \
		np.diag(np.multiply(d2, np.dot(self.network.connectivity_matrix.T, dx)))

        """ Compute the gradient of F(x)"""
	def grad(self, x):
            """
            Args:
                x : []float - activations of each neuron in the network

            Returns:
                []float - Gradient of the network at time t
            """
	    self.network.membrane_potential = x
	    dx = self.p_prop(self.context)
	    n, n = np.shape(self.network.connectivity_matrix)
	    rp = self.f(x) * (1 - self.f(x))
	    h = np.multiply(self.network.connectivity_matrix.T,
                    np.outer(rp, np.ones(n))) - np.eye(n)
	    return np.dot(h, dx)

        """ Equation to minimize: q(x) = 1/2 * || F(x) ||^2"""
	def q(self, x):
            """
            Args:
                x : []float - activations of each neuron in the network

            Returns:
                q(x) : float - q(x) at some point x
            """
	    self.network.membrane_potential = x
	    dx = self.p_prop(self.context)
	    return np.dot(dx, dx) / 2

        """ Make sure everything is running okay"""
	def check_ourg(self, x):
            """
            Args:
                x : []float - activations of each neuron in the network

            Returns:
                None
            """
	    print(check_grad(self.q, self.grad, x))

        """ Propagate the network forward one step in time. Notice the difference between
            this equation and the one used in CleanNetwork.py: the one in CleanNetwork is
            the Euler integration approximation, whereas this one is the right hand side of
            the differential equation governing RNN dynamics"""
	def p_prop(self, context):
            """
            Args:
                context : []float - sensory evidence + context cue

            Returns
                []float - activations of network in next time step
            """
	    r = self.f(self.network.membrane_potential)
	    return -self.network.membrane_potential + np.dot(self.network.connectivity_matrix, r) + \
		    np.dot(context, self.network.input)

	""" Callback function for minimizer if tracking is desired"""
	def track(self, x):
            """
            Args:
                x : []float - activations of network in this time step

            Returns:
                None
            """
	    self.trace.append(x)

        """ Find the minimum of q(x) using some starting point"""
	def solve(self, ic):
            """
            Args:
                ic : tuple([]float, []float) - a tuple with a starting set of neuron activations
                                               and a context set (sensory evidence + context cue)
                                               that the network integrates

            Returns:
                tuple([]float, []float, min{}, [][]float) - the first tuple entry is the inital
                                               condition, the second is the context vector, the
                                               third is the minimizer object returned by
                                               scipy's minimizer (with various information about
                                               the minimization), and the final entry is the
                                               trace of the system as it's minimized
            """
            self.network.membrane_potential = ic[0]
            self.context = ic[1]
            # minimum = fmin(self.q, ic[0])
            mins = minimize(self.q, ic[0], method='L-BFGS-B', jac=self.grad, hess=self.hess,
                    options={'maxiter':20000, 'disp':None, 'maxfun':1000000, 'iprint':0}, callback=self.track)
            if not mins['success']:
                    self.trace = None
            trace = np.copy(self.trace)
            self.trace = []
            return ic[0], ic[1], mins, trace

# def main():
# 	from InternallyRecurrentDriverMOandIOnCaller import Driver
# 	period = 230.0 # 1840.0 (need to take this out)
# 	time_constant = 0.1
# 	chaotic_constant = 1.3
# 	input_num = 5
# 	num_neurons = 726
# 	output_num = 726
# 	gg_sparseness = 0.9
# 	gz_sparseness = 0.9
# 	fg_sparseness = 0.9
# 	readout_sparseness = 0.1
# 	g_gz = 0.2
# 	alpha = 1.0
# 	dt = 0.1
# 	p = 0.1
# 	sigma = 0.0 # noise scale
# 	epochs = 6

# 	# In[3]
# 	targ = [ 0 for i in range(3000) ]
# 	tests = [ targ for i in range(num_neurons) ]
# 	driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
# 	             input_num, output_num, gg_sparseness, gz_sparseness,
# 	             fg_sparseness, readout_sparseness, g_gz, alpha, dt,
# 	             tests, sigma, target_in_network=True)
# 	da = dynamicAnalysis(driv.network)
# 	da.context = np.array([0.5, 0.5, 1, 0, 0])
# 	np.random.seed(6739779)
# 	x = np.random.randn(num_neurons)
# 	M = da.M(x)
# 	guess = np.dot(M.T, x)
# 	print(guess[0:10])
# 	print(scipy.optimize.approx_fprime(x, da.q, 1/200000+np.zeros(len(x)))[0:10])
# 	print(scipy.optimize.check_grad(da.q, da.q_prime, x))

# if __name__ == '__main__':
# 	main()
