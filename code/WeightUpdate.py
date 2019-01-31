################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: WeightUpdate.py                                                      #
#  Description: A set of helper functions for updating the network output      #
#               during the FORCE learning procedure.                           #
#                                                                              #
################################################################################

import numpy as np

""" Update cross-correlation matrix"""
def P_t(P_backstep, c, Pr):
    return (P_backstep - c * np.outer(Pr, Pr))

def w_t(num_output, P_backstep, r, w_backstep, target_val, z, c):
    return (w_backstep + (-1 * e_(z, target_val)) * (np.dot(P_backstep, r) * c))

"""
P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
num_output = 3
timestep = 1e-3
target_val = 0.0576927
r = np.array([0.003, 0.8729, 0.1274])
w_backstep = np.array([0.991, 0.008, 0.019])
w = w_t(num_output, timestep, P, r, w_backstep, target_val)
print(w[:])
"""
