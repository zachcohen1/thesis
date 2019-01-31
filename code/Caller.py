################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: Caller.py                                                            #
#  Description: A general network caller. Network architecture 3               #
#                                                                              #
################################################################################

from InternallyRecurrentDriverMOandIO import Driver
import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

# In[1]
period = 230.0 # 1840.0
time_constant = 0.1
num_neurons = 300
chaotic_constant = 1.3
input_num = 5
output_num = 300
gg_sparseness = 0.9
gz_sparseness = 0.9
fg_sparseness = 0.9
readout_sparseness = 0.1
g_gz = 0.2
alpha = 1.0
dt = 0.1
p = 0.1
sigma = 1.0 # noise scale
epochs = 2

# In[2]
def sigmoid(x):
    return x / math.sqrt(100 + x * x)

# In[3]
targ = [ 0 for i in range(3000) ]
tests = [ targ for i in range(num_neurons) ]
driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
             input_num, output_num, gg_sparseness, gz_sparseness,
             fg_sparseness, readout_sparseness, g_gz, alpha, dt,
             tests, sigma)

poss_maps = {
    'pos'   : [[ sigmoid(i) for i in range(100)            ] for k in range(num_neurons)],
    'neg'   : [[ -1 * sigmoid(i) for i in range(100)       ] for k in range(num_neurons)],
    'r_pos' : [[ 1 - sigmoid(i) for i in range(100)        ] for k in range(num_neurons)],
    'r_neg' : [[ -1 * (1 - sigmoid(i)) for i in range(100) ] for k in range(num_neurons)],
    'base'  : [[ 0 for i in range(100)                     ] for k in range(num_neurons)]
}

# In[5]
"""

A stimulus vector =

    [cc, mc, clc, mc, ret_1]

    cc    - color coherence  [cc \in (0, 1)]
    mc    - motion coherence [mc \in (0, 1)]
    clc   - color context    [clc is 0 or 1]
    mc    - motion context   [mc is 0 or 1]
    ret_1 - ret              [ret_1 is 0 or 1]

"""

# pos, attend color
pos_c = [
    np.array([0.500, 0.100, 1, 0, 0]),
    np.array([0.450, 0.100, 1, 0, 0]),
    np.array([0.400, 0.100, 1, 0, 0]),
    np.array([0.350, 0.100, 1, 0, 0]),
    np.array([0.300, 0.100, 1, 0, 0]),
    np.array([0.250, 0.100, 1, 0, 0]),
    np.array([0.200, 0.100, 1, 0, 0]),
    np.array([0.150, 0.100, 1, 0, 0]),
    np.array([0.100, 0.100, 1, 0, 0]),
    np.array([0.050, 0.100, 1, 0, 0]),
] # len(poss) = 10

# neg, attend color
neg_c = [
    np.array([-0.500, 0.100, 1, 0, 0]),
    np.array([-0.450, 0.100, 1, 0, 0]),
    np.array([-0.400, 0.100, 1, 0, 0]),
    np.array([-0.350, 0.100, 1, 0, 0]),
    np.array([-0.300, 0.100, 1, 0, 0]),
    np.array([-0.250, 0.100, 1, 0, 0]),
    np.array([-0.200, 0.100, 1, 0, 0]),
    np.array([-0.150, 0.100, 1, 0, 0]),
    np.array([-0.100, 0.100, 1, 0, 0]),
    np.array([-0.050, 0.100, 1, 0, 0]),
] # len(poss) = 10

# pos, attend motion
pos_m = [
    np.array([0.100, 0.500, 0, 1, 0]),
    np.array([0.100, 0.450, 0, 1, 0]),
    np.array([0.100, 0.400, 0, 1, 0]),
    np.array([0.100, 0.350, 0, 1, 0]),
    np.array([0.100, 0.300, 0, 1, 0]),
    np.array([0.100, 0.250, 0, 1, 0]),
    np.array([0.100, 0.200, 0, 1, 0]),
    np.array([0.100, 0.150, 0, 1, 0]),
    np.array([0.100, 0.100, 0, 1, 0]),
    np.array([0.100, 0.050, 0, 1, 0]),
] # len(poss) = 10

# neg, attend motion
neg_m = [
    np.array([0.100, -0.500, 0, 1, 0]),
    np.array([0.100, -0.450, 0, 1, 0]),
    np.array([0.100, -0.400, 0, 1, 0]),
    np.array([0.100, -0.350, 0, 1, 0]),
    np.array([0.100, -0.300, 0, 1, 0]),
    np.array([0.100, -0.250, 0, 1, 0]),
    np.array([0.100, -0.200, 0, 1, 0]),
    np.array([0.100, -0.150, 0, 1, 0]),
    np.array([0.100, -0.100, 0, 1, 0]),
    np.array([0.100, -0.050, 0, 1, 0]),
] # len(poss) = 10

ret = np.array([0, 0, 0, 0, 1])

def train_pos_c(i):
    driv.train(poss_maps['pos'  ], pos_c[i])
    driv.train(poss_maps['r_pos'], ret)
    driv.train(poss_maps['base' ], ret)

def train_pos_m(i):
    driv.train(poss_maps['pos'  ], pos_m[i])
    driv.train(poss_maps['r_pos'], ret)
    driv.train(poss_maps['base' ], ret)

def train_neg_c(i):
    driv.train(poss_maps['neg'  ], neg_c[i])
    driv.train(poss_maps['r_neg'], ret)
    driv.train(poss_maps['base' ], ret)

def train_neg_m(i):
    driv.train(poss_maps['neg'  ], neg_m[i])
    driv.train(poss_maps['r_neg'], ret)
    driv.train(poss_maps['base' ], ret)

# In[6]
"""

    TRAIN THE NETWORK
    -----------------

    Train the network for range(ephocs), rearranging
    the order of input stimulus presentation.

"""
training_calls = [train_pos_m, train_neg_m, train_neg_c, train_pos_c]
start = time.time()
for epoch in range(epochs):
    np.random.shuffle(training_calls)
    if epoch == epochs / 4:
        end = time.time()
        print("[ 25%] training complete.", end - start, "seconds ellapsed.")
    if epoch == epochs / 2:
        end = time.time()
        print("[ 50%] training complete.", end - start, "seconds ellapsed.")
    if epoch == 3 * (epochs / 4):
        end = time.time()
        print("[ 75%] training complete.", end - start, "seconds ellapsed.")
    for i in range(40):
        if i < 10: # poss_m, attend motion
            training_calls[0](i)
        if i >= 10 and i < 20: # neg_m, attend m
            training_calls[1](i - 10)
        if i >= 20 and i < 30: # poss_c, attend c
            training_calls[2](i - 20)
        if i >= 30 and i < 40: # neg_c, attend c
            training_calls[3](i - 30)

end = time.time()
print("[100%] training complete", end - start, "seconds ellapsed.")


# ---------------- TEST ------------------- #

"""

    TEST THE NETWORK
    ----------------

    Test the network with various contexts.

"""

# In[7]
driv.targets = [ [] for i in range(300) ]
driv.errors = [ [] for i in range(300) ]
driv.neuron = [ [] for i in range(300) ]
driv.signal = []
sb = []
mb = []
hit = 0
miss = 0

samps = [ [] for i in range(20) ]
errors = [ [] for i in range(20) ]
for p in range(5):

    driv.errors = [ [] for i in range(300) ]
    driv.targets = [ [] for i in range(300) ]
    driv.neuron = [ [] for i in range(300) ]
    driv.signal = []
    sb = []
    mb = []
    for i in range(10):
        whatever = 0.4 * np.random.randn()
        for k in range(100):
            sb.append(pos_c[i][0])
            mb.append(whatever)
        driv.test(poss_maps['pos'], pos_c[i], off=100*(3*i+0), twn=True, scale=0.3) # positive color test
        test = np.sum(driv.neuron[0][100*(3*i+0) : 100*(3*i+0)+100])
        if test < 0:
            miss += 1
        else:
            hit  += 1
        for k in range(200):
            sb.append(ret[0])
            mb.append(ret[0])
        driv.test(poss_maps['r_pos'], ret, off=100*(3*i+1))
        driv.test(poss_maps['base' ], ret, off=100*(3*i+2))

    samps[p] = np.copy(driv.neuron[0])
    errors[p] = np.copy(driv.errors[0])
    """
    if i == 0 or i == 9:
        driv.test(poss_maps['pos'], pos_c[i], off=100*(3*i+0)) # positive color test
        driv.test(poss_maps['r_pos'], ret,  off=100*(3*i+1))
        driv.test(poss_maps['base'], ret,  off=100*(3*i+2))
    elif i == 1:
        driv.test(poss_maps['pos'], pos_m[i], off=100*(3*i+0)) # positive color test
        driv.test(poss_maps['r_pos'], ret,  off=100*(3*i+1))
        driv.test(poss_maps['base'], ret,  off=100*(3*i+2))
    elif i == 3:
        driv.test(poss_maps['neg'], neg_c[i], off=100*(3*i+0)) # negative color
        driv.test(poss_maps['r_neg'], ret,  off=100*(3*i+1))
        driv.test(poss_maps['base'], ret,  off=100*(3*i+2))
    elif i == 5:
        driv.test(poss_maps['neg'], neg_m[i], off=100*(3*i+0)) # negative motion
        driv.test(poss_maps['r_neg'], ret,  off=100*(3*i+1))
        driv.test(poss_maps['base'], ret,  off=100*(3*i+2))
    else:
        driv.test(poss_maps['base'], ret, off=100*(3*i+0)) # positive color
        driv.test(poss_maps['base'], ret, off=100*(3*i+1))
        driv.test(poss_maps['base'], ret, off=100*(3*i+2))
    """
# ---------------- PLOT ------------------- #
print(hit)
print(miss)
# In[8]
# driv.plot(0)

col = []
mot = []

for i in range(10):
    if i == 0 or i == 9:
        for k in range(100):
            col.append(pos_c[i][0])
            mot.append(pos_c[i][1])
        for k in range(200):
            col.append(ret[0])
            mot.append(ret[1])
    elif i == 1:
        for k in range(100):
            col.append(pos_m[i][0])
            mot.append(pos_m[i][1])
        for k in range(200):
            col.append(ret[0])
            mot.append(ret[1])
    elif i == 3:
        for k in range(100):
            col.append(neg_c[i][0])
            mot.append(neg_c[i][1])
        for k in range(200):
            col.append(ret[0])
            mot.append(ret[1])
    elif i == 5:
        for k in range(100):
            col.append(neg_m[i][0])
            mot.append(neg_m[i][1])
        for k in range(200):
            col.append(ret[0])
            mot.append(ret[1])
    else:
        for k in range(300):
            col.append(ret[0])
            mot.append(ret[1])
"""
NOTES
scale = 0.5: hit = 176, miss = 24 (200 trials)
scale = 1.0: hit = 154, miss = 46 (200 trials)
scale = 0.1: hit = 185, miss = 15 (200 trials)
"""

##### ------ #####

# errors = np.array(driv.errors)

# samp = np.random.randint(0, high=200);

# And we're done, so plot the results
plt.figure(0)

plt.subplot(412)
plt.plot(driv.xs, driv.targets[0], 'r-')
plt.ylabel('Target')
plt.axis([-30, 3035, -1.5, 1.5])

plt.subplot(411)
plt.plot(driv.xs, driv.signal)
plt.plot(driv.xs, sb, label='color')
plt.plot(driv.xs, mb, label='motion')
plt.legend()
plt.ylabel('Input signal')
plt.title('Feature discrimination', fontsize=16)
plt.axis([-30, 3035, -2.5, 2.5])

plt.subplot(413)
plt.plot(driv.xs, driv.neuron[0])
plt.plot(driv.xs, samps[0])
plt.plot(driv.xs, samps[1])
plt.plot(driv.xs, samps[2])
plt.plot(driv.xs, samps[3])
plt.plot(driv.xs, samps[4])
plt.ylabel('Network z(t)')
plt.axis([-30, 3035, -2.5, 2.5])

plt.subplot(414)
plt.plot(driv.xs, errors[0])
plt.plot(driv.xs, errors[1])
plt.plot(driv.xs, errors[2])
plt.plot(driv.xs, errors[3])
plt.plot(driv.xs, errors[4])
plt.ylabel('Error')
plt.xlabel('Time')
plt.axis([-30, 3035, -2.5, 2.5])

plt.tight_layout()
plt.savefig('feature_discrimination_noise01', dpi=300)
plt.show()
