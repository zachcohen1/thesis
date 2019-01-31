################################################################################
#                                                                              #
#  Author: Zach Cohen                                                          #
#  Title: Caller.py                                                            #
#  Description: A general network caller. Network architecture 3               #
#                                                                              #
################################################################################

from InternallyRecurrentDriverMOandIO import Driver
from pfc import *
import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

############################   Load neural data   ############################


dm = True # discard mistrials
head = '/Users/zachcohen/Dropbox/ManteData_justdata/PFC data 1/'
agg_ = gatherdata(head, dm, start=700) # gather pfc data, see pfc.py for details
agg = agg + agg_
num_neurons = len(agg)
num_neurons

np.save('loaded_data0', agg)

############################       Network       ############################


period = 230.0 # 1840.0 (need to take this out)
time_constant = 0.1
chaotic_constant = 1.3
input_num = 5
output_num = num_neurons
gg_sparseness = 0.9
gz_sparseness = 0.9
fg_sparseness = 0.9
readout_sparseness = 0.1
g_gz = 0.2
alpha = 1.0
dt = 0.1
p = 0.1
sigma = 0.0 # noise scale
epochs = 5

# In[3]
targ = [ 0 for i in range(3000) ]
tests = [ targ for i in range(num_neurons) ]
driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
             input_num, output_num, gg_sparseness, gz_sparseness,
             fg_sparseness, readout_sparseness, g_gz, alpha, dt,
             tests, sigma)

targ = [ 0 for i in range(3000) ]
tests = [ targ for i in range(num_neurons) ]
driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
             input_num, output_num, gg_sparseness, gz_sparseness,
             fg_sparseness, readout_sparseness, g_gz, alpha, dt,
             tests, sigma)

driv.num_neurons

############################ Organize contexts ############################


cc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # color coherence values
mc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # motion coherence values
vecs = []  # context / coherence vecs
ret = [0, 0, 0, 0, 1]

for i in range(2):
    for j in range(len(cc)):
        for k in range(len(mc)):
            if i == 0: vecs.append(np.array([ cc[j], mc[k], 1, 0, 0 ]))
            else: vecs.append(np.array([ cc[j], mc[k], 0, 1, 0 ]))


############################ Train the network ############################

base = np.zeros(200)
basenet = [ base for i in range(num_neurons) ]
def train(batch, vec):
    driv.train([ np.linspace(0, batch[p][0], 20) for p in range(len(batch)) ],
        vec) # smooth gain
    driv.train(batch, vec) # correspondence between data and context
    driv.train([ np.linspace(batch[p][-1], 0, 20) for p in range(len(batch)) ],
        ret) # smooth decline
    driv.train(basenet, ret)

# In[6]
"""

    TRAIN THE NETWORK
    -----------------

    Train the network for range(ephocs), rearranging
    the order of input stimulus presentation.

"""
start = time.time()
for epoch in range(epochs):
    if epoch == epochs / 4:
        end = time.time()
        print("[ 25%] training complete.", end - start, "seconds ellapsed.")
    if epoch == epochs / 2:
        end = time.time()
        print("[ 50%] training complete.", end - start, "seconds ellapsed.")
    if epoch == 3 * (epochs / 4):
        end = time.time()
        print("[ 75%] training complete.", end - start, "seconds ellapsed.")
    for i in range(int(len(vecs)/8)):
        if i == len(vecs) / 2: print("Processed 50% of contexts")
        train([ agg[k][i] for k in range(len(agg)) ], vecs[i])

sig = [ [] for i in range(72) ]
targ = [ [] for i in range(72) ]

res = [ [] for i in range(72) ]
#start = time.time()
for t in range(72):
    for i in range(5):
        train([ agg[k][t] for k in range(len(agg)) ], vecs[t])
    #end = time.time()
    #end - start
    #print("[100%] training complete", end - start, "seconds ellapsed.")


    ############################ Test the network ############################

    """

        TEST THE NETWORK
        ----------------

        Test the network with various contexts.

    """

    # In[7]
    driv.targets = [ [] for i in range(num_neurons) ]
    driv.errors = [ [] for i in range(num_neurons) ]
    driv.neuron = [ [] for i in range(num_neurons) ]
    driv.signal = []

    test = [ agg[k][t] for k in range(len(agg)) ]

    driv.test([ np.linspace(0, test[p][0], 20) for p in range(len(test)) ],
        vecs[t])
    driv.test(test, vecs[t], off=100, twn=True, scale=0.1)
    driv.test([ np.linspace(0, test[t][-1], 20) for p in range(len(test)) ], ret,
        off=len(test[t])+100)
    driv.test(basenet, ret, off=len(test[t])+200)

    res[t] = [ driv.neuron[i][20:(len(toshow1[0])+20)] for i in range(num_neurons) ]
    # ---------------- PLOT ------------------- #

    #res = [ [] for i in range(72) ]

    ##### ------ #####

    # from matplotlib import rc
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # rc('text', usetex=True)
    #
    # import matplotlib.colors as colors

np.save('exp_samples_1_5_18', res)

# t=0
# plt.figure(0)
# plt.rc('axes', titlesize=6)
# ax1 = plt.subplot2grid((4,6), (0,0), colspan=6)
# ax1.plot(np.arange(len(sig[t])), sig[t])
# ax1.set_xlabel(r'Time')
# ax1.set_ylabel(r'Input signal')
# plt.title(r'$\texttt{CC} = -0.5, \texttt{MC} = -0.5, \rho^2 = 0.1$', fontsize=16)
# ax1.axis([-10, len(test[0])+10, -1, 1])
#
# toshow1 = [ agg[i][t] for i in range(len(agg)) ]
# ax2 = plt.subplot2grid((4,6), (1,0), colspan=3, rowspan=3)
# im = ax2.imshow(targ[t], cmap='gnuplot2', interpolation='hanning',
#     aspect='auto', vmin=-3.25, vmax=3.25)
# plt.colorbar(im, ax=ax2)
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Target')
#
# len(driv.neuron[0])
# ts = [driv.neuron[i] for i in range(num_neurons)]
# err = [ driv.errors[i][20:len(toshow1[0])+20] for i in range(num_neurons) ]
# toshow = [driv.neuron[i][20:(len(toshow1[0])+20)] for i in range(num_neurons)]
# ax3 = plt.subplot2grid((4,6), (1,3), colspan=3, rowspan=3)
# im = ax3.imshow(res[t], cmap='gnuplot2', interpolation='hanning',
#     aspect='auto', vmin=-3.25, vmax=3.25)
# plt.colorbar(im, ax=ax3)
# ax3.set_xlabel('Time')
# ax3.set_ylabel('Network z(t)')
#
# plt.tight_layout()
# plt.savefig('728neupng000_', dpi=300)
# plt.show()


# for t in range(22):
#     targ[t] = [ agg[i][t] for i in range(len(agg)) ]
#
# ## save
# res[t] = toshow
# sig[t] = driv.signal[20:len(test[0])+20]
# targ[t] = toshow1
from numpy import linalg as LA
pVar = np.zeros(72)
pHolder = np.zeros(72)
mHolder = np.zeros(72)
# pholder = np.zeros(len(targ[0]))
# for n in range(1, 72):
#     for i in range(len(targ[n])):
#         for p in range(len(targ[n][i])):
#             pholder[i] += targ[n][i][p] - res[n][i][p]



for n in range(72):
    for p in range(len(targ[n])):
        for k in range(len(targ[n][p])):
            pHolder[n] += np.abs(targ[n][p][k] - res[n][p][k])
    # pHolder[n] = np.array(targ[n]) - np.array(res[n])
    # pVar[n] = LA.norm(pHolder[n], 'fro')
    det = np.mean(targ[n])
    for p in range(len(targ[n])):
        for k in range(len(targ[n][p])):
            mHolder[n] += np.abs(targ[n][p][k] - det)

    pVar[n] = pHolder[n]**2 / mHolder[n]**2
pVar
1-np.array(pVar)


for n in range(72):
    for p in range(len(targ[n])):
        for k in range(len(targ[n][p])):
            pHolder[n] += targ[n][p][k] - res[n][p][k]
    # pHolder[n] = np.array(targ[n]) - np.array(res[n])
    # pVar[n] = LA.norm(pHolder[n], 'fro')
    det = np.mean(targ[n], axis=0)
    for p in range(len(targ[n])):
        for k in range(len(targ[n][p])):
            mHolder[n] += targ[n][p][k] - det[k]

    pVar[n] = pHolder[n]**2 / mHolder[n]**2
pVar

np.var(pVar[0:36])

pvar1 = 1-np.array(pVar)
pvar1

np.mean(pvar1[36:72])
np.var(pvar1[36:72])

np.save('pvar', pVar)
np.save('1-pVar', pvar1)

for t in range(num_neurons):
    plt.figure(0)
    plt.rc('axes', titlesize=6)

    ax2 = plt.subplot2grid((3,7), (0,0), colspan=3, rowspan=3)
    im = ax2.imshow(agg[t], cmap='gnuplot2', interpolation='hanning',
        aspect='auto', vmin=-3.25, vmax=3.25)
    plt.colorbar(im, ax=ax2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Target')
    title = 'Neuron ' + str(t+1)
    plt.suptitle(title, fontsize=16)

    len(driv.neuron[0])
    ts = [driv.neuron[i] for i in range(num_neurons)]
    err = [ driv.errors[i][20:len(toshow1[0])+20] for i in range(num_neurons) ]
    toshow12 = [ res[i][t] for i in range(72) ]
    ax3 = plt.subplot2grid((3,7), (0,4), colspan=3, rowspan=3)
    im = ax3.imshow(toshow12, cmap='gnuplot2', interpolation='hanning',
        aspect='auto', vmin=-3.25, vmax=3.25)
    plt.colorbar(im, ax=ax3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Network z(t)')
    #plt.tight_layout()
    filename = 'neuron'+str(t)
    plt.savefig(filename, dpi=300)

t = 0
k = 7
plt.figure(0)
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), toshow1[k], 'k-', linewidth=0.5, label='target')
plt.plot(np.arange(len(res[t][k])), res[t][k], 'r-', linewidth=0.5, label='network')
plt.show()

t = 0
k = 8
plt.figure(0)
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), toshow1[k], 'k-', linewidth=0.5, label='target')
plt.plot(np.arange(len(res[t][k])), res[t][k], 'r-', linewidth=0.5, label='network')
plt.show()


t = 0
k = 11
plt.figure(0)
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), toshow1[k], 'k-', linewidth=0.5, label='target')
plt.plot(np.arange(len(res[t][k])), res[t][k], 'r-', linewidth=0.5, label='network')
plt.legend()
plt.show()



fig = plt.figure(0)

t = 0
k = 11
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), toshow1[k], 'k-', linewidth=0.5, label='target')
plt.plot(np.arange(len(res[t][k])), res[t][k], 'r-', linewidth=0.5, label='network')

t = 0
k = 8
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), np.array(toshow1[k]) + 4, 'k-', linewidth=0.5)
plt.plot(np.arange(len(res[t][k])), np.array(res[t][k]) + 4, 'r-', linewidth=0.5)

t = 1
k = 9
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), np.array(toshow1[k]) + 8, 'k-', linewidth=0.5)
plt.plot(np.arange(len(res[t][k])), np.array(res[t][k]) + 8, 'r-', linewidth=0.5)

t = 0
k = 41
toshow1 = [ agg[i][t] for i in range(len(agg)) ]
plt.plot(np.arange(len(toshow1[k])), np.array(toshow1[k]) + 12, 'k-', linewidth=0.5)
plt.plot(np.arange(len(res[t][k])), np.array(res[t][k]) + 12, 'r-', linewidth=0.5)

frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
frame1.axes.set_xlabel('Time')
plt.legend()
plt.savefig('imprecise', dpi=300)
plt.show()

plt.figure(0)
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[0:6], 'ro', label='cc = -0.5')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[6:12], 'bo', label='cc = -0.17')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[12:18], 'go', label='cc = -0.05')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[18:24], 'yo', label='cc = 0.05')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[24:30], 'co', label='cc = 0.17')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[30:36], 'mo', label='cc = 0.5')
plt.xticks([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5])
plt.xlabel('Motion coherence')
plt.ylabel(r'\texttt{pAcc}')
plt.title('Response accuracy in color context')
axes.set_ylim([0.71,1.01])
plt.legend()
plt.savefig('colacc', dpi=300)
plt.show()


plt.figure(0)
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[36:42], 'ro', label='cc = -0.5')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[42:48], 'bo', label='cc = -0.17')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[48:54], 'go', label='cc = -0.05')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[54:60], 'yo', label='cc = 0.05')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[60:66], 'co', label='cc = 0.17')
plt.plot([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5], pvar1[66:72], 'mo', label='cc = 0.5')
plt.xticks([-0.5, -0.17, -0.05, 0.05, 0.17, 0.5])
plt.xlabel('Motion coherence')
plt.ylabel(r'\texttt{pAcc}')
plt.title('Response accuracy in motion context')
axes = plt.gca()
axes.set_ylim([0.71,1.01])
plt.legend()
plt.savefig('motacc', dpi=300)
plt.show()



"""
The MSE between each conditional PSTH and the model PSTH for each condition,
compared to the MSE between the conditional PSTH and the marginal PSTH
(computed by averaging the data over all 72 conditions).
"""

predict = np.load('/Users/zachcohen/Dropbox/JuniorYear/fall17/IW/datafiles/exp_samples_1_5_18.npy')
target  = np.load('/Users/zachcohen/Dropbox/JuniorYear/fall17/IW/datafiles/loaded_data0.npy')

predict.shape
mse = []
for i in range(72):
    grab = [ target[j][i] for j in range(726) ]
    grab = np.array(grab)
    mse.append(((grab - predict[i]) ** 2).mean())

mse


avg_n_data = target.mean(axis=1)

mse_n = []
for i in range(72):
    grab = [ target[j][i] for j in range(726) ]
    grab = np.array(grab)
    mse_n.append(((grab - avg_n_data) ** 2).mean())

mse_n

###########################################################################
########################                           ########################
########################       New Experiment      ########################
########################                           ########################
###########################################################################

data = np.load('loaded_data0.npy')
input_num = len(data)
num_neurons = 1000
driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
             input_num, output_num, gg_sparseness, gz_sparseness,
             fg_sparseness, readout_sparseness, g_gz, alpha, dt,
             tests, sigma)


############################ Organize targets  ############################

def sigmoid(x):
    return x / math.sqrt(100 + x * x)

poss_maps = {
    'baseline': [[ 0 for i in range(len(agg))                ] for k in range(num_neurons)]
    'pos'     : [[ sigmoid(i) for i in range(100)            ] for k in range(num_neurons)],
    'neg'     : [[ -1 * sigmoid(i) for i in range(100)       ] for k in range(num_neurons)],
    'r_pos'   : [[ 1 - sigmoid(i) for i in range(100)        ] for k in range(num_neurons)],
    'r_neg'   : [[ -1 * (1 - sigmoid(i)) for i in range(100) ] for k in range(num_neurons)],
    'base'    : [[ 0 for i in range(100)                     ] for k in range(num_neurons)]
}

done = [[ 0 for i in range(300) ] for k in range(input_num)]


############################ Train the network ############################


base = np.zeros(300)
basenet = [ base for i in range(num_neurons) ]
def train(batch, ret_batch, vec):
    driv.train(poss_maps['baseline'], vec) # neural data presentation
    driv.train(batch, basenet) # stimulus response
    driv.train(ret_batch, ret) # return to baseline
    driv.train(basenet, ret) # baseline

cc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # color coherence values
mc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # motion coherence values
pos  = []
for k in range(2):
    for i in range(len(cc)):
        for j in range(len(mc)):
            if k == 0:
                if cc[i] < 0: pos.append(False)
                else pos.append(True)
            else:
                if mc[j] < 0: pos.append(False)
                else pos.append(True)


start = time.time()
for epoch in range(epochs):
    if epoch == epochs / 4:
        end = time.time()
        print("[ 25%] training complete.", end - start, "seconds ellapsed.")
    if epoch == epochs / 2:
        end = time.time()
        print("[ 50%] training complete.", end - start, "seconds ellapsed.")
    if epoch == 3 * (epochs / 4):
        end = time.time()
        print("[ 75%] training complete.", end - start, "seconds ellapsed.")
    for i in range(len(vecs)):
        if pos[i]:
            train(poss_maps['pos'], poss_maps['r_pos'], [ agg[k][i] for k in range(len(agg)) ])
        else:
            train(poss_maps['neg'], poss_maps['r_neg'], [ agg[k][i] for k in range(len(agg)) ])

end = time.time()
print("[100%] training complete.", end - start, "seconds ellapsed.")


############################ Test the network ############################


res_0 = [ [] for i in range(72) ]
for i in range(len(vecs)):
    test_vec = [ agg[k][i] for k in range(len(agg)) ]
    driv.test(poss_maps['baseline'], test_vec)
    if pos[i]: driv.test(poss_maps['pos'], basenet, off=len(agg))
    else: driv.test(poss_maps['neg'], basenet, off=len(agg))
    if pos[i]: driv.test(poss_maps['r_pos'], basenet, off=len(agg))
    else: driv.test(poss_maps['r_neg'], basenet, off=len(agg))
    res[i] = [ driv.neuron[k] for k in range(num_neurons) ]

np.save('test_decoder', res_0)

###########################################################################
########################                           ########################
########################       New Experiment      ########################
########################                           ########################
###########################################################################

data = np.load('loaded_data0.npy')
input_num = len(data)
num_neurons = 1000
driv = Driver(period, time_constant, num_neurons, p, chaotic_constant,
             input_num, output_num, gg_sparseness, gz_sparseness,
             fg_sparseness, readout_sparseness, g_gz, alpha, dt,
             tests, sigma)


############################ Organize targets  ############################

def sigmoid(x):
    return x / math.sqrt(100 + x * x)

poss_maps = {
    'baseline': [[ 0 for i in range(len(agg))                ] for k in range(num_neurons)]
    'pos'     : [[ sigmoid(i) for i in range(100)            ] for k in range(num_neurons)],
    'neg'     : [[ -1 * sigmoid(i) for i in range(100)       ] for k in range(num_neurons)],
    'r_pos'   : [[ 1 - sigmoid(i) for i in range(100)        ] for k in range(num_neurons)],
    'r_neg'   : [[ -1 * (1 - sigmoid(i)) for i in range(100) ] for k in range(num_neurons)],
    'base'    : [[ 0 for i in range(100)                     ] for k in range(num_neurons)]
}

done = [[ 0 for i in range(300) ] for k in range(input_num)]


############################ Train the network ############################


base = np.zeros(300)
basenet = [ base for i in range(num_neurons) ]
def train(batch, ret_batch, vec):
    driv.train(poss_maps['baseline'], vec) # neural data presentation
    driv.train(batch, basenet) # stimulus response
    driv.train(ret_batch, ret) # return to baseline
    driv.train(basenet, ret) # baseline

cc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # color coherence values
mc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # motion coherence values
pos  = []
for k in range(2):
    for i in range(len(cc)):
        for j in range(len(mc)):
            if k == 0:
                if cc[i] < 0: pos.append(False)
                else pos.append(True)
            else:
                if mc[j] < 0: pos.append(False)
                else pos.append(True)


start = time.time()
for epoch in range(epochs):
    if epoch == epochs / 4:
        end = time.time()
        print("[ 25%] training complete.", end - start, "seconds ellapsed.")
    if epoch == epochs / 2:
        end = time.time()
        print("[ 50%] training complete.", end - start, "seconds ellapsed.")
    if epoch == 3 * (epochs / 4):
        end = time.time()
        print("[ 75%] training complete.", end - start, "seconds ellapsed.")
    for i in range(len(vecs)):
        if pos[i]:
            train(poss_maps['pos'], poss_maps['r_pos'], [ agg[k][i] for k in range(len(agg)) ])
        else:
            train(poss_maps['neg'], poss_maps['r_neg'], [ agg[k][i] for k in range(len(agg)) ])

end = time.time()
print("[100%] training complete.", end - start, "seconds ellapsed.")


############################ Test the network ############################


res_0 = [ [] for i in range(72) ]
for i in range(len(vecs)):
    test_vec = [ agg[k][i] for k in range(len(agg)) ]
    driv.test(poss_maps['baseline'], test_vec)
    if pos[i]: driv.test(poss_maps['pos'], basenet, off=len(agg))
    else: driv.test(poss_maps['neg'], basenet, off=len(agg))
    if pos[i]: driv.test(poss_maps['r_pos'], basenet, off=len(agg))
    else: driv.test(poss_maps['r_neg'], basenet, off=len(agg))
    res[i] = [ driv.neuron[k] for k in range(num_neurons) ]

np.save('test_decoder', res_0)