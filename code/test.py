import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.colors as colors

data = np.load('/Users/zachcohen/Dropbox/JuniorYear/fall17/IW/datafiles/fixed_exp_feb26_shuffle.npy')
neu = np.load('/Users/zachcohen/Dropbox/JuniorYear/fall17/IW/datafiles/loaded_data0.npy')

golden = len(neu[0][0]) + 300
def sigmoid(x):
    return x / math.sqrt(100 + x * x)


# poss_maps = {
#     'baseline': [[ 0 for i in range(len(agg))                ] for k in range(num_neurons)]
#     'pos'     : [[ sigmoid(i) for i in range(100)            ] for k in range(num_neurons)],
#     'neg'     : [[ -1 * sigmoid(i) for i in range(100)       ] for k in range(num_neurons)],
#     'r_pos'   : [[ 1 - sigmoid(i) for i in range(100)        ] for k in range(num_neurons)],
#     'r_neg'   : [[ -1 * (1 - sigmoid(i)) for i in range(100) ] for k in range(num_neurons)],
#     'base'    : [[ 0 for i in range(100)                     ] for k in range(num_neurons)]
# }


def createNeuTrace():
	test_vec = np.array( [ np.zeros(golden * 5) for i in range(len(neu)) ] )
	for i in range(len(neu)):
		vec = np.array([])
		for j in [1 , 5, 18, 23, 55]:
			vec = np.append(vec, neu[i][j])
			vec = np.append(vec, np.zeros(100))
			vec = np.append(vec, 0.2 + np.zeros(100))
			vec = np.append(vec, np.zeros(100))
		test_vec[i] = np.copy(vec)
	return test_vec


def createNeuTraceAlt():
	test_vec = np.array( [ np.zeros(len(neu[0][0]) + 200) for i in range(len(neu)) ] )
	for i in range(len(neu)):
		vec = np.array([])
		for j in [1]:
			vec = np.append(vec, neu[i][j])
			vec = np.append(vec, np.zeros(100))
			vec = np.append(vec, 0.2 + np.zeros(100))
		test_vec[i] = np.copy(vec)
	return test_vec

def createTarget():
	ret = []
	for i in [1, -1, 1, 1, -1]:
		for t in range(len(neu[0][0])):
			ret.append(0)
		for t in range(100):
			if i == 1: ret.append(sigmoid(t))
			else: ret.append(-1 * sigmoid(t))
		for t in range(100):
			if i == 1: ret.append(1 - sigmoid(t))
			else: ret.append(-1 * (1 - sigmoid(t)))
		for t in range(100):
			ret.append(0)
	return ret

def createTargetAlt():
	ret = []
	for i in [-1]:
		for t in range(len(neu[0][0])):
			ret.append(0)
		for t in range(100):
			if i == 1: ret.append(sigmoid(t))
			else: ret.append(-1 * sigmoid(t))
		for t in range(100):
			if i == 1: ret.append(1 - sigmoid(t))
			else: ret.append(-1 * (1 - sigmoid(t)))
		for t in range(100):
			ret.append(0)
	return ret

plt.figure(0)
plt.title('Decoder experiment, $N = 1000$')
plt.subplot(3, 1, 1)
plt.imshow(createNeuTrace(), aspect='auto')
plt.ylabel("Neuron activity")
plt.xticks([])

plt.subplot(3, 1, 2)
plt.plot(createTarget(), color='cornflowerblue')
plt.xlim((0, len(data[0:5*golden])))
plt.ylabel("Target response")
plt.xticks([])

plt.subplot(3, 1, 3)
plt.plot(np.arange(len(data[0:5*golden])), data[0*golden:5*golden], 'r-')
plt.xlim((0, len(data[0:5*golden])))
plt.xlabel("Time (ms)")
plt.ylabel("Network output")
plt.tight_layout()
plt.show()
# plt.savefig('example_decoder_feb26_shuffle', dpi=300)

####

# plt.figure(0)
# plt.title('Decoder experiment, $N = 1000$')
# plt.subplot(3, 1, 1)
# plt.imshow(createNeuTraceAlt(), aspect='auto')
# plt.ylabel("Neuron activity")

# plt.subplot(3, 1, 2)
# plt.plot(createTargetAlt())
# plt.ylim((-1.2, 1))
# plt.xlim((0, len(data[0:311])))
# plt.ylabel("Target response")

# plt.subplot(3, 1, 3)
# plt.plot(np.arange(len(data[0:1*371])), data[0*371:1*371], 'r-')
# plt.xlim((0, len(data[0:1*311])))
# plt.ylim((-1.2, 1))
# plt.xlabel("Time (ms)")
# plt.ylabel("Network output")
# plt.tight_layout()
# plt.savefig('example_decoder_feb25_one', dpi=300)
