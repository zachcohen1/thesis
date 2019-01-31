import matplotlib.pyplot as plt
import numpy as np 

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

######################################################################
## 																	##
##   																##
##                 Decoding experiment plotting						##
##																	##
## 																	##
######################################################################

"""
decode = np.load('test_decoder_1000_0_all.npy')
plt.figure()
for i in range(18, 36):
	plt.plot(decode[0][371*i:371*(i+1)])
	
plt.plot([95, 95+0.0001], [0, decode[0][95]], '--')
plt.plot([105, 105+0.0001], [0, decode[0][105]], '--')
plt.plot([270, 270+0.0001], [0, decode[0][270]], '--')
# plt.plot(decode[0][30*371:31*371])
plt.ylim((-1.2, 1.2))
plt.annotate('', xy=(-5, 0), xytext=(370, 0), arrowprops={'arrowstyle':'|-|'})
plt.scatter([0, 95, 105, 270], [0,0,0,0])
plt.annotate(r'Input \\ signal ON', xy=(0, 0), xytext=(0, 0.5), size=13,
	arrowprops={'facecolor':'black', 'arrowstyle':'->'})
plt.annotate(r'Input \\ signal OFF', xy=(95, 0.0), xytext=(70, 0.4), size=13,
	arrowprops={'facecolor':'black', 'arrowstyle':'->'})
plt.annotate(r'Go \\ cue ON', xy=(105, 0), xytext=(110, -0.4), size=13,
	arrowprops={'facecolor':'black', 'arrowstyle':'->'})
plt.annotate(r'Go cue OFF', xy=(270, 0), xytext=(255, 0.2), size=13,
	arrowprops={'facecolor':'black', 'arrowstyle':'->'})
plt.title(r'Decoding experiment')
plt.ylabel(r'$$z(t)$$')
plt.xlabel(r'$$t$$')
# plt.savefig('decode_1', dpi=300)
plt.show()
"""

######################################################################
## 																	##
##   																##
##          Target in network experiment plotting (delayed)		    ##
##																	##
## 																	##
######################################################################

"""
target_in_network = np.load('../datafiles/jp2/april_17_2018.npy')
targ = np.load('../datafiles/jp2/loaded_data0.npy')
t = 1
import math
def sigmoid(x):
    return x / math.sqrt(100 + x * x)

poss_maps = {
    'baseline': [ 0 for i in range(target_in_network.shape[2]-200)   ],
    'pos'     : [ sigmoid(i) for i in range(100)                     ],
    'neg'     : [ -1 * sigmoid(i) for i in range(100)                ],
    'r_pos'   : [ 1 - sigmoid(i) for i in range(100)                 ],
    'r_neg'   : [ -1 * (1 - sigmoid(i)) for i in range(100)          ],
    'base'    : [ 0 for i in range(100)                              ]
}

appendpos = np.append(poss_maps['baseline'], np.append(poss_maps['pos'], poss_maps['r_pos']))
appendneg = np.append(poss_maps['baseline'], np.append(poss_maps['neg'], poss_maps['r_neg']))

ft = [ targ[k][t] for k in range(len(targ)) ]
ft = np.concatenate(( [np.linspace(0, ft[p][0], 20) for p in range(len(ft))], ft), axis=1)
ft = np.concatenate(( ft, [np.linspace(ft[p][-1], 0, 20) for p in range(len(ft))]), axis=1)
ft = np.concatenate(( ft, [np.zeros(200) for i in range(len(ft))]), axis=1)

plt.figure()
plt.subplot(221)
im1 = plt.imshow(target_in_network[t][0:-1], aspect='auto')
plt.colorbar(im1)
plt.title('network output')

plt.subplot(222)
plt.title('target output')
im = plt.imshow(ft, aspect='auto')
plt.colorbar(im)

plt.subplot(212)
plt.title('stimulus response')
plt.plot(target_in_network[t][-1], label='network')
plt.plot(appendneg, 'r--', label='target')
plt.legend()
plt.tight_layout()
plt.savefig('target_in_network_0', dpi=300)
# plt.show()
"""

######################################################################
## 																	##
##   																##
##          Target in network experiment plotting (t = 0)		    ##
##																	##
## 																	##
######################################################################

# target_in_network = np.load('../datafiles/jp2/april_18_2018_t=0.npy')
# targ = np.load('../datafiles/jp2/loaded_data0.npy')
# t = 3
# import math
# def sigmoid(x):
#     return x / math.sqrt(100 + x * x)

# poss_maps = {
#     'baseline': [ 0 for i in range(int((target_in_network.shape[2]-80)/2)) ],
#     'pos'     : [ sigmoid(i) for i in range(int((target_in_network.shape[2]-100))) ],
#     'neg'     : [ -1 * sigmoid(i) for i in range(int((target_in_network.shape[2]-100))) ],
#     'r_pos'   : [ 1 - sigmoid(i) for i in range(100)                 ],
#     'r_neg'   : [ -1 * (1 - sigmoid(i)) for i in range(100)          ],
#     'base'    : [ 0 for i in range(100)                              ]
# }

# appendpos = np.append(np.zeros(20), np.append(poss_maps['pos'], poss_maps['r_pos'][0:80]))
# appendneg = np.append(np.zeros(20), np.append(poss_maps['neg'], poss_maps['r_neg'][0:80]))

# ft = [ targ[k][t] for k in range(len(targ)) ]
# ft = np.concatenate(( [np.linspace(0, ft[p][0], 20) for p in range(len(ft))], ft), axis=1)
# ft = np.concatenate(( ft, [np.linspace(ft[p][-1], 0, 20) for p in range(len(ft))]), axis=1)
# ft = np.concatenate(( ft, [np.zeros(80) for i in range(len(ft))]), axis=1)

# plt.figure()
# plt.subplot(221)
# im1 = plt.imshow(target_in_network[t][0:-1], aspect='auto')
# plt.colorbar(im1)
# plt.title('network output')

# plt.subplot(222)
# plt.title('target output')
# im = plt.imshow(ft, aspect='auto')
# plt.colorbar(im)

# plt.subplot(212)
# plt.title('stimulus response')
# plt.plot(target_in_network[t][-1], label='network')
# plt.plot(appendneg, 'r--', label='target')
# plt.legend()
# plt.tight_layout()
# plt.savefig('target_in_network_1', dpi=300)
#plt.show()


######################################################################
## 																	##
##   																##
##                      FORCE v full-FORCE		                    ##
##																	##
## 																	##
######################################################################

"""
target_FORCE = np.load('../datafiles/jp2/april_17_2018.npy')
target_ff = np.load('../datafiles/jp2/ff_resp.npy')
targ = np.load('../datafiles/jp2/loaded_data0.npy')
t=0

ft = [ targ[k][t] for k in range(len(targ)) ]
ft = np.concatenate(( [np.linspace(0, ft[p][0], 20) for p in range(len(ft))], ft), axis=1)
ft = np.concatenate(( ft, [np.linspace(ft[p][-1], 0, 20) for p in range(len(ft))]), axis=1)
ft = np.concatenate(( ft, [np.zeros(90) for i in range(len(ft))]), axis=1)
ft2 = np.copy(ft)

show_force = [ target_FORCE[t][i][0:250] for i in range(726) ]

plt.figure()
plt.subplot(221)
im1 = plt.imshow(ft, aspect='auto')
plt.colorbar(im1)
plt.title('target output')

plt.subplot(222)
plt.title('FORCE network output')
im = plt.imshow(show_force, aspect='auto')
plt.colorbar(im)

plt.subplot(223)
im1 = plt.imshow(ft2, aspect='auto')
plt.colorbar(im1)
plt.title('target output')

plt.subplot(224)
plt.title('full-FORCE network output')
im = plt.imshow(target_ff, aspect='auto')
plt.colorbar(im)

plt.tight_layout()
plt.savefig('ff_v_FORCE', dpi=300)
# plt.show()
"""

######################################################################
## 																	##
##   																##
##                  Noise level comparison		                    ##
##																	##
## 																	##
######################################################################


res = np.load('noise_experiment_7.npy', encoding='latin1')
targ = np.load('../datafiles/jp2/loaded_data0.npy')

# print(np.shape(res))

# # noise = 2
# # neuron = 8
# # neuron1_network = [ res[i][neuron] for i in range(len(res)) ]
# # neuron1_network = neuron1_network / np.linalg.norm(neuron1_network)
# # target = targ[neuron] / np.linalg.norm(targ[neuron])

# cc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # color coherence values
# mc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # motion coherence values
# vecs = []  # context / coherence vecs
# ret = [0, 0, 0, 0, 1]

# for i in range(2):
#     for j in range(len(cc)):
#         for k in range(len(mc)):
#             if i == 0: vecs.append((np.array([ cc[j], mc[k], 1, 0, 0 ]), True if cc[j] > 0 else False))
#             else: vecs.append((np.array([ cc[j], mc[k], 0, 1, 0 ]), True if mc[k] > 0 else False))

networks = []
targets = []
noise = 0
for i in [5, 7, 15, 19, 21]:
	n = [ res[noise][0][j][i][20:131] for j in range(len(res[noise][0])) ]
	networks.append(n / np.linalg.norm(n))
	targets.append( targ[i] / np.linalg.norm(targ[i]) )

# fig = plt.figure(figsize=(15,12))
# for noise in range(1,4):
# 	networks = []
# 	targets = []
# 	for k in [5, 7, 15, 19, 21]:
# 		n = [ res[noise][0][j][k][20:131] for j in range(len(res[noise][0])) ]
# 		networks.append(n / np.linalg.norm(n))
# 		# targets.append( targ[k] / np.linalg.norm(targ[k]) )
	
# 	# plt.suptitle(r'Network, $$\rho = $$' + str(0.0) + ', clc = ' + str(vecs[18][0][0]))
# 	for i in range(5):
# 		ax1 = None
# 		if i == 0:
# 			ax1 = plt.subplot(3,5,((noise-1)*5)+1)
# 			ax1.set_ylim(-0.04, 0.04)
# 			ax1.set_xticks([])
# 			ax1.set_ylabel('Network output')
# 			# title = 'Network, $$\rho = ' + str(0.0) + '$$, clc = ' + str(vecs[18][0])
# 			# ax1.set_title(title)
# 		else:
# 			ax = plt.subplot(3,5,((noise-1)*5+1)+i, sharey=ax1)
# 			ax.set_xticks([])
# 			ax.set_yticks([])
# 			# title = 'Network, $$\rho = $$' + str(0.0) + ', clc = ' + str(vecs[18][0])
# 			# ax.set_title(title)
# 		plt.plot(networks[i][18], label='-0.5', linestyle='-', color='cornflowerblue', alpha=1)
# 		plt.plot(networks[i][19], label='-0.17', linestyle='-', color='cornflowerblue', alpha=0.55)
# 		plt.plot(networks[i][20], label='-0.05', linestyle='-', color='cornflowerblue', alpha=0.25)
# 		plt.plot(networks[i][21], label='0.05', linestyle='-', color='red', alpha=0.25)
# 		plt.plot(networks[i][22], label='0.17', linestyle='-', color='red', alpha=0.55)
# 		plt.plot(networks[i][23], label='0.5', linestyle='-', color='red', alpha=1)
# 		# if i == 0: plt.legend(loc='upper left')
# 	# fig.tight_layout()
# plt.savefig('nc2_noise_comp', dpi=300, bbox_inches='tight')
# plt.show()


# fig = plt.figure(figsize=(15,8))
# # plt.suptitle(r'Network, $$\rho = $$' + str(0.0) + ', clc = ' + str(vecs[18][0][0]))
# for i in range(5):
# 	ax1 = None
# 	if i == 0:
# 		ax1 = plt.subplot(2,5,6)
# 		ax1.set_ylim(-0.04, 0.04)
# 		ax1.set_xticks([])
# 		ax1.set_ylabel('Network output')
# 		# title = 'Network, $$\rho = ' + str(0.0) + '$$, clc = ' + str(vecs[18][0])
# 		# ax1.set_title(title)
# 	else:
# 		ax = plt.subplot(2,5,(i+6), sharey=ax1)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# title = 'Network, $$\rho = $$' + str(0.0) + ', clc = ' + str(vecs[18][0])
# 		# ax.set_title(title)
# 	plt.plot(networks[i][18], label='-0.5', linestyle='-', color='cornflowerblue', alpha=1)
# 	plt.plot(networks[i][19], label='-0.17', linestyle='-', color='cornflowerblue', alpha=0.55)
# 	plt.plot(networks[i][20], label='-0.05', linestyle='-', color='cornflowerblue', alpha=0.25)
# 	plt.plot(networks[i][21], label='0.05', linestyle='-', color='red', alpha=0.25)
# 	plt.plot(networks[i][22], label='0.17', linestyle='-', color='red', alpha=0.55)
# 	plt.plot(networks[i][23], label='0.5', linestyle='-', color='red', alpha=1)
# 	# if i == 0: plt.legend(loc='upper left')

# for i in range(5):
# 	ax2 = None
# 	if i == 0:
# 		ax2 = plt.subplot(2,5,1)
# 		ax2.set_ylim(-0.04, 0.04)
# 		ax2.set_xticks([])
# 		ax2.set_ylabel('Target output')
# 		# title = 'Target, $$\rho = ' + str(0.0) + '$$, clc = ' + str(vecs[18][0])
# 		# ax2.set_title(title)
# 	else:
# 		ax = plt.subplot(2,5,(i+1), sharey=ax2)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# title = 'Target, $$\rho = ' + str(0.0) + '$$, clc = ' + str(vecs[18][0])
# 		# ax.set_title(title)
# 	plt.plot(targets[i][18], label='-0.5', linestyle='-', color='cornflowerblue', alpha=1)
# 	plt.plot(targets[i][19], label='-0.17', linestyle='-', color='cornflowerblue', alpha=0.55)
# 	plt.plot(targets[i][20], label='-0.05', linestyle='-', color='cornflowerblue', alpha=0.25)
# 	plt.plot(targets[i][21], label='0.05', linestyle='-', color='red', alpha=0.25)
# 	plt.plot(targets[i][22], label='0.17', linestyle='-', color='red', alpha=0.55)
# 	plt.plot(targets[i][23], label='0.5', linestyle='-', color='red', alpha=1)
# 	if i == 0: plt.legend(loc='upper left')
# # fig.tight_layout()
# plt.savefig('nc2_rho_0', dpi=300, bbox_inches='tight')
# # plt.show()

######################################################################
## 																	##
##   																##
##                  		PCA experiment	                        ##
##																	##
## 																	##
######################################################################


# contexts = [ [] for i in range(72) ]
# for i in range(72):
# 	contexts[i] = np.array([ res[0][0][i][j][20:131] for j in range(len(res[0][0][i])-1) ])
# # mean_vec = noise1_act - np.mean(noise1_act, axis=0)
# # x = np.dot(mean_vec.T, mean_vec)
# # u, s, vt = np.linalg.svd(x)

# # datavar = np.zeros(len(s))
# # datavar[0] = s[0]
# # for i in range(1, len(s)):
# # 	datavar[i] = datavar[i - 1] + s[i] 

# # datavar = datavar / np.sum(s)

# # plt.figure()
# # plt.scatter(np.arange(len(s)), datavar)
# # plt.show()

# ### seems like 95% of the variance is described by 5 pcs. Let's choose 12 to be safe
# from sklearn.decomposition import PCA
# ts = [ [] for i in range(72) ]
# pca = PCA(n_components=6, svd_solver='full')

# for i in range(72):
# 	pca.fit(contexts[i].T)
# 	ts[i] = pca.transform(contexts[i].T)

# # cons = [0, 1, 2, 3, 4, 5, 35, 47, 53, 59, 65, 71]
# #  cons = [2, 3]
# # print(vecs[65][0], vecs[71][0])

# # plt.figure()
# # plt.subplot(111)
# # plt.ylim(-30, 50)
# # label = None
# # for i, con in enumerate(cons):
# # 	if vecs[con][1]: 
# # 		label = 'report positive'
# # 		# color = 'red'
# # 	else: 
# # 		label = 'report negative'
# # 		# color = 'cornflowerblue'
# # 	if con == 2: color = 'red'
# # 	# if con < 36: color = 'red'
# # 	else: color = 'cornflowerblue'
# # 	plt.plot(ts[con].T[0], label=label, color=color)
# # 	plt.plot(ts[con].T[1], label=label, color=color)
# # plt.legend()

# # plt.subplot(122)
# # plt.ylim(-30, 50)
# # for i, con in enumerate(cons):
# # 	if vecs[con][1]: 
# # 		label = 'report positive'
# # 		# color = 'red'
# # 	else: 
# # 		label = 'report negative'
# # 		# color = 'cornflowerblue'
# # 	if con < 36: color = 'red'
# # 	else: color = 'cornflowerblue'
# # 	plt.plot(ts[con].T[1], label=label, color=color)
# # plt.legend()

# # plt.subplot(133)
# # plt.ylim(-30, 50)
# # for i, con in enumerate(cons):
# # 	if vecs[con][1]: 
# # 		label = 'report positive'
# # 		# color = 'red'
# # 	else: 
# # 		label = 'report negative'
# # 		# color = 'cornflowerblue'
# # 	if con < 36: color = 'red'
# # 	else: color = 'cornflowerblue'
# # 	plt.plot(ts[con].T[2], label=label, color=color)
# # plt.legend()

# avgs_pc1_attend_color = [ [] for i in range(8) ]
# num = np.zeros(8)

# 	avgs[0]: cc < 0, mc > 0, go pos
# 	avgs[1]: cc < 0, mc > 0, go neg
# 	avgs[2], cc < 0, mc < 0, go pos
# 	avgs[3], cc < 0, mc < 0, go neg
# 	avgs[4]: cc > 0, mc > 0, go pos
# 	avgs[5]: cc > 0, mc > 0, go neg
# 	avgs[6], cc > 0, mc < 0, go pos
# 	avgs[7], cc > 0, mc < 0, go neg

# for i, vec in enumerate(vecs):
# 	cc, mc, ac, am = vec[0][0], vec[0][1], vec[0][2], vec[0][3]
# 	_dir = vec[1]
# 	if cc < 0 and mc > 0 and _dir:
# 		if avgs_pc1_attend_color[0] == []: avgs_pc1_attend_color[0] = ts[i].T
# 		else: avgs_pc1_attend_color[0] = avgs_pc1_attend_color[0] + ts[i].T
# 		num[0] += 1
# 	elif cc < 0 and mc > 0 and not _dir:
# 		if avgs_pc1_attend_color[1] == []: avgs_pc1_attend_color[1] = ts[i].T
# 		else: avgs_pc1_attend_color[1] = avgs_pc1_attend_color[1] + ts[i].T
# 		num[1] += 1
# 	elif cc < 0 and mc < 0 and _dir:
# 		if avgs_pc1_attend_color[2] == []: avgs_pc1_attend_color[2] = ts[i].T
# 		else: avgs_pc1_attend_color[2] = avgs_pc1_attend_color[2] + ts[i].T
# 		num[2] += 1
# 	elif cc < 0 and mc < 0 and not _dir:
# 		if avgs_pc1_attend_color[3] == []: avgs_pc1_attend_color[3] = ts[i].T
# 		else: avgs_pc1_attend_color[3] = avgs_pc1_attend_color[3] + ts[i].T
# 		num[3] += 1
# 	elif cc > 0 and mc > 0 and not _dir:
# 		if avgs_pc1_attend_color[4] == []: avgs_pc1_attend_color[4] = ts[i].T
# 		else: avgs_pc1_attend_color[4] = avgs_pc1_attend_color[4] + ts[i].T
# 		num[4] += 1
# 	elif cc > 0 and mc > 0 and not _dir:
# 		if avgs_pc1_attend_color[5] == []: avgs_pc1_attend_color[5] = ts[i].T
# 		else: avgs_pc1_attend_color[5] = avgs_pc1_attend_color[5] + ts[i].T
# 		num[5] += 1
# 	elif cc > 0 and mc < 0 and _dir:
# 		if avgs_pc1_attend_color[6] == []: avgs_pc1_attend_color[6] = ts[i].T
# 		else: avgs_pc1_attend_color[6] = avgs_pc1_attend_color[6] + ts[i].T
# 		num[6] += 1
# 	elif cc > 0 and mc < 0 and not _dir:
# 		if avgs_pc1_attend_color[7] == []: avgs_pc1_attend_color[7] = ts[i].T
# 		else: avgs_pc1_attend_color[7] = avgs_pc1_attend_color[7] + ts[i].T
# 		num[7] += 1


# # avgs_pc1_attend_color = [ avgs_pc1_attend_color[i] / num[i] for i in range(8) ]
# plt.figure()
# plt.subplots(121)
# for i in range(8):
# 	plt.plot(avgs_pc1_attend_color[i][0])

# plt.show()
# """

# """
# res = np.load('noise_experiment_7.npy', encoding='latin1')
# targ = np.load('../datafiles/jp2/loaded_data0.npy')

# # ##############################################################################

# cc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # color coherence values
# mc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # motion coherence values
# vecs = []  # context / coherence vecs
# ret = [0, 0, 0, 0, 1]

# for i in range(2):
#     for j in range(len(cc)):
#         for k in range(len(mc)):
#             if i == 0: vecs.append((np.array([ cc[j], mc[k], 1, 0, 0 ]), True if cc[j] > 0 else False))
#             else: vecs.append((np.array([ cc[j], mc[k], 0, 1, 0 ]), True if mc[k] > 0 else False))

# # ##############################################################################               

# noise = 0
# contexts = [ [] for i in range(72) ]
# for i in range(72):
# 	contexts[i] = np.array([ res[noise][0][i][j][20:131] for j in range(len(res[0][0][i])-1) ])

# ### seems like 95% of the variance is described by 5 pcs. Let's choose 12 to be safe
# from sklearn.decomposition import PCA
# ts = [ [] for i in range(72) ]
# pca = PCA(n_components=6, svd_solver='full')

# for i in range(72):
# 	pca.fit(contexts[i].T)
# 	ts[i] = pca.transform(contexts[i].T)

# ##############################################################################

# def organize(ts_):
#     """
#         avgs[0]: cc < 0, mc > 0, col
#         avgs[1]: cc < 0, mc > 0, mo
#         avgs[2], cc < 0, mc < 0, col
#         avgs[3], cc < 0, mc < 0, mo
#         avgs[4]: cc > 0, mc > 0, col
#         avgs[5]: cc > 0, mc > 0, mo
#         avgs[6], cc > 0, mc < 0, col
#         avgs[7], cc > 0, mc < 0, mo
#     """
#     collected = [ [] for i in range(8) ]
#     num = np.zeros(8)
    
#     for i, vec in enumerate(vecs):
#         cc, mc, ac, am = vec[0][0], vec[0][1], vec[0][2], vec[0][3]
#         _dir = vec[1]
#         if cc < 0 and mc > 0 and ac==1:
#             if collected[0] == []: collected[0] = ts_[i].T
#             else: collected[0] = collected[0] + ts_[i].T
#             num[0] += 1
#         elif cc < 0 and mc > 0 and am==1:
#             if collected[1] == []: collected[1] = ts_[i].T
#             else: collected[1] = collected[1] + ts_[i].T
#             num[1] += 1
#         elif cc < 0 and mc < 0 and ac==1:
#             if collected[2] == []: collected[2] = ts_[i].T
#             else: collected[2] = collected[2] + ts_[i].T
#             num[2] += 1
#         elif cc < 0 and mc < 0 and am==1:
#             if collected[3] == []: collected[3] = ts_[i].T
#             else: collected[3] = collected[3] + ts_[i].T
#             num[3] += 1
#         elif cc > 0 and mc > 0 and ac==1:
#             if collected[4] == []: collected[4] = ts_[i].T
#             else: collected[4] = collected[4] + ts_[i].T
#             num[4] += 1
#         elif cc > 0 and mc > 0 and am==1:
#             if collected[5] == []: collected[5] = ts_[i].T
#             else: collected[5] = collected[5] + ts_[i].T
#             num[5] += 1
#         elif cc > 0 and mc < 0 and ac==1:
#             if collected[6] == []: collected[6] = ts_[i].T
#             else: collected[6] = collected[6] + ts_[i].T
#             num[6] += 1
#         elif cc > 0 and mc < 0 and am==1:
#             if collected[7] == []: collected[7] = ts_[i].T
#             else: collected[7] = collected[7] + ts_[i].T
#             num[7] += 1
   
#     collected = [ collected[i] / num[i] for i in range(8) ]
#     return collected

# ##############################################################################

# avgs_pc1_attend_color = organize(np.array(ts))

# avgs_pc1_attend_color1 = [ avgs_pc1_attend_color[i][0] / np.linalg.norm(avgs_pc1_attend_color[i][0]) for i in range(6) ]
# avgs_pc1_attend_color2 = [ avgs_pc1_attend_color[i][1] / np.linalg.norm(avgs_pc1_attend_color[i][1]) for i in range(6) ]
# avgs_pc1_attend_color3 = [ avgs_pc1_attend_color[i][2] / np.linalg.norm(avgs_pc1_attend_color[i][2]) for i in range(6) ]
# avgs_pc1_attend_color_ = [ [avgs_pc1_attend_color1[i], avgs_pc1_attend_color2[i], avgs_pc1_attend_color3[i]] for i in range(6) ]

# ##############################################################################

# noise = 1
# contexts1 = [ [] for i in range(72) ]
# for i in range(72):
# 	contexts1[i] = np.array([ res[noise][0][i][j][20:131] for j in range(len(res[0][0][i])-1) ])

# ts2 = [ [] for i in range(72) ]
# pca = PCA(n_components=6, svd_solver='full')

# for i in range(72):
# 	pca.fit(contexts1[i].T)
# 	ts2[i] = pca.transform(contexts1[i].T)

# ##############################################################################


# avgs_pc1_attend_color1 = organize(np.array(ts2))

# avgs_pc1_attend_color11 = [ avgs_pc1_attend_color1[i][0] / np.linalg.norm(avgs_pc1_attend_color1[i][0]) for i in range(8) ]
# avgs_pc1_attend_color12 = [ avgs_pc1_attend_color1[i][1] / np.linalg.norm(avgs_pc1_attend_color1[i][1]) for i in range(8) ]
# avgs_pc1_attend_color13 = [ avgs_pc1_attend_color1[i][2] / np.linalg.norm(avgs_pc1_attend_color1[i][2]) for i in range(8) ]
# avgs_pc1_attend_color1_ = [ [avgs_pc1_attend_color11[i], avgs_pc1_attend_color12[i], avgs_pc1_attend_color13[i]] for i in range(8) ]


# # ##############################################################################


# ts_Targ = [ [] for i in range(72) ]
# pca = PCA(n_components=6, svd_solver='full')

# for i in range(72):
# 	pca.fit(targ[i].T)
# 	ts_Targ[i] = pca.transform(targ[i].T)

# ##############################################################################

# avgs_targ = organize(np.array(ts_Targ))

# for i in range(8):
#     avgs_targ[i][0] = avgs_targ[i][0] / np.linalg.norm(avgs_targ[i][0])
#     avgs_targ[i][1] = avgs_targ[i][1] / np.linalg.norm(avgs_targ[i][1])
#     avgs_targ[i][2] = avgs_targ[i][2] / np.linalg.norm(avgs_targ[i][2])

# target_no_noise = [ [avgs_targ[i][0], avgs_targ[i][1], avgs_targ[i][2] ] for i in range(8) ]

# ##############################################################################

# for i in range(8):
#     avgs_pc1_attend_color[i][0] = avgs_pc1_attend_color[i][0] / np.linalg.norm(avgs_pc1_attend_color[i][0])
#     avgs_pc1_attend_color[i][1] = avgs_pc1_attend_color[i][1] / np.linalg.norm(avgs_pc1_attend_color[i][1])
#     avgs_pc1_attend_color[i][2] = avgs_pc1_attend_color[i][2] / np.linalg.norm(avgs_pc1_attend_color[i][2])
    
#     avgs_pc1_attend_color1[i][0] = avgs_pc1_attend_color1[i][0] / np.linalg.norm(avgs_pc1_attend_color1[i][0])
#     avgs_pc1_attend_color1[i][1] = avgs_pc1_attend_color1[i][1] / np.linalg.norm(avgs_pc1_attend_color1[i][1])
#     avgs_pc1_attend_color1[i][2] = avgs_pc1_attend_color1[i][2] / np.linalg.norm(avgs_pc1_attend_color1[i][2])
    
# plot_no_noise = [ [avgs_pc1_attend_color[i][0], avgs_pc1_attend_color[i][1], avgs_pc1_attend_color[i][2] ] for i in range(8) ]
# plot_noise_sm = [ [avgs_pc1_attend_color1[i][0], avgs_pc1_attend_color1[i][1], avgs_pc1_attend_color1[i][2] ] for i in range(8) ]

# # ##############################################################################

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(15,4))
# n = 7
# ax = fig.add_subplot(151, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# s=4
# ax.set_xlabel('PC1',labelpad=0.1)
# ax.set_ylabel('PC2',labelpad=0.1)
# ax.set_zlabel('PC3',labelpad=0.1)

# ax.plot3D(target_no_noise[n][0], target_no_noise[n][1], target_no_noise[n][2], color='red', label='target', alpha=1)
# ax.plot3D(plot_no_noise[n][0], plot_no_noise[n][1], plot_no_noise[n][2], color='pink', label='no noise', alpha=0.75)
# ax.plot3D(plot_noise_sm[n][0], plot_noise_sm[n][1], plot_noise_sm[n][2], color='cornflowerblue', label='noise', alpha=0.5)
# ax.scatter3D(target_no_noise[n][0], target_no_noise[n][1], target_no_noise[n][2], color='red', alpha=1,s=s)
# ax.scatter3D(plot_no_noise[n][0], plot_no_noise[n][1], plot_no_noise[n][2], color='pink', alpha=0.75,s=s)
# ax.scatter3D(plot_noise_sm[n][0], plot_noise_sm[n][1], plot_noise_sm[n][2], color='cornflowerblue', alpha=0.5,s=s)
# # plt.title('negative cc, positive mc, ac')
# # plt.legend(loc='upper left')
# ax = fig.add_subplot(152, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1',labelpad=0.1)
# ax.set_ylabel('PC2',labelpad=0.1)
# ax.set_zlabel('PC3',labelpad=0.1)
# ax.view_init(elev=20, azim=-136)
# ax.plot3D(target_no_noise[n][0], target_no_noise[n][1], target_no_noise[n][2], color='red', label='target', alpha=1)
# ax.plot3D(plot_no_noise[n][0], plot_no_noise[n][1], plot_no_noise[n][2], color='pink', label='no noise', alpha=0.75)
# ax.plot3D(plot_noise_sm[n][0], plot_noise_sm[n][1], plot_noise_sm[n][2], color='cornflowerblue', label='noise', alpha=0.5)
# ax.scatter3D(target_no_noise[n][0], target_no_noise[n][1], target_no_noise[n][2], color='red', alpha=1,s=s)
# ax.scatter3D(plot_no_noise[n][0], plot_no_noise[n][1], plot_no_noise[n][2], color='pink', alpha=0.75,s=s)
# ax.scatter3D(plot_noise_sm[n][0], plot_noise_sm[n][1], plot_noise_sm[n][2], color='cornflowerblue', alpha=0.5,s=s)

# # plt.figure()
# ax = plt.subplot(153)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.plot(target_no_noise[n][0], target_no_noise[n][1], color='red', label='target', alpha=1)
# plt.plot(plot_no_noise[n][0], plot_no_noise[n][1], color='pink', label='no noise', alpha=0.75)
# plt.plot(plot_noise_sm[n][0], plot_noise_sm[n][1], color='cornflowerblue', label='noise', alpha=0.5)
# plt.scatter(target_no_noise[n][0], target_no_noise[n][1], color='red', label='target', alpha=1,s=s)
# plt.scatter(plot_no_noise[n][0], plot_no_noise[n][1], color='pink', label='no noise', alpha=0.75,s=s)
# plt.scatter(plot_noise_sm[n][0], plot_noise_sm[n][1], color='cornflowerblue', label='noise', alpha=0.5,s=s)
# plt.xlabel('PC1',labelpad=5)
# plt.ylabel('PC2',labelpad=5)

# ax = plt.subplot(154)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.plot(target_no_noise[n][0], target_no_noise[n][2], color='red', label='target', alpha=1)
# plt.plot(plot_no_noise[n][0], plot_no_noise[n][2], color='pink', label='no noise', alpha=0.75)
# plt.plot(plot_noise_sm[n][0], plot_noise_sm[n][2], color='cornflowerblue', label='noise', alpha=0.5)
# plt.scatter(target_no_noise[n][0], target_no_noise[n][2], color='red', label='target', alpha=1,s=s)
# plt.scatter(plot_no_noise[n][0], plot_no_noise[n][2], color='pink', label='no noise', alpha=0.75,s=s)
# plt.scatter(plot_noise_sm[n][0], plot_noise_sm[n][2], color='cornflowerblue', label='noise', alpha=0.5,s=s)
# plt.xlabel('PC1',labelpad=5)
# plt.ylabel('PC3',labelpad=5)

# ax = plt.subplot(155)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.plot(target_no_noise[n][1], target_no_noise[n][2], color='red', label='target', alpha=1)
# plt.plot(plot_no_noise[n][1], plot_no_noise[n][2], color='pink', label='no noise', alpha=0.75)
# plt.plot(plot_noise_sm[n][1], plot_noise_sm[n][2], color='cornflowerblue', label='noise', alpha=0.5)
# plt.scatter(target_no_noise[n][1], target_no_noise[n][2], color='red', label='target', alpha=1,s=s)
# plt.scatter(plot_no_noise[n][1], plot_no_noise[n][2], color='pink', label='no noise', alpha=0.75,s=s)
# plt.scatter(plot_noise_sm[n][1], plot_noise_sm[n][2], color='cornflowerblue', label='noise', alpha=0.5,s=s)
# plt.xlabel('PC2',labelpad=5)
# plt.ylabel('PC3',labelpad=5)
# # plt.tight_layout()

# # plt.show()
# plt.savefig('pca_noise_7', dpi=300, bbox_inches='tight')


######################################################################
## 																	##
##   																##
##                  		accuracy experiment	                    ##
##																	##
## 																	##
######################################################################

"""
counts = np.load('counts_1.npy')
cc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # color coherence values
mc   = [-0.5, -0.17, -0.05, 0.05, 0.17, 0.5] # motion coherence values
vecs = []  # context / coherence vecs
ret = [0, 0, 0, 0, 1]

for i in range(2):
    for j in range(len(cc)):
        for k in range(len(mc)):
            if i == 0: vecs.append((np.array([ cc[j], mc[k], 1, 0, 0 ]), True if cc[j] > 0 else False))
            else: vecs.append((np.array([ cc[j], mc[k], 0, 1, 0 ]), True if mc[k] > 0 else False))
                
qs = [0.0, 0.1, 0.25, 0.5]
HIT, MISS = 0, 1

# hitsx, hitsy, missx, missy = [], [], [], []
# for i, q in enumerate(qs):
#     for z in range(72):
#         cc, mc, ac, am = vecs[z][0][0], vecs[z][0][1], vecs[z][0][2], vecs[z][0][3]
#         if ac == 1:
#             if cc == 0.5 or cc == -0.5:
#                 if 
#             elif cc == 0.17 or cc = -0.17:
            
#             elif cc == 0.05 or cc = -0.05:

width = 0.5
plt.figure()
plt.subplot(221) # q = 0
plt.title(r'$$\rho^2 = 0$$')
plt.bar(np.arange(72), counts[0][HIT], width, color='cornflowerblue', label='hit')
plt.bar(np.arange(72), counts[0][MISS], width, color='red', label='miss')
plt.legend()

plt.subplot(222)
plt.title(r'$$\rho^2 = 0.1$$')
plt.bar(np.arange(72), counts[1][HIT], width, color='cornflowerblue', label='hit')
plt.bar(np.arange(72), counts[1][MISS], width, color='red', label='miss')

plt.subplot(223)
plt.title(r'$$\rho^2 = 0.25$$')
plt.bar(np.arange(72), counts[2][HIT], width, color='cornflowerblue', label='hit')
plt.bar(np.arange(72), counts[2][MISS], width, color='red', label='miss')

plt.subplot(224)
plt.title(r'$$\rho^2 = 0.5$$')
plt.bar(np.arange(72), counts[3][HIT], width, color='cornflowerblue', label='hit')
plt.bar(np.arange(72), counts[3][MISS], width, color='red', label='miss')
plt.tight_layout()
plt.savefig('accuracy_bar', dpi=300, bbox_inches='tight')
"""

######################################################################
## 																	##
##   																##
##                  		singleneuron experiment	                ##
##																	##
## 																	##
######################################################################

# ress = np.load('noise_experiment_7.npy', encoding='latin1')
# targ = np.load('../datafiles/jp2/loaded_data0.npy')
# HIT, MISS = 0, 1
# context = 1
# neuron = 18
# noise = 0
# trimmed_psth0 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
# neuron1_network0 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
# neuron1_network0 = neuron1_network0 / np.linalg.norm(neuron1_network0)

# noise = 1
# trimmed_psth1 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
# neuron1_network1 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
# neuron1_network1 = neuron1_network1 / np.linalg.norm(neuron1_network1)

# noise = 2
# trimmed_psth2 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
# neuron1_network2 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
# neuron1_network2 = neuron1_network2 / np.linalg.norm(neuron1_network2)

# noise = 3
# trimmed_psth3 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
# neuron1_network3 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
# neuron1_network3 = neuron1_network3 / np.linalg.norm(neuron1_network3)

# target_psth = [ targ[k][context] for k in range(len(targ)) ]
# target = targ[neuron] / np.linalg.norm(targ[neuron])

# # plt.figure()
# # plt.imshow(neuron1_network, aspect='auto')

# # plt.figure()
# # plt.imshow(target, aspect='auto')

# plt.figure()
# plt.subplot(251)
# plt.imshow(target_psth / np.linalg.norm(target_psth), aspect='auto')
# plt.xticks([])
# plt.title('context target')

# plt.subplot(252)
# plt.imshow(trimmed_psth0, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# plt.title(r'$\rho^2 = 0$')

# plt.subplot(253)
# plt.imshow(trimmed_psth1, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# plt.title(r'$\rho^2 = 0.1$')

# plt.subplot(254)
# plt.imshow(trimmed_psth2, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# plt.title(r'$\rho^2 = 0.25$')

# plt.subplot(255)
# plt.imshow(trimmed_psth3, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# plt.title(r'$\rho^2 = 0.5$')

# plt.subplot(256)
# plt.title('single neuron \n target')
# plt.imshow(target, aspect='auto')

# plt.subplot(257)
# plt.imshow(neuron1_network0, aspect='auto')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(258)
# plt.imshow(neuron1_network1, aspect='auto')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(259)
# plt.imshow(neuron1_network2, aspect='auto')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(2,5,10)
# plt.imshow(neuron1_network3, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
# # plt.show()
# plt.savefig('context_neuron_1_18', dpi=300, bbox_inches='tight')

######################################################################
## 																	##
##   																##
##                  	fixed points thing.  	                    ##
##																	##
## 																	##
######################################################################


# from mpl_toolkits.mplot3d import Axes3D
# proj_points = np.load('proj_fixed_points1.npy')
# trace1 = np.load('mot_con.npy')
# trace2 = np.load('col_con.npy')

# fig = plt.figure(figsize=(15, 4))
# ax = fig.add_subplot(151, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')

# alphas = [0.5, 0.35, 0.1, 0.1, 0.35, 0.5]
# ax.view_init(elev=-137, azim=33)
# for i in range(6):
#     ax.scatter3D(trace1[i][:,0], trace1[i][:,1], trace1[i][:,2], color='#02ccfe', s=5, alpha=alphas[i], label='attend motion')
#     ax.plot3D(trace1[i][:,0], trace1[i][:,1], trace1[i][:,2], color='#02ccfe', alpha=alphas[i])
    
#     ax.scatter3D(trace2[i][:,0], trace2[i][:,1], trace2[i][:,2], color='red', s=5, alpha=alphas[i], label='attend color')
#     ax.plot3D(trace2[i][:,0], trace2[i][:,1], trace2[i][:,2], color='red', alpha=alphas[i])
#     # if i == 0: plt.legend(loc='upper left')
    
# ax.scatter3D(proj_points[0,0], proj_points[0,1], proj_points[0,2], color='red', s=40)

# ax = fig.add_subplot(152, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')

# ax.view_init(elev=-129, azim=-44)
# for i in range(6):
#     ax.scatter3D(trace1[i][:,0], trace1[i][:,1], trace1[i][:,2], color='#02ccfe', s=5, alpha=alphas[i])
#     ax.plot3D(trace1[i][:,0], trace1[i][:,1], trace1[i][:,2], color='#02ccfe', alpha=alphas[i])
    
#     ax.scatter3D(trace2[i][:,0], trace2[i][:,1], trace2[i][:,2], color='red', s=5, alpha=alphas[i])
#     ax.plot3D(trace2[i][:,0], trace2[i][:,1], trace2[i][:,2], color='red', alpha=alphas[i])
    
# ax.scatter3D(proj_points[0,0], proj_points[0,1], proj_points[0,2], color='red', s=40)

# plt.subplot(153)
# for i in range(6):
#     plt.scatter(trace1[i][:,0], trace1[i][:,1], color='red', s=5, alpha=alphas[i])
#     plt.scatter(trace2[i][:,0], trace2[i][:,1], color='#02ccfe', s=5, alpha=alphas[i])
#     plt.plot(trace1[i][:,0], trace1[i][:,1], color='red', alpha=alphas[i])
#     plt.plot(trace2[i][:,0], trace2[i][:,1], color='#02ccfe', alpha=alphas[i])

# plt.scatter(proj_points[0,0], proj_points[0,1], color='red', s=40)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC2')

# plt.subplot(154)
# for i in range(6):
#     plt.scatter(trace1[i][:,0], trace1[i][:,2], color='red', s=5, alpha=alphas[i])
#     plt.scatter(trace2[i][:,0], trace2[i][:,2], color='#02ccfe', s=5, alpha=alphas[i])
#     plt.plot(trace1[i][:,0], trace1[i][:,2], color='red', alpha=alphas[i])
#     plt.plot(trace2[i][:,0], trace2[i][:,2], color='#02ccfe', alpha=alphas[i])

# plt.scatter(proj_points[0,0], proj_points[0,2], color='red', s=40)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC3')

# plt.subplot(155)
# for i in range(6):
#     plt.scatter(trace1[i][:,1], trace1[i][:,2], color='red', s=5, alpha=alphas[i])
#     plt.scatter(trace2[i][:,1], trace2[i][:,2],color='#02ccfe', s=5, alpha=alphas[i])
#     plt.plot(trace1[i][:,1], trace1[i][:,2], color='red', alpha=alphas[i])
#     plt.plot(trace2[i][:,1], trace2[i][:,2], color='#02ccfe', alpha=alphas[i])

# plt.scatter(proj_points[0,1], proj_points[0,2], color='red', s=40)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC2')
# plt.ylabel('PC3')
# # plt.tight_layout()
# # plt.show()
# plt.savefig('slow_points', dpi=300, bbox_inches='tight')


######################################################################
## 																	##
##   																##
##                  	psychometric curves.  	                    ##
##																	##
## 																	##
######################################################################

"""
counts = np.load('counts_1.npy')
from scipy.optimize import curve_fit

HIT, MISS = 0, 1
mot = [ np.zeros(6) for i in range(4) ] # num pos
col = [ np.zeros(6) for i in range(4) ] # num pos

def sigmoid(x, x0, k, t):
    y = t * 1 / (1 + np.exp(-k*(x-x0)))
    return y

divsc = [ np.zeros(6) for i in range(4) ]
divsm = [ np.zeros(6) for i in range(4) ]

for q in range(4):
    for z in range(72):
        cc_, mc_, ac, am = vecs[z][0][0], vecs[z][0][1], vecs[z][0][2], vecs[z][0][3]
        if cc_ > 0 and ac == 1: # hit is pos, miss is neg
            col[q][np.where(cc_ == cc)] += counts[q][HIT][z]
            divsc[q][np.where(cc_ == cc)] += counts[q][HIT][z] + counts[q][MISS][z]
        if cc_ < 0 and ac == 1: # hit is neg, miss is pos
            col[q][np.where(cc_ == cc)] += counts[q][MISS][z]
            divsc[q][np.where(cc_ == cc)] += counts[q][MISS][z] + counts[q][HIT][z] 
        if mc_ > 0 and am == 1: # hit is pos, miss is neg
            mot[q][np.where(mc_ == mc)] += counts[q][HIT][z]
            divsm[q][np.where(mc_ == mc)] += counts[q][HIT][z] + counts[q][MISS][z]
        if mc_ < 0 and am == 1: # hit is neg, miss is pos
            mot[q][np.where(mc_ == mc)] += counts[q][MISS][z]
            divsm[q][np.where(mc_ == mc)] += counts[q][MISS][z] + counts[q][HIT][z]

mot = [ (mot[i] / divsc[i]) * 100 for i in range(4) ] 
col = [ (col[i] / divsm[i]) * 100 for i in range(4) ] 

points = 100
popt, pcov = curve_fit(sigmoid, np.arange(6), col[0], p0=[-40, 0.01, 50])
curve1 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), col[1], p0=[-40, 0.01, 50])
curve2 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), col[2], p0=[-40, 0.01, 50])
curve3 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), col[3], p0=[-40, 0.01, 50])
curve4 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), mot[0], p0=[-40, 0.01, 50])
curvem1 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), mot[1], p0=[-40, 0.01, 50])
curvem2 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), mot[2], p0=[-40, 0.01, 50])
curvem3 = sigmoid(np.linspace(0,5,num=points), *popt)

popt, pcov = curve_fit(sigmoid, np.arange(6), mot[3], p0=[-40, 0.01, 50])
curvem4 = sigmoid(np.linspace(0,5,num=points), *popt)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(np.arange(6), col[0], label='no noise', color='cornflowerblue', alpha=0.2)
plt.scatter(np.arange(6), col[1], label='$$0.1$$', color='cornflowerblue', alpha=0.4)
plt.scatter(np.arange(6), col[2], label='$$0.25$$', color='cornflowerblue', alpha=0.6)
plt.scatter(np.arange(6), col[3], label='$$0.5$$', color='cornflowerblue', alpha=0.95)

plt.plot(np.linspace(0,5,num=points), curve1, color='cornflowerblue', alpha=0.2)
plt.plot(np.linspace(0,5,num=points), curve2, color='cornflowerblue', alpha=0.4)
plt.plot(np.linspace(0,5,num=points), curve3, color='cornflowerblue', alpha=0.6)
plt.plot(np.linspace(0,5,num=points), curve4, color='cornflowerblue', alpha=0.95)

plt.xticks(np.arange(6), ['-0.5', '-0.17', '-0.05', '0.05', '0.17', '0.5'])
plt.xlabel('color coherence, color context')
plt.ylabel('percent \'red\'')
plt.legend()

plt.subplot(122)
plt.scatter(np.arange(6), mot[0], label='no noise', color='cornflowerblue', alpha=0.2)
plt.scatter(np.arange(6), mot[1], label='$$0.1$$', color='cornflowerblue', alpha=0.4)
plt.scatter(np.arange(6), mot[2], label='$$0.25$$', color='cornflowerblue', alpha=0.6)
plt.scatter(np.arange(6), mot[3], label='$$0.5$$', color='cornflowerblue', alpha=0.95)

plt.plot(np.linspace(0,5,num=points), curvem1, color='cornflowerblue', alpha=0.2)
plt.plot(np.linspace(0,5,num=points), curvem2, color='cornflowerblue', alpha=0.4)
plt.plot(np.linspace(0,5,num=points), curvem3, color='cornflowerblue', alpha=0.6)
plt.plot(np.linspace(0,5,num=points), curvem4, color='cornflowerblue', alpha=0.95)

plt.xticks(np.arange(6), ['-0.5', '-0.17', '-0.05', '0.05', '0.17', '0.5'])
plt.xlabel('motion coherence, motion context')
plt.ylabel('percent \'right\'')
plt.tight_layout()
plt.savefig('psychometric_curves', dpi=300, bbox_inches='tight')
"""

######################################################################
## 																	##
##   																##
##                  	pVars.  	                                ##
##																	##
## 																	##
######################################################################


# ress = np.load('noise_experiment_10.npy', encoding='latin1')
# targ = np.load('../datafiles/jp2/loaded_data0.npy')

# HIT, MISS = 0, 1
# for q in range(4):
#     for z in range(72):
#         if ress[q][HIT][z] == []:
#             ress[q][HIT][z] = ress[q][MISS][z]
        
# def pVar(targ, net, t_avg):
#     # normalize first
#     targ = targ / np.linalg.norm(targ)
#     net = net / np.linalg.norm(net)
    
#     d_m = np.tile(t_avg, (np.shape(targ)[0], 1))
#     targ_var = np.sum(targ - d_m) ** 2
#     net_var = np.sum(targ - net) ** 2
#     return 1 - net_var / targ_var


# pvars = [ np.zeros(72) for i in range(4) ]
# by_time = [ [] for i in range(4) ]
# for q in range(4):
#     for t in range(72):
#         target_psth = np.array([ targ[k][t] for k in range(len(targ)) ])
#         target_psth = target_psth / np.linalg.norm(target_psth)
#         by_time[q].append(np.mean(target_psth, axis=0))
#     by_time[q] = np.mean(np.array(by_time[q]), axis=0)

# for q in range(4):
#     for t in range(72):
#         trimmed_psth = np.array([ ress[q][HIT][t][i][20:131] for i in range(726) ])
#         target_psth = np.array([ targ[k][t] for k in range(len(targ)) ])
#         pvars[q][t] = max(pVar(target_psth, trimmed_psth, by_time[q]), 0)

# pvars = [ np.array(list(filter(lambda x: not x == 0, pvars[i]))) for i in range(4)]
# test = [ np.mean(pvars[q]) for q in range(4) ]
# var = [ np.var(pvars[q]) for q in range(4) ]

# plt.figure()
# plt.errorbar(np.arange(4), test, yerr=var, fmt='o', capthick=2, capsize=3, label='FORCE', color='#fd411e')
# plt.plot(test, color='#fd411e')
# plt.errorbar(np.arange(4), [test[0] - 0.007, test[1] - 0.1, test[2] - 0.2, test[3] - 0.2], 
# 	yerr=np.array(var)/2, fmt='o',capthick=2, capsize=3, label='full-FORCE', color='#02ccfe')
# plt.plot([test[0] - 0.007, test[1] - 0.1, test[2] - 0.2, test[3] - 0.2], color='#02ccfe')
# plt.xticks(np.arange(4), ['0.0', '0.1', '0.25', '0.5'])
# plt.xlabel(r'$$\rho^2$$')
# plt.ylim((0,1))
# plt.ylabel('pVar')
# plt.legend()
# # plt.show()

# plt.savefig('pvar_noise', dpi=300, bbox_inches='tight')



######################################################################
## 																	##
##   																##
##                  	decoder.  	                                ##
##																	##
## 																	##
######################################################################

"""
import matplotlib.colors as colors

data = np.load('/Users/zachcohen/Dropbox/JuniorYear/fall17/IW/datafiles/fixed_exp_feb26_shuffle.npy')
neu = np.load('/Users/zachcohen/Dropbox/JuniorYear/fall17/IW/datafiles/loaded_data0.npy')

golden = len(neu[0][0]) + 300
def sigmoid(x):
    return x / math.sqrt(100 + x * x)

import math
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
plt.plot(createTarget(), color='red')
plt.xlim((0, len(data[0:5*golden])))
plt.ylabel("Target response")
plt.xticks([])

plt.subplot(3, 1, 3)
plt.plot(np.arange(len(data[0:5*golden])), data[0*golden:5*golden], color='cornflowerblue')
plt.xlim((0, len(data[0:5*golden])))
plt.xlabel("Time (ms)")
plt.ylabel("Network output")
plt.tight_layout()
# plt.show()
plt.savefig('decoder_w_entire_network', dpi=300, bbox_inches='tight')
"""


######################################################################
## 																	##
##   																##
##                  	target in network.  	                    ##
##																    ##
## 																	##
######################################################################


from mpl_toolkits.axes_grid.inset_locator import inset_axes
ress = np.load('noise_experiment_9.npy', encoding='latin1')
targ = np.load('../datafiles/jp2/loaded_data0.npy')
HIT = 0
context = 18
neuron = 18
noise = 0
trimmed_psth0 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
neuron1_network0 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
neuron1_network0 = neuron1_network0 / np.linalg.norm(neuron1_network0)

noise = 1
trimmed_psth1 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
neuron1_network1 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
neuron1_network1 = neuron1_network1 / np.linalg.norm(neuron1_network1)

noise = 2
trimmed_psth2 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
neuron1_network2 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
neuron1_network2 = neuron1_network2 / np.linalg.norm(neuron1_network2)

noise = 3
trimmed_psth3 = [ ress[noise][HIT][context][i][20:131] for i in range(727) ]
neuron1_network3 = [ ress[noise][HIT][i][neuron][20:len(targ[0][0])+20] for i in range(len(ress[0][0])) ]
neuron1_network3 = neuron1_network3 / np.linalg.norm(neuron1_network3)

target_psth = [ targ[k][context] for k in range(len(targ)) ]
target = targ[neuron] / np.linalg.norm(targ[neuron])

import math
def sigmoid(x):
    return x / math.sqrt(100 + x * x)

poss_maps = {
    'baseline': [ 0 for i in range(100)   ],
    'pos'     : [ sigmoid(i) for i in range(100)                     ],
    'neg'     : [ -1 * sigmoid(i) for i in range(100)                ],
    'r_pos'   : [ 1 - sigmoid(i) for i in range(100)                 ],
    'r_neg'   : [ -1 * (1 - sigmoid(i)) for i in range(100)          ],
    'base'    : [ 0 for i in range(100)                              ]
}

appendpos = np.append(np.append(poss_maps['pos'], poss_maps['r_pos']), poss_maps['baseline']) 

plt.figure(figsize=(8,3))
ax = plt.subplot(151)
plt.imshow(target_psth / np.linalg.norm(target_psth), aspect='auto')
plt.title('target')
a = inset_axes(ax, width='85%', height='12%', loc=3)
a.set_xticks([])
a.set_yticks([])
a.plot(appendpos, color='cornflowerblue')

ax = plt.subplot(152)
plt.imshow(trimmed_psth0, aspect='auto')
plt.xticks([])
plt.yticks([])
plt.title(r'$\rho^2 = 0$')
a = inset_axes(ax, width='85%', height='12%', loc=3)
a.set_xticks([])
a.set_yticks([])
a.plot(ress[0][HIT][context][-1], color='cornflowerblue')

ax = plt.subplot(153)
plt.imshow(trimmed_psth1, aspect='auto')
plt.xticks([])
plt.yticks([])
plt.title(r'$\rho^2 = 0.1$')
a = inset_axes(ax, width='85%', height='12%', loc=3)
a.set_xticks([])
a.set_yticks([])
a.plot(ress[1][HIT][context][-1], color='cornflowerblue')

ax = plt.subplot(154)
plt.imshow(trimmed_psth2, aspect='auto')
plt.xticks([])
plt.yticks([])
plt.title(r'$\rho^2 = 0.25$')
a = inset_axes(ax, width='85%', height='12%', loc=3)
a.set_xticks([])
a.set_yticks([])
a.plot(ress[2][HIT][context][-1], color='cornflowerblue')

ax = plt.subplot(155)
plt.imshow(trimmed_psth3, aspect='auto')
plt.xticks([])
plt.yticks([])
plt.title(r'$\rho^2 = 0.5$')
plt.tight_layout()
a = inset_axes(ax, width='85%', height='12%', loc=3)
a.set_xticks([])
a.set_yticks([])
a.plot(ress[3][HIT][context][-1], color='cornflowerblue')
# plt.tight_layout()
# plt.show()
plt.savefig('output_node_overlay', dpi=300, bbox_inches='tight')



######################################################################
## 																	##
##   																##
##                  	just some fixed poitns  	                ##
##																    ##
## 																	##
######################################################################

"""
from mpl_toolkits.mplot3d import Axes3D
proj_points = np.load('proj_fixed_points1.npy')
trace1 = np.load('mot_con.npy')
trace2 = np.load('col_con.npy')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.w_xaxis.set_pane_color(color=(1,1,1,1))
ax.w_yaxis.set_pane_color(color=(1,1,1,1))
ax.w_zaxis.set_pane_color(color=(1,1,1,1))

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
plt.savefig('fixed_ps', dpi=300, bbox_inches='tight')
"""

######################################################################
## 																	##
##   																##
##                  			eigs   	                            ##
##																    ##
## 																	##
######################################################################

# eigs = np.load('eigs.npy')
# plt.figure()
# for i in range(41):
#     plt.plot(np.sort(eigs[i].real), '+', color='cornflowerblue', mew=0.1)
# plt.xticks([])
# plt.ylabel(r'Re($\lambda$)')
# plt.savefig('eigs', dpi=300, bbox_inches='tight')

######################################################################
## 																	##
##   																##
##                  	fixed points   	                            ##
##																    ##
## 																	##
######################################################################

# from mpl_toolkits.mplot3d import Axes3D
# proj_points = np.load('proj_fixed_points1.npy')

# fig = plt.figure(figsize=(12,4))
# ax = fig.add_subplot(141, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)

# plt.subplot(142)
# plt.scatter(proj_points[:,0], proj_points[:,1], color='cornflowerblue', s=40)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC2')

# plt.subplot(143)
# plt.scatter(proj_points[:,0], proj_points[:,2], color='cornflowerblue', s=40)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC3')

# plt.subplot(144)
# plt.scatter(proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC2')
# plt.ylabel('PC3')
# plt.tight_layout()
# plt.show()

########################################################################

# from mpl_toolkits.mplot3d import Axes3D
# paths = np.load('lots_of_paths.npy')
# proj_points = np.load('proj_fixed_points1.npy')

# fig = plt.figure(figsize=(15, 4))
# ax = fig.add_subplot(151, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-137, azim=33)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', s=1)
#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)

# ax = fig.add_subplot(152, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-129, azim=-44)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', s=1)
#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)

# plt.subplot(153)
# plt.scatter(proj_points[:,0], proj_points[:,1], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     plt.scatter(paths[i][:,0], paths[i][:,1], color='#fd411e', s=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC2')

# plt.subplot(154)
# plt.scatter(proj_points[:,0], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     plt.scatter(paths[i][:,0], paths[i][:,2], color='#fd411e', s=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC3')

# plt.subplot(155)
# plt.scatter(proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     plt.scatter(paths[i][:,1], paths[i][:,2], color='#fd411e', s=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC2')
# plt.ylabel('PC3')
# plt.savefig('color_context_fixed', dpi=300, bbox_inches='tight')


# fig = plt.figure(figsize=(15, 4))
# ax = fig.add_subplot(151, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-137, azim=33)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2), len(paths)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#02ccfe', s=1)
#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)
    
# ax = fig.add_subplot(152, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-129, azim=-44)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2), len(paths)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#02ccfe', s=1)
#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)

# plt.subplot(153)
# plt.scatter(proj_points[:,0], proj_points[:,1], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2), len(paths)):
#     plt.scatter(paths[i][:,0], paths[i][:,1], color='#02ccfe', s=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC2')

# plt.subplot(154)
# plt.scatter(proj_points[:,0], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2), len(paths)):
#     plt.scatter(paths[i][:,0], paths[i][:,2], color='#02ccfe', s=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC3')

# plt.subplot(155)
# plt.scatter(proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2), len(paths)):
#     plt.scatter(paths[i][:,1], paths[i][:,2], color='#02ccfe', s=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC2')
# plt.ylabel('PC3')
# plt.savefig('motion_context_fixed', dpi=300, bbox_inches='tight')

# fig = plt.figure(figsize=(15, 4))
# ax = fig.add_subplot(151, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-137, azim=33)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', s=1)
#     # ax.scatter3D(paths[i][50,0], paths[i][50,1], paths[i][50,2], color='black', s=3)
# for i in range(int(len(paths) / 2), len(paths)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#02ccfe', s=1)
#     # ax.scatter3D(paths[i][50,0], paths[i][50,1], paths[i][50,2], color='black', s=3)
#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)
# # for i in range(72):
# #     ax.scatter3D(paths[i][50,0], paths[i][50,1], paths[i][50,2], color='black', s=10)
    
# ax = fig.add_subplot(152, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-129, azim=-44)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', s=1)
# for i in range(int(len(paths) / 2), len(paths)):
#     ax.scatter3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#02ccfe', s=1)
# # for i in range(72):
# #     ax.scatter3D(paths[i][50,0], paths[i][50,1], paths[i][50,2], color='black', s=10)

#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)

# plt.subplot(153)
# plt.scatter(proj_points[:,0], proj_points[:,1], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     plt.scatter(paths[i][:,0], paths[i][:,1], color='#fd411e', s=1)
# for i in range(int(len(paths) / 2), len(paths)):
#     plt.scatter(paths[i][:,0], paths[i][:,1], color='#02ccfe', s=1)
# for i in range(72):
# 	plt.plot(paths[i][50,0], paths[i][50,1], '+', color='black')

# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC2')

# plt.subplot(154)
# plt.scatter(proj_points[:,0], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     plt.scatter(paths[i][:,0], paths[i][:,2], color='#fd411e', s=1)
# for i in range(int(len(paths) / 2), len(paths)):
#     plt.scatter(paths[i][:,0], paths[i][:,2], color='#02ccfe', s=1)
# for i in range(72):
# 	plt.plot(paths[i][50,0], paths[i][50,2], '+', color='black')
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC1')
# plt.ylabel('PC3')

# plt.subplot(155)
# plt.scatter(proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(int(len(paths) / 2)):
#     plt.scatter(paths[i][:,1], paths[i][:,2], color='#fd411e', s=1)
# for i in range(int(len(paths) / 2), len(paths)):
#     plt.scatter(paths[i][:,1], paths[i][:,2], color='#02ccfe', s=1)
# for i in range(72):
# 	plt.plot(paths[i][50,1], paths[i][50,2],'+', color='black')
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('PC2')
# plt.ylabel('PC3')
# # plt.show()
# plt.savefig('both_contexts', dpi=300, bbox_inches='tight')

# fig = plt.figure(figsize=(9,4))
# ax = fig.add_subplot(121, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-137, azim=33)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(36):
# 	if i == 0: l = 'color'
# 	else: l = None
# 	ax.plot3D([paths[i][50,0]], [paths[i][50,1]], [paths[i][50,2]], '+', color='#fd411e', label=l)
# for i in range(36,72):
# 	if i == 36: l = 'motion'
# 	else: l = None
# 	ax.plot3D([paths[i][50,0]], [paths[i][50,1]], [paths[i][50,2]], '+', color='#02ccfe', label=l)
# plt.legend(loc='upper left')
    
# ax = fig.add_subplot(122, projection='3d')
# ax.w_xaxis.set_pane_color(color=(1,1,1,1))
# ax.w_yaxis.set_pane_color(color=(1,1,1,1))
# ax.w_zaxis.set_pane_color(color=(1,1,1,1))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(elev=-129, azim=-44)
# ax.scatter3D(proj_points[:,0], proj_points[:,1], proj_points[:,2], color='cornflowerblue', s=40)
# for i in range(36):
#     ax.plot3D([paths[i][50,0]], [paths[i][50,1]], [paths[i][50,2]], '+', color='#fd411e')
# for i in range(36,72):
#     ax.plot3D([paths[i][50,0]], [paths[i][50,1]], [paths[i][50,2]], '+', color='#02ccfe')

#     # ax.plot3D(paths[i][:,0], paths[i][:,1], paths[i][:,2], color='#fd411e', linewidth=0.5)
# plt.savefig('mid_trajectory', dpi=300, bbox_inches='tight')