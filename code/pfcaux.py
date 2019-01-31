import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal, stats
from scipy.signal import gaussian
from scipy.ndimage import filters

def load_pfc(file_name):
    return sio.loadmat(file_name)

def select_pfc_recording(mat_data):
    data = mat_data['unit'][0][0]
    return data[0]

def select_signal(mat, stream='color', all_t=True, at=0):
    u = mat['unit'][0][0]
    cues = []
    if stream == 'color': cues = u[1][0][0][1]
    else: cues = u[1][0][0][0]

    if all_t: return cues
    else: return cues[at]

def select_context(mat, all_t=True, at=0):
    u = mat['unit'][0][0]
    con = u[1][0][0][4]
    if all_t: return con
    else: return con[at]

def select_correct(mat, all_t=True, at=0):
    u = mat['unit'][0][0]
    con = u[1][0][0][5]
    if all_t: return con
    else: return con[at]

def sort_by_coherence(neu, trials, cues, context='color'):
    ind = 0
    count = 0
    targ = np.unique(trials)
    np.sort(targ)

    sort = np.array([ np.arange(len(neu[0])) for i in range(len(neu)) ],
        dtype=np.float32)
    avg = [ [] for i in range(len(targ)) ]

    if context == 'color': right = 1
    else: right = -1

    # sort (not an advanced sort method)
    for i in range(len(targ)):
        start = ind
        for j in range(len(trials)):
            if trials[j] == targ[i]:
                sort[ind] = int(cues[j] == right) * np.copy(np.array(neu[j]))
                ind += 1
                count += int(cues[j] == right)

        if ind-start == 0: avg[i] = 3 * sort[start:ind].sum(axis=0) / 1
        else: avg[i] = 3 * sort[start:ind].sum(axis=0) / count
        count = 0
    return sort, avg

def sort_by_condition(neu, ctrails, mtrials, correct, cues, discard_mistrials=False):
    ctarg = np.sort(np.unique(ctrails)).tolist() # color coherence
    mtarg = np.sort(np.unique(mtrials)).tolist() # motion coherence

    lctarg, lmtarg = len(ctarg), len(mtarg)
    cats           = lctarg * lmtarg * 2
    cindtrack      = [ 0 for i in range(cats) ]

    build     = [ [] for i in range(cats) ]
    avg       = [ np.arange(len(neu[0])) for i in range(cats) ]
    ret       = True

    for i in range(len(neu)):
        con = 0 if cues[i] == 1 else 36 # color = 1, motion = 2
        typ = (ctarg.index(ctrails[i]) * 6 + mtarg.index(mtrials[i])) + con
        if discard_mistrials:
            if correct[i] == 1: build[typ].append(neu[i])
        else: build[typ].append(neu[i])

    if discard_mistrials:
        for b in build:
            if b == []: return None # discard this trial if, for each trial,
                                    # this particular condition saw only error
                                    # trials

    for i in range(len(build)):
        build[i] = np.array(build[i])
        avg[i]   = build[i].sum(axis=0) / len(build[i])

    return avg

def win(scale):
    return gaussian(50, scale)

def smooth(unit_var, win):
    return [ signal.fftconvolve(unit_var[i], win, \
             mode='same') / sum(win) for i in range(len(unit_var)) ]

def score(neu):
    return (np.mean(neu, axis=(0,1)), np.std(neu, axis=(0,1)))

def boxresample(neu):
    resamp = [ [] for t in range(len(neu)) ]
    for i in range(len(neu)):
        for k in range(int(len(neu[i])/20)):
            resamp[i].append(np.mean(neu[i][k*20:(k+1)*20]))
    return resamp
    # return [ np.mean(neu[i][k*20:(k+1)*20]) for i in range(len(neu)) for k in range(int(len(neu[i])/20)) ]
    # return [ signal.resample(neu[i], len(neu[i])) for i in range(len(neu)) ]

def signalresample(neu):
    return [ signal.resample(neu[i], len(neu[i]) * 3) for i in range(len(neu)) ]

def render(signal):
    plt.figure(figsize=(12,12))
    plt.imshow(signal, cmap='hot', interpolation='nearest')
    plt.show()

def show_avgs(neu, avg, savg):
    plt.figure(0)
    # plt.figure(figsize=(12,12))

    #for i in range(len(avg)):
    plt.plot(np.arange(len(savg[70])), savg[70])
    plt.plot(np.arange(len(avg[70])), avg[70])


    plt.show()

"""
mat = load_pfc('/Users/zachcohen/Dropbox/ManteData_justdata/PFC data 1/ar090313_1_a3_Vstim_100_850_ms.mat')
neu_, mot, col = select_pfc_recording(mat), select_signal(mat, stream='motion'), select_signal(mat)
conv_window = win(1)
avg = sort_by_condition(neu_, col, mot, select_context(mat))
neu_smooth = smooth(avg, conv_window)
show_avgs(neu_, neu_smooth, avg)

mat = load_pfc('/Users/zachcohen/Dropbox/ManteData_justdata/PFC data 1/ar090313_1_a3_Vstim_100_850_ms.mat')
neu_, mot, col = select_pfc_recording(mat), select_signal(mat, stream='motion'), select_signal(mat)
conv_window = win(2)
avg = sort_by_condition(neu_, col, mot, select_context(mat))
neu_smooth = smooth(avg, conv_window)
show_avgs(neu_, neu_smooth, avg)

mat = load_pfc('/Users/zachcohen/Dropbox/ManteData_justdata/PFC data 1/ar090313_1_a3_Vstim_100_850_ms.mat')
neu_, mot, col = select_pfc_recording(mat), select_signal(mat, stream='motion'), select_signal(mat)
conv_window = win(7)
avg = sort_by_condition(neu_, col, mot, select_context(mat))
neu_smooth = smooth(avg, conv_window)
show_avgs(neu_, neu_smooth, avg)

mat = load_pfc('/Users/zachcohen/Dropbox/ManteData_justdata/PFC data 1/ar090313_1_a3_Vstim_100_850_ms.mat')
neu_, mot, col = select_pfc_recording(mat), select_signal(mat, stream='motion'), select_signal(mat)
conv_window = win(20)
avg = sort_by_condition(neu_, col, mot, select_context(mat))
neu_smooth = smooth(avg, conv_window)
show_avgs(neu_, neu_smooth, avg)
"""
