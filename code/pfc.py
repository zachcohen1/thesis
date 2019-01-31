import scipy as sp
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from pfcaux import *

def load_all_files(directory):
    return [ f for f in listdir(directory) ]

def num_files(head):
    return len(load_all_files(head))

def gatherdata(head, dm, start=0, files_to_load=-1):
    files = load_all_files(head)
    if files_to_load == -1: files_to_load = len(files)
    conv_window = win(40)
    agg = []
    c = 0
    for file_ in files[start:files_to_load]:
        mat = load_pfc(head + file_)
        neu, mot, col = select_pfc_recording(mat), \
                        select_signal(mat, stream='motion'), \
                        select_signal(mat)
        neu = boxresample(neu)
        neu = signalresample(neu)
        a1 = sort_by_condition(neu, col, mot, select_correct(mat), \
            select_context(mat), discard_mistrials=dm)
        if a1 is None: continue

        neu_smooth = smooth(a1, conv_window)
        mean, std = score(neu_smooth)
        agg.append((np.array(neu_smooth) - mean)/(std+c))
    return agg

"""
head = '/Users/zachcohen/Dropbox/ManteData_justdata/PFC data 1/'
agg = gatherdata(head, True, files_to_load=50)
len(agg[0][0])

plt.figure(1)
for i in range(72):
    plt.plot(np.arange(len(agg[18][i])), agg[18][i])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()


plt.figure(1)
for i in range(72):
    plt.plot(np.arange(len(agg[1][i])), agg[1][i])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()


plt.figure(1)
for i in range(72):
    plt.plot(np.arange(len(agg[19][i])), agg[19][i])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()


plt.figure(1)
for i in range(72):
    plt.plot(np.arange(len(agg[5][i])), agg[5][i])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()


plt.figure(1)
for i in range(len(agg)):
    plt.plot(np.arange(len(agg[i][6])), agg[i][6])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()



plt.figure(1)
for i in range(len(agg)):
    plt.plot(np.arange(len(agg[i][36])), agg[i][36])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()


plt.figure(1)
for i in range(len(agg)):
    plt.plot(np.arange(len(agg[i][71])), agg[i][71])
plt.axis([-10, 120, -3.0, 4.0])
plt.show()



plt.figure(figsize=(8,20))
plt.imshow(agg[5], cmap='hot', interpolation='nearest')
plt.show()

plt.figure(figsize=(8,20))
plt.imshow(agg[100], cmap='hot', interpolation='nearest')
plt.show()

# BINGGOOOOO!!!!
plt.figure(figsize=(8,20))
plt.imshow(agg[18], cmap='hot', interpolation='nearest')
plt.show()

plt.figure(figsize=(8,20))
plt.imshow(agg[19], cmap='hot', interpolation='nearest')
plt.show()

plt.figure(figsize=(8,20))
plt.imshow(agg[20], cmap='hot', interpolation='nearest')
plt.show()

plt.figure(figsize=(8,20))
plt.imshow(agg[30], cmap='hot', interpolation='nearest')
plt.show()

plt.figure(figsize=(8,20))
plt.imshow(agg[31], cmap='hot', interpolation='nearest')
plt.show()

## BINGO!!!!
plt.figure(figsize=(8,20))
plt.imshow(agg[131], cmap='hot', interpolation='nearest')
plt.show()

plt.figure(figsize=(8,20))
plt.imshow(agg[132], cmap='hot', interpolation='nearest')
plt.show()

strongcolor = [ agg[i][0] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(strongcolor, cmap='hot', interpolation='nearest')
plt.show()

strongcolor1 = [ agg[i][1] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(strongcolor1, cmap='hot', interpolation='nearest')
plt.show()

strongcolor2 = [ agg[i][2] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(strongcolor2, cmap='hot', interpolation='nearest')
plt.show()

strongcolor3 = [ agg[i][3] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(strongcolor3, cmap='hot', interpolation='nearest')
plt.show()

strongcolor4 = [ agg[i][4] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(strongcolor4, cmap='hot', interpolation='nearest')
plt.show()

attendmot = [ agg[i][36] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(attendmot, cmap='hot', interpolation='nearest')
plt.show()

np.save('compressed_neu.npy', agg)
attendmot = [ agg[i][36] for i in range(len(agg)) ]
plt.figure(figsize=(8,20))
plt.imshow(attendmot, cmap='hot')
plt.show()


attendmot = [ agg[i][0] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()


attendmot = [ agg[i][1] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()




attendmot = [ agg[i][3] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()

attendmot = [ agg[i][4] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()


attendmot = [ agg[i][18] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()

attendmot = [ agg[i][6] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()

attendmot = [ agg[i][71] for i in range(len(agg)) ]
im = plt.imshow(attendmot, cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.tight_layout()
plt.show()

im = plt.imshow(agg[18], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()

im = plt.imshow(agg[19], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()

im = plt.imshow(agg[2], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()


im = plt.imshow(agg[3], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()


im = plt.imshow(agg[4], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()


im = plt.imshow(agg[5], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()


im = plt.imshow(agg[6], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()
"""
"""
im = plt.imshow(agg[7], cmap='gnuplot2', interpolation=None)
plt.colorbar(im, orientation="horizontal")
plt.show()

"""
