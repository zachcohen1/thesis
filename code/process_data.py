import scipy.io as sio
from pfcaux import *

# result of running the first part of the data pre-processing
# from the mante paper which sorts recordings by 'unit', which
# we treat as a single neuron and box averages the spike
# sequences to approximate a firing rate function
mante_stim = sio.loadmat('mante_collected.mat')

def normalize(run, min_, max_):
    return min_ + ((run - np.nanmin(run)) * (max_ - min_)) / (np.nanmax(run) - np.nanmin(run))

NUM_NEURONS = 762; NUM_CONDITIONS = 72; TRIAL_TIME_COURSE = 150;

# intermediate storage of processing results
data     = np.zeros((2, NUM_CONDITIONS, NUM_NEURONS, TRIAL_TIME_COURSE))
datap    = np.zeros((2, NUM_CONDITIONS, NUM_NEURONS, TRIAL_TIME_COURSE))
count    = np.zeros((2, NUM_CONDITIONS, NUM_NEURONS))
corrects = np.zeros((2, NUM_CONDITIONS))

# generate index for first index of data placement
def gen_choice(con, targ_dir, targ_col):
    if con[0] == -1: # color context
        return 0 if targ_col[0] == -1.0 else 1
    else:
        return 0 if targ_dir[0] == -1.0 else 1

# generate condition index
def gen_ind(col, mot, con, ind):
    cont = 1 if con[0] == 1.0 else 0
    cols = list(np.unique(col))
    mots = list(np.unique(mot))
    return int(cont * 36 + int(cols.index(col[ind])) * 6 + int(mots.index(mot[ind])))

for i in range(NUM_NEURONS):
    # annoying compatibility issues with python/matlab...
    record = mante_stim['dataT']['unit'][0][0][i][0]
    response = record[0]
    task_variable = record[1][0][0]
    stim_dir = task_variable[0]
    stim_col = task_variable[1]
    context = task_variable[4]
    correct = task_variable[5]
    for j in range(len(stim_dir)):
        choice = 0 if correct[j] == 1 else 1 # right: 0, wrong: 1
        ind = gen_ind(stim_col, stim_dir, context[j], j)
        corrects[choice][ind] += 1
        data[choice][ind][i] += response[j]
        count[choice][ind][i] += 1

# avergae by condition
for choice in range(2):
    for i in range(NUM_CONDITIONS):
        for j in range(NUM_NEURONS):
            data[choice][i][j] /= count[choice][i][j]

# gaussian filter + smooth
for choice in range(2):
    for i in range(NUM_CONDITIONS):
        for j in range(NUM_NEURONS):
            data[choice][i][j] = gaussian_filter1d(data[choice][i][j], 1.5)
            datap[choice][i][j] = interp1d(np.arange(15), data[choice][i][j])(np.linspace(0, 14, num=TRIAL_TIME_COURSE))

# z score
for choice in range(2):
    for i in range(NUM_NEURONS):
        if i == 0 and choice == 0:
            print(np.shape(data[choice][:,i]))
        datap[choice][:,i] = (datap[choice][:,i] - np.nanmean(datap[choice][:,i])) / np.nanstd(datap[choice][:,i])

# Replace nan runs with mean firing rate across condition if such
# data exists
for choice in range(2):
    for i in range(6): # color
        for j in range(NUM_NEURONS):
            check = np.array(datap[choice])[i*6:(i+1)*6,j]
            if len(np.where(np.isnan(check[:,0]))[0]) > 0: # nans
                for t in np.where(np.isnan(check[:,0]))[0]:
                    datap[choice][i*6 + t][j] = np.nanmean(check, axis=0)
    mot_start = 36 + 6 * np.arange(6)
    for i in range(6):
        for j in range(NUM_NEURONS):
            check = np.array(datap[choice])[mot_start + i,j]
            if len(np.where(np.isnan(check[:,0]))[0]) > 0: # nans
                for t in np.where(np.isnan(check[:,0]))[0]:
                    datap[choice][mot_start[t] + i][j] = np.nanmean(check, axis=0)

# Normalize response of each neuron across conditions
datad = np.zeros((2, NUM_CONDITIONS, NUM_NEURONS, TRIAL_TIME_COURSE)) #final data array
for choice in range(2):
    for i in range(762):
        datad[choice][:,i] = normalize(datap[choice][:,i], 0, 1)

np.save('processed_data', datad)
