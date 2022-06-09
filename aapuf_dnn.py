import os
#from utility import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM, GRU
from keras import initializers
from keras.callbacks import EarlyStopping
import keras.backend as K

import numpy as np
import apuf_lib as ap
import challengeUtils as cts
import utils
import pdb
import random
from InvertFunc import InvertFunctions
from apuf_lib import APUF, CRP

debug = True

training_set = CRP(np.zeros(1), np.zeros(1))

def getPoisonedRes(r, invert_idx):
    p_res = r.copy()
    for i in invert_idx:
       p_res[i] = not p_res[i]
    return p_res

# 15,50 2 layers for CCA
def trainPUFModel(in_dim, train_x, train_y, epoch, batch_size, test_x, test_y, ftype='XOR'):
    if ftype=='TFF_AND':
        train_x = train_x.reshape(-1, in_dim)
        test_x = test_x.reshape(-1, in_dim)
        train_y = train_y.flatten()
        test_y = test_y.flatten()
    model = Sequential()
    model.add(Dense(15,input_dim=in_dim, activation='relu'))
    model.add(Dense(50,input_dim=in_dim, activation='relu'))
    #model.add(Dense(50,input_dim=in_dim, activation='relu'))
    #model.add(Dense(200,activation='relu'))
    #model.add(Dense(100,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])
    #callback = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=25)

    results = model.fit(
     train_x, train_y,
     epochs= epoch,
     #verbose=1,
     batch_size = batch_size,
     #callbacks = [callback],
     validation_data = (test_x, test_y)
    )

    scores = model.evaluate(test_x, test_y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model


# DNN attacks on APUF
# np.random.seed(43)
length = 128
dim = length+1
nrof_chs = 2600000
train_test_ratio = 0.9

nrof_tr_bits = 6
split_idx = int(nrof_chs*train_test_ratio)
ivt_idx = -1

apuf = APUF(length)
# chs = prepareChallenges(length, nrof_chs)
# cts.prepareDiffChs: ratio=0 means all challenges are generated randomly
chs = cts.prepareDiffChs(length, nrof_chs, ratio=0.85)#0.85)
parity = ap.challengeTransform(chs, length, nrof_chs)
res = apuf.getPufResponse(parity, nrof_chs, noisefree=True)

f = InvertFunctions(chs)
training_set = CRP(parity[:split_idx], res[:split_idx])
testing_set = CRP(parity[split_idx:], res[split_idx:])

#noise_rate = 0.4 
c_idx = np.array(random.sample(range(length), nrof_tr_bits))

mask = f.XOR(c_idx)
ivt_idx = np.where(mask==1)[0]
print("c_idx = {}\nInverted_idx = {}, shape={}".format(c_idx, ivt_idx, ivt_idx.shape))
noisy_res = getPoisonedRes(res, ivt_idx)

training_set.response = noisy_res[:split_idx]
testing_set.response = noisy_res[split_idx:]

epoch = 500
batch_size = 2000
model = trainPUFModel(length+1, training_set.challenge, training_set.response,\
                      epoch, batch_size, testing_set.challenge, testing_set.response)

# ==================================
# Heldout test
# ==================================
nrof_heldout = 10000
h_chs = utils.genNChallenges(length, nrof_heldout)
h_pty = ap.challengeTransform(h_chs, length, nrof_heldout)
h_res = apuf.getPufResponse(h_pty, nrof_heldout, noisefree=True)
hf = InvertFunctions(h_chs)

h_mask = hf.XOR(c_idx)
h_ivt_idx = np.where(h_mask==1)[0]
h_noisy_res = getPoisonedRes(h_res, h_ivt_idx)

heldout_err = model.evaluate(h_pty, h_noisy_res)
heldout_ivt_err = model.evaluate(h_pty[h_ivt_idx], h_noisy_res[h_ivt_idx])
print("heldout_test err = {}".format(heldout_err))
print("heldout_test inverted_err = {}".format(heldout_ivt_err))

pdb.set_trace()


