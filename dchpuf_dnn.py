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
from apuf_lib import APUF, CRP

debug = True

training_set = CRP(np.zeros(1), np.zeros(1))

def trainPUFModel(in_dim, train_x, train_y, epoch, batch_size, test_x, test_y):
    model = Sequential()
    model.add(Dense(5,input_dim=in_dim, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    #model.add(Dense(50, activation='relu'))
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


# simulate DNN attacks on DCHPUF
#np.random.seed(43)
length = 128
dim = length+1
nrof_chs = 300000
train_test_ratio = 0.9
split_idx = int(nrof_chs*train_test_ratio)


dch = ap.DCH(length)
dch.lfsr.state = np.array([1,0,0,1,1,1,1,1,1,1,1])
chal = utils.genNChallenges(length, nrof_chs)
parity = ap.challengeTransform(chal, length, nrof_chs)
init_state = dch.lfsr.state
print("challenge: \n{}".format(chal))
print("dch_lfsr_state={}".format(dch.lfsr.state))

res = dch.getOneStatePufResponse(chal, noisefree=True)

#pdb.set_trace()

training_set = CRP(parity[:split_idx], res[:split_idx])
testing_set = CRP(parity[split_idx:], res[split_idx:])

epoch = 6000
batch_size = 5000
model = trainPUFModel(length+1, training_set.challenge, training_set.response,\
                      epoch, batch_size, testing_set.challenge, testing_set.response)

# ==================================
# Heldout test
# ==================================
'''
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
'''
pdb.set_trace()


