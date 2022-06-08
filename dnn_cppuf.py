import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,LSTM, Input
from keras import initializers
import keras.backend as K

from sklearn import metrics
import pandas as pd
import time
import pdb
import utils

isLSTM=False

m=128
length=128
epoch=500
batch_size= 5120

chs = 100000
nrof_aut = int(chs/m)
nchs = nrof_aut*m

#===========================================================================
y_file = "csapuf-128-res-stb-nf-2642048"
#===========================================================================

dfx=pd.read_csv('dataset/dataset-128bit-challenge/csa_stable_chs/stb_parity-2642048.csv',header=None)
dfy=pd.read_csv('response/seed42/csapuf/CRPseed29/{}.csv'.format(y_file),header=None)
#===========================================================================
# fix random seed for reproducibility
np.random.seed(31)
# split into input (X) and output (Y) variables
X = dfx.values[:nchs].reshape(-1, m, length+1)
Y = dfy.values[:nrof_aut]

nbatches = Y.shape[0]
train_test_idx = np.arange(nbatches)
# Shuffle
np.random.shuffle(train_test_idx)
X = X[train_test_idx]
Y = Y[train_test_idx]

train_test_split = int(nbatches*0.9)
train_features = X[:train_test_split].reshape(-1,length+1)
test_features = X[train_test_split:].reshape(-1,length+1)
train_labels = Y[:train_test_split].flatten()
test_labels = Y[train_test_split:].flatten()

if isLSTM:
    train_features = train_features.reshape(-1, m, length+1)
    test_features = test_features.reshape(-1, m, length+1)
    train_labels = train_labels.reshape(-1, m, 1)
    test_labels = test_labels.reshape(-1, m, 1)

#pdb.set_trace()

# create model
if isLSTM:
    X_input = Input(train_features.shape[1:])
    X = LSTM(50, return_sequences=True, input_shape = (m,length+1), name = 'LSTM_1')(X_input)
    X = LSTM(50, return_sequences=True)(X)
    #X = LSTM(50, return_sequences=True)(X)
    #X = LSTM(50, return_sequences=True)(X)
    #X = Dense(50, activation='relu')(X)
    #X = Dense(50, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)
    ##The model object
    model = Model(inputs = X_input, outputs = X, name='LSTMModel')
else:
    model = Sequential()
    model.add(Dense(15,input_dim=length+1, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #model.add(Dense(200,input_dim=length+1, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

# print the model
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])

# fit the model
results = model.fit(
 train_features, train_labels,
 epochs= epoch,
 #verbose=1,
 batch_size = batch_size,
 validation_data = (test_features, test_labels)
)
# evaluvate the model
scores = model.evaluate(test_features, test_labels)
print("\n{}: {}, std={}".format(model.metrics_names[1], scores[1], np.std(scores[1])))

#pdb.set_trace()
#print("save model...")
#outfile = "csapuf-128-noisyres-600k_train"
#model.save('model/{}.h5'.format(outfile))
