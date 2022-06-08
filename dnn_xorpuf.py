import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import initializers
from keras.optimizers import Optimizer
import keras.backend as K

from sklearn import metrics
import pandas as pd
import time
import pdb
import utils

length=128
epoch=10000
batch_size= 10000
nchallenge = 3000000#85000

#===========================================================================
dfx=pd.read_csv('../dataset/dataset-{}bit-challenge/dataset-3m.csv'.format(length),header=None)
dfy=pd.read_csv('response/seed42/xorpuf/6_xorpuf-128-hres-nf-3m.csv',header=None)
#===========================================================================
# fix random seed for reproducibility
np.random.seed(31)
# split into input (X) and output (Y) variables
nchallenge = int(nchallenge*1)
X = dfx.iloc[:nchallenge,:length+1].values
Y = dfy.iloc[:nchallenge,:].values

#pdb.set_trace()

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.1, random_state = 31)
# create model
model = Sequential()

model.add(Dense(60,input_dim=length+1, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
#model.add(Dense(60, activation='relu'))
#model.add(Dense(60, activation='relu'))
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
print("\n{}: {}".format(model.metrics_names[1], scores[1]))

#pdb.set_trace()
#print("save model...")
#model.save('model/{}.h5'.format(outfile))
