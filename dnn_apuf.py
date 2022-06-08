import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import initializers
import keras.backend as K

from sklearn import metrics
import pandas as pd
import time
import pdb
import utils


def getReliableXY(x, y, threshold):
    swapuf = load_model("model/apuf-2-128-res-n0.5.h5")
    res = swapuf.predict(x).flatten()
    mask = utils.getReliabilityMask(res, threshold)
    return x[mask], y[mask], mask

no = 1
nsigma = 1

length=128
epoch=800
batch_size= 500
nchallenge = 20000#85000

#===========================================================================
#y_file = "apuf-2-128-res-n0.5-500000"
#y_file = "apuf-1-128-res-n0.5-100000"
y_file = "apuf-{}-128-noisyres-n{}-500000".format(no, nsigma)

#===========================================================================
outfile = "apuf-{}-128-noisyres-n{}".format(no, nsigma)

dfx=pd.read_csv('../dataset/dataset-{}bit-challenge/dataset-3m.csv'.format(length),header=None)
#===========================================================================
#dfy=pd.read_csv('response/seed42/apuf128-res-nf-100000.csv'.format(length),header=None)
#dfy=pd.read_csv('response/seed42/softRes/{}.csv'.format(y_file),header=None)
dfy=pd.read_csv('response/seed42/softRes/{}.csv'.format(y_file),header=None)
#===========================================================================
# fix random seed for reproducibility
np.random.seed(31)
#dataset = numpy.loadtxt("sixxor.csv")
# split into input (X) and output (Y) variables
nchallenge = int(nchallenge*1)
X = dfx.iloc[:nchallenge,:length+1].values
Y = dfy.iloc[:nchallenge,:].values

#X,Y, mask = getReliableXY(X, Y, 0.001)

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.1, random_state = 31)
# create model
model = Sequential()

model.add(Dense(5,input_dim=length+1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# print the model
model.summary()

# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])#,f1])

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

pdb.set_trace()
print("save model...")
model.save('model/{}.h5'.format(outfile))
