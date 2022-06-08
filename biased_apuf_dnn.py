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
import apuf_lib as ap

train_nsig=1.5
biased =False

def getReliableXY(x, y, threshold, invert=False):
    swapuf = load_model("model/apuf-1-128-noisyres-n{}.h5".format(train_nsig))
    res = swapuf.predict(x).flatten()
    mask = utils.getReliabilityMask(res, threshold)
    if invert:
        mask = ~mask
    return x[mask], y[mask], mask, res

no = 1
env_nsig = 4

length=128
epoch=300
batch_size= 500
pretrain_chs = 20000
nchallenge = 20000#85000

# fix random seed for reproducibility
np.random.seed(31)

#===========================================================================
#y_file = "apuf-{}-128-noisyres-n{}-500000".format(no, nsigma)
#===========================================================================
outfile = "apuf-{}-biased-thres0.0005-128-noisyres-n{}".format(no, env_nsig)

dfx=pd.read_csv('../dataset/dataset-{}bit-challenge/dataset-3m.csv'.format(length),header=None).values[pretrain_chs:]

# init Y (PUF)
ap_fname = "apuf-1-128"
ap_param = np.load('PUFs/{}.npy'.format(ap_fname))
ap = ap.APUF(128, nSigma=env_nsig, delay=ap_param[:-1])
apres = ap.getPufResponse(dfx, nchs=len(dfx))
apres_gt = ap.getPufResponse(dfx, nchs=len(dfx), noisefree=True)
print("bit-error-rate of APUF under env. noise sig={}: {}".format(env_nsig, (apres==apres_gt).sum()/len(apres)))

X = dfx[:nchallenge, :length+1]
Y = apres[:nchallenge]

if biased:
    X,Y, mask, swapres = getReliableXY(dfx, apres, 0.1, invert=False)
    h_swres = utils.softResToHard(swapres)

    # split into input (X) and output (Y) variables
    X = X[:nchallenge,:length+1]
    Y = Y[:nchallenge]

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.1, random_state = 31)
# create model
model = Sequential()

model.add(Dense(5,input_dim=length+1, activation='relu'))
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

pdb.set_trace()
print("save model...")
model.save('biased_model/{}.h5'.format(outfile))

