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

class RProp(Optimizer):
    def __init__(self, init_alpha=0.01, scale_up=1.2, scale_down=0.5, min_alpha=0.00001, max_alpha=50., **kwargs):
        super(RProp, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for p, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            grad = K.sign(grad)
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0),K.maximum(alpha * self.scale_down, self.min_alpha),alpha)    
             )

            grad = K.switch(K.less(grad * old_grad, 0),K.zeros_like(grad),grad)
            new_p = p - grad * new_alpha 

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRprop_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

length=128
epoch=1000
batch_size= 90000
nchallenge = 100000#85000

#===========================================================================
dfx=pd.read_csv('../dataset/dataset-{}bit-challenge/dataset-3m.csv'.format(length),header=None)
dfy=pd.read_csv('response/seed42/xorpuf/4_xorpuf-1-128-hres-nf-3m.csv',header=None)
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

model.add(Dense(1, input_dim = length+1, activation='sigmoid'))
'''
model.add(Dense(60,input_dim=length+1, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
#model.add(Dense(60, activation='relu'))
#model.add(Dense(60, activation='relu'))
#model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''
# print the model
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer=RProp(), metrics=['accuracy'])#,f1])

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
