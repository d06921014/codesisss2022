import os
import numpy as np
import apuf_lib as ap
import pdb

length=64
ndim=length+1
nchal=300000

topDir = os.path.abspath(os.path.join(os.getcwd(),".."))
dataDir = os.path.join(topDir, "dataset/dataset-{}bit-challenge/selected-challenge".format(length))
paritypath = os.path.join(dataDir, 'dataset-{:.2f}m.csv'.format(nchal*(length+1)/1000000))
chalpath = os.path.join(dataDir, 'dataset-challenge-{:.2f}m.csv'.format(nchal*(length+1)/1000000))

np.random.seed(812)

f = open(paritypath, 'a')
f_ch = open(chalpath, 'a')

chs = np.

fchal = ap.genChallengesforSAC(length, nchal)
#pdb.set_trace()

print("save challenges...")
np.savetxt(f_ch, fchal, fmt="%d", delimiter=',')

f_ch.flush()
f_ch.close()

print("transform challenges...")
aphi = ap.challengeTransform2(fchal, chalSize=length, nrows=nchal*(length+1))

print("save aphi...")
np.savetxt(f, aphi, fmt="%d", delimiter=',')

f.flush()
f.close()

#pdb.set_trace()
