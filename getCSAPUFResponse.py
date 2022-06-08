import os
import pdb
import sys
import numpy as np
import pandas as pd
import apuf_lib as ap
import utils
import matplotlib.pyplot as plt
from keras.models import load_model

debug = True

def getCSAPUFResponse(stb_chs, nrof_batched_chs, nrow, csapuf, swpufRes1, swpufRes2):
    csaRes = np.zeros((nrof_batched_chs, nrow))
    csapuf1 = np.zeros((nrof_batched_chs, nrow))
    csapuf2 = np.zeros((nrof_batched_chs, nrow))

    gt_csaRes = np.zeros(((nrof_batched_chs, nrow)))
    gt_csapuf1 = np.zeros((nrof_batched_chs, nrow))
    gt_csapuf2 = np.zeros((nrof_batched_chs, nrow))

    for i in range(nrof_batched_chs):
        if i %1000 == 0:
            print("progress: {}/{}".format(i, nrof_batched_chs))
        csaRes[i], csapuf1[i], csapuf2[i] = csapuf.getPufResponse(stb_chs[i].reshape(nrow, dim))
        gt_csaRes[i], gt_csapuf1[i], gt_csapuf2[i] = csapuf.getPufResponse(stb_chs[i].reshape(nrow, dim), noisefree=True)
    return csaRes, csapuf1, csapuf2, gt_csaRes, gt_csapuf1, gt_csapuf2


trained_nsig = 1.5
env_nsig = 1.5
length = 128

dim = 128
nrow = 128
chs_all = 6000000
batches = chs_all//128
nchs = batches*nrow

print("load data...")
xDir = "dataset/dataset-128bit-challenge/dataset-6m.csv"
chsDir =  "dataset/dataset-128bit-challenge/dataset-challenge-6m.csv"
parity = pd.read_csv(xDir,header=None, dtype=np.int8).iloc[:nchs, :].values
chs = pd.read_csv(chsDir,header=None, dtype=np.int8).iloc[:nchs, :].values

print("Init PUFs...")
# Init CSAPUF
ap1_fname = "apuf-1-128"
ap2_fname = "apuf-2-128"
ap1_param = np.load('PUFs/{}.npy'.format(ap1_fname))
ap2_param = np.load('PUFs/{}.npy'.format(ap2_fname))
ap1 = ap.APUF(length, nSigma=env_nsig, delay=ap1_param[:-1])
ap2 = ap.APUF(length, nSigma=env_nsig, delay=ap2_param[:-1])

csapuf = ap.CSAPUF(ap1, ap2, dim, nrow)

# Init Server
thres1 = 0.0005#0.001
thres2 = 0.05
swap1 = load_model("model/apuf-1-128-noisyres-n{}.h5".format(trained_nsig))
swap2 = load_model("model/apuf-2-128-noisyres-n{}.h5".format(trained_nsig))

softres1 = swap1.predict(parity).flatten()
stb1 = utils.getReliabilityMask(softres1, thres1)

softres2 = swap2.predict(parity).flatten()
stb2 = utils.getReliabilityMask(softres2, thres2)


# Determin which challenges are reliable 
stbmask = stb1*stb2
stb_chs = chs[stbmask]
mRes1 = softres1[stbmask]
mRes2 = softres2[stbmask]
print("% of Available challenge: {}".format(stbmask.sum()/nchs))

nrof_batches = int(stb_chs.shape[0]/nrow)
nrof_batched_chs = nrof_batches*nrow
stb_chs = stb_chs[:nrof_batched_chs, :].reshape(-1, nrow, dim)
mRes1 = mRes1[:nrof_batched_chs].reshape(-1, nrow)
mRes2 = mRes2[:nrof_batched_chs].reshape(-1, nrow)

pdb.set_trace()

print("Get stable responses...")
# Noisy/noise-free hard response
# Get CSAPUF Resopnse from "reliable challenges"
csaRes, csapuf1, csapuf2, gt_csaRes, gt_csapuf1, gt_csapuf2 = getCSAPUFResponse(stb_chs, nrof_batches, nrow, csapuf, mRes1, mRes2)

print("reliability of CSAPUF from reliable challenges(stbmask): {}".format((csaRes==gt_csaRes).sum()/nrof_batched_chs))

print("Get random responses...")
# Get CSAPUF Resopnse from "random challenges"
rChs = chs[:nrof_batched_chs, :].reshape(-1, nrow, dim)
rswapres1 = softres1[:nrof_batched_chs].reshape(-1, nrow)
rswapres2 = softres2[:nrof_batched_chs].reshape(-1, nrow)
rcsaRes, rcsapuf1, rcsapuf2, rgt_csaRes, rgt_csapuf1, rgt_csapuf2 = getCSAPUFResponse(rChs, nrof_batches, nrow, csapuf, rswapres1, rswapres2)

print("reliability of CSAPUF from random challenges: {}".format((rcsaRes==rgt_csaRes).sum()/nrof_batched_chs))

pdb.set_trace()

#print("save hard responses...")
#np.savetxt('response/seed42/csapuf/CRPseed29/csapuf-{}-res-stb-envn{}-{}.csv'.format(length, env_nsig, nrof_batched_chs), csaRes, fmt="%d", delimiter=',')
#np.savetxt('response/seed42/csapuf/CRPseed29/csapuf-{}-res-stb-nf-{}.csv'.format(length, nrof_batched_chs), gt_csaRes, fmt="%d", delimiter=',')

#np.savetxt('dataset/dataset-128bit-challenge/csa_stable_chs/stb_chs-{}.csv'.format(length, nrof_batched_chs), stb_chs.reshape(-1,128), fmt="%d", delimiter=',')
#
