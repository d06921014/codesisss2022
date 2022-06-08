import apuf_lib as ap
import pandas as pd
import numpy as np
import utils
import pdb
from keras.models import load_model

debug = True

def server_expected_res(mChs, apufRes1, apufRes2):
    hres1 = utils.softResToHard(apufRes1)
    hres2 = utils.softResToHard(apufRes2)
    response = (mChs.dot(hres1)+hres2)%2
    return response

def getCSAPUFResponse(stb_chs, nrof_batched_chs, nrow, csapuf, swpufRes1, swpufRes2):
    csaRes = np.zeros((nrof_batched_chs, nrow))
    csapuf1 = np.zeros((nrof_batched_chs, nrow))
    csapuf2 = np.zeros((nrof_batched_chs, nrow))

    gt_csaRes = np.zeros(((nrof_batched_chs, nrow)))
    gt_csapuf1 = np.zeros((nrof_batched_chs, nrow))
    gt_csapuf2 = np.zeros((nrof_batched_chs, nrow))
    server_predict = np.zeros(((nrof_batched_chs, nrow)))
    for i in range(nrof_batched_chs):
        csaRes[i], csapuf1[i], csapuf2[i] = csapuf.getPufResponse(stb_chs[i].reshape(nrow, dim))
        gt_csaRes[i], gt_csapuf1[i], gt_csapuf2[i] = csapuf.getPufResponse(stb_chs[i].reshape(nrow, dim), noisefree=True)
        server_predict[i] = server_expected_res(stb_chs[i], swpufRes1[i], swpufRes2[i])
    return csaRes, csapuf1, csapuf2, gt_csaRes, gt_csapuf1, gt_csapuf2, server_predict 

def serverRecover
    

dim = 128
nrow = 128

# Load data
pretrain = 20000
batches = 1000
nchs = batches*nrow
xDir = "../dataset/dataset-128bit-challenge/dataset-3m.csv"
parity = pd.read_csv(xDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values
chsDir =  "../dataset/dataset-128bit-challenge/dataset-challenge-3m.csv"
chs = pd.read_csv(chsDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values

y1nfDir = "response/seed42/apuf-1-128-res-nf-500000.csv"
y2nfDir = "response/seed42/apuf-2-128-res-nf-500000.csv"
y1nf = pd.read_csv(y1nfDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values
y2nf = pd.read_csv(y2nfDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values

# Init csapuf
ap1_fname = "apuf-1-128-res-n0.5"
ap2_fname = "apuf-2-128-res-n0.5"
ap1_param = np.load('PUFs/{}.npy'.format(ap1_fname))
ap2_param = np.load('PUFs/{}.npy'.format(ap2_fname))
ap1 = ap.APUF(128, nSigma=ap1_param[-1], delay=ap1_param[:-1])
ap2 = ap.APUF(128, nSigma=ap2_param[-1], delay=ap2_param[:-1])

csapuf = ap.CSAPUF(ap1, ap2, dim, nrow)

# Init server
thres1 = 0
thres2 = 0.2
swap1 = load_model("model/apuf-1-128-res-n0.5.h5")
swap2 = load_model("model/apuf-2-128-res-n0.5.h5")

softres1 = swap1.predict(parity).flatten()
stb1 = utils.getReliabilityMask(softres1, thres1)

softres2 = swap2.predict(parity).flatten()
stb2 = utils.getReliabilityMask(softres2, thres2)

# Determin which challenges are reliable (for puf1 and puf2)
stbmask = stb1*stb2
stb_chs = chs[stbmask]
mRes1 = softres1[stbmask]
mRes2 = softres2[stbmask]

# Determin which challenges are unreliable (for puf1 only)
thrsunstb1 = 0.2
unstb1 = ~utils.getReliabilityMask(softres1, thrsunstb1)
ucset = chs[unstb1]

nrof_batches = int(stb_chs.shape[0]/nrow)
nrof_batched_chs = nrof_batches*nrow
rcset = stb_chs[:nrof_batched_chs, :].reshape(-1, nrow, dim)
mRes1 = mRes1[:nrof_batched_chs].reshape(-1, nrow)
mRes2 = mRes2[:nrof_batched_chs].reshape(-1, nrow)

# Replace challenges with unreliable one(s)
k = 3
# How many challenges will be replaced in RC set
nrof_uc = np.random.randint(k, size = nrof_batches)
# List to store the indices of unreliable challenges (for each batch)
ucidxlist = []
count = 0
if debug:
    print("batches of RC set: {}, number of UC set: {}".format( nrof_batches, len(ucset)))

for i in range(nrof_batches):
    if count+nrof_uc[i] >= len(ucset):
        break
    ucidx = np.random.choice(nrow, nrof_uc[i], replace = False)
    ucidxlist.append(ucidx)
    for j in range(nrof_uc[i]):
        #pdb.set_trace()
        if len(ucidx) > 0:
            for k in ucidx:
                rcset[i][k] = ucset[count]
                count = count + 1
if debug:
    print("inserted UCs: {} \n".format(count) + \
           "number of No UC batch: {}\n".format(len(nrof_uc[nrof_uc==0])) + \
           "number of 1 UC batch: {}\n".format(len(nrof_uc[nrof_uc==1])) + \
           "number of 2 UC batch: {}".format(len(nrof_uc[nrof_uc==2])))
# ==============================================================================
# So far, rcset are inserted with some unreliable challenges. Send rc to the puf

csaRes, csapuf1, csapuf2, gt_csaRes, gt_csapuf1, gt_csapuf2, server_predict = getCSAPUFResponse(rcset, nrof_batches, nrow, csapuf, mRes1, mRes2)

# To see ground truth reliability 
passthres = 0.97
puf_gt = (csaRes == gt_csaRes).reshape(-1,128)
puf_gt_rel = np.sum(puf_gt,axis=1)/dim
passrate = pufgt_rel[pufgtrel > passthres]/nrof_batches

sver_gt = (server_predict == gt_csaRes).reshape(-1,128)
sver_gt_rel = np.sum(sver_gt,axis=1)/dim

# To see performance of the server CSAPUF model 
sver_acc = np.mean(sver_gt_rel)

# Try recover original response
mdpd_puf = (server_predict == csaRes).reshape(-1,128)
server_recover(csaRes, mdpd_puf)


pdb.set_trace()
#=====================
