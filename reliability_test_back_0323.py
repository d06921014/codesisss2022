import apuf_lib as ap
import pandas as pd
import numpy as np
import utils
import pdb
from keras.models import load_model

def server_expected_res(mChs, apufRes1, apufRes2):
    hres1 = utils.softResToHard(apufRes1)
    hres2 = utils.softResToHard(apufRes2)
    response = server_expected_CSAres(mChs, hres1, hres2)
    return response

def server_expected_CSAres(mChs, hres1, hres2):
    return (mChs.dot(hres1)+hres2)%2

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

#dim = (batch, nrow, ndim), ndim=1 for all respones, ndim=128 for challengees 
def server_calibration(fchs, auth_pass_thres, nrof_batches, nrow, fcsaRes, swapufRes1, swapufRes2):
    auth_calibrat_result = np.zeros(nrof_batches)
    for i in range(nrof_batches):       
        res1 = np.zeros(nrow)
        csaRes = fcsaRes[i].copy()
        hres1 = utils.softResToHard(swapufRes1[i])
        hres2 = utils.softResToHard(swapufRes2[i])
        
        for j in range(nrow):
            res1 = hres1.copy()
            res1[j] = 1 if hres1[j] == 0 else 0
            csa_pred = server_expected_CSAres(fchs[i], res1, hres2)
            auth_result = (csa_pred==csaRes)
            if (auth_result.sum()/nrow) > auth_pass_thres:
                auth_calibrat_result[i]=1
                break
    print("server recover rate: {}".format(auth_calibrat_result.sum()/nrof_batches))
    return auth_calibrat_result
            
#dim = (batch, nrow, ndim), ndim=1 for all respones, ndim=128 for challengees 
def server_calibration_2bts(fchs, auth_pass_thres, nrof_batches, nrow, fcsaRes, swapufRes1, swapufRes2, randomRes):
    auth_calibrat_result = np.zeros(nrof_batches)
    2bit_cal_success = np.zeros(nrof_batches)
    
    for i in range(nrof_batches):       
        res1 = np.zeros(nrow)
        csaRes = fcsaRes[i].copy()
        hres1 = utils.softResToHard(swapufRes1[i])
        hres2 = utils.softResToHard(swapufRes2[i])
        
        #1bit cal
        for j in range(nrow):
            res1 = hres1.copy()
            res1[j] = 1 if hres1[j] == 0 else 0
            csa_pred = server_expected_CSAres(fchs[i], res1, hres2)
            auth_result = (csa_pred==csaRes)
            if (auth_result.sum()/nrow) > auth_pass_thres:
                auth_calibrat_result[i]=1
                break
        #1bit cal failed, try 2bit cal
        if auth_calibrat_result[i]!=1:
            print("1-bit cal failed, try 2-bit cal.")
            for j in np.arange(nrow-1):
                res1 = hres1.copy()
                res1[j] = 1 if hres1[j] == 0 else 0
                for k in np.arange(j+1, norw):
                    res1_second = res1.copy()
                    res1_second[k] = 1 if hres1[k] == 0 else 0
                    csa_pred = server_expected_CSAres(fchs[i], res1_second, hres2)
                    auth_result = (csa_pred==csaRes)
                    if (auth_result.sum()/nrow) > auth_pass_thres:
                        auth_calibrat_result[i]=1
                        break
                if auth_calibrat_result[i]==1:
                    2bit_cal_success[i]=1
                    break

    print("server recover rate: {}".format(auth_calibrat_result.sum()/nrof_batches))
    print("server recover 2bit rate: {}".format(2bit_cal_success.sum()/nrof_batches))
    return auth_calibrat_result, 2bit_cal_success
    

analysis = True

dim = 128
nrow = 128

#load data
trained_nsig = 1.5 
env_nsig = 1.5 #2.2
auth_pass_thres = 0.85 #0.95
pretrain = 20000
batches = 23437
nchs = batches*nrow
xDir = "../dataset/dataset-128bit-challenge/dataset-3m.csv"
parity = pd.read_csv(xDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values
chsDir =  "../dataset/dataset-128bit-challenge/dataset-challenge-3m.csv"
chs = pd.read_csv(chsDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values

#y1Dir = "response/seed42/softRes/apuf-1-128-noisyres-n{}-500000.csv".format(nsigma)
#y2Dir = "response/seed42/softRes/apuf-2-128-noisyres-n{}-200000.csv".format(nsigma)
#y1 = pd.read_csv(y1Dir,header=None, dtype=np.float32).iloc[pretrain:pretrain + nchs, :].values
#y2 = pd.read_csv(y2Dir,header=None, dtype=np.float32).iloc[pretrain:pretrain + nchs, :].values

y1nfDir = "response/seed42/apuf-1-128-res-nf-500000.csv"
y2nfDir = "response/seed42/apuf-2-128-res-nf-500000.csv"
y1nf = pd.read_csv(y1nfDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values
y2nf = pd.read_csv(y2nfDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values

#init csapuf
ap1_fname = "apuf-1-128"
ap2_fname = "apuf-2-128"
ap1_param = np.load('PUFs/{}.npy'.format(ap1_fname))
ap2_param = np.load('PUFs/{}.npy'.format(ap2_fname))
ap1 = ap.APUF(128, nSigma=env_nsig, delay=ap1_param[:-1])
ap2 = ap.APUF(128, nSigma=env_nsig, delay=ap2_param[:-1])

csapuf = ap.CSAPUF(ap1, ap2, dim, nrow)

#Init Server
thres1 = 0.0005#0.001
thres2 = 0.05
swap1 = load_model("model/apuf-1-128-noisyres-n{}.h5".format(trained_nsig))
swap2 = load_model("model/apuf-2-128-noisyres-n{}.h5".format(trained_nsig))

softres1 = swap1.predict(parity).flatten()
stb1 = utils.getReliabilityMask(softres1, thres1)

softres2 = swap2.predict(parity).flatten()
stb2 = utils.getReliabilityMask(softres2, thres2)


#Determin which challenges are reliable 
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

#get CSAPUF Resopnse from "reliable challenges"
csaRes, csapuf1, csapuf2, gt_csaRes, gt_csapuf1, gt_csapuf2, server_predict = getCSAPUFResponse(stb_chs, nrof_batches, nrow, csapuf, mRes1, mRes2)

#get CSAPUF Resopnse from "random challenges"
rChs = chs[:nrof_batched_chs, :].reshape(-1, nrow, dim)
rswapres1 = softres1[:nrof_batched_chs].reshape(-1, nrow)
rswapres2 = softres2[:nrof_batched_chs].reshape(-1, nrow)
rcsaRes, rcsapuf1, rcsapuf2, rgt_csaRes, rgt_csapuf1, rgt_csapuf2, rserver_predict = getCSAPUFResponse(rChs, nrof_batches, nrow, csapuf, rswapres1, rswapres2)

ap1rel = utils.diff(csapuf1, gt_csapuf1)
ap2rel = utils.diff(csapuf2, gt_csapuf2)
print("reliability of CSAPUF from reliable challenges(stbmask): {}".format((csaRes==gt_csaRes).sum()/nrof_batched_chs))
print("reliability of apuf#1 (reliable): {}".format(ap1rel))
print("reliability of apuf#2 (reliable): {}".format(ap2rel))
print("acc of server predict (reliable): {}".format((server_predict==gt_csaRes).sum()/nrof_batched_chs))
print("\n")

print("reliability of CSAPUF from random challenges: {}".format((rcsaRes==rgt_csaRes).sum()/nrof_batched_chs))
print("reliability of apuf#1 (random): {}".format(utils.diff(rcsapuf1, rgt_csapuf1)))
print("reliability of apuf#2 (random): {}".format(utils.diff(rcsapuf2, rgt_csapuf2)))
print("acc of server predict (random): {}".format((rserver_predict==rgt_csaRes).sum()/nrof_batched_chs))
print("\n")

auth_result = (server_predict.flatten()==csaRes.flatten()).reshape(nrof_batches, nrow)
authrate = np.sum(auth_result, axis=1)/nrow
#===========================================================
#Log for False Acceptance
randres = np.random.choice(2,nrof_batches*nrow).reshape(nrof_batches, nrow))
randguess = (server_predict.flatten()==randres.flatten()).reshape(nrof_batches, nrow))
rdg_acc = np.sum(randguess, axis=1)/nrow
rdgpass_rate = rdg_acc >= auth_pass_thres

#[w/o calibration] calculate authentication successful rate for reliabile challenges
succmsk = authrate >= auth_pass_thres
succauth = authrate[succmsk]
succauth_rate = len(succauth)/nrof_batches
print("successful authentication rate (reliable, without calibration): {}".format(succauth_rate))

if succauth_rate != 1:
    #[ w calibration ] calculate authentication successful rate for reliabile challenges
    fchs = stb_chs[~succmsk]
    fswapuf1 = mRes1[~succmsk]
    fswapuf2 = mRes2[~succmsk]
    fcsaRes = csaRes[~succmsk]
    calibrated, 2bt_cal = server_calibration_2bts(fchs, auth_pass_thres, len(fchs), nrow, fcsaRes, fswapuf1, fswapuf2, randres[~succmsk])
    print("successful authentication rate (reliable, with calibration): {}".format((len(succauth)+calibrated.sum())/nrof_batches))

pfr = utils.false_rejected(dim, ap1rel, ap2rel, auth_pass_thres)
print("CSAPUF P_FR (w 1-bit cal) = {}".format(pfr))
pfa

#===========================================================
#Calculate authentication successful rate for random challenges
rauth_result = (rserver_predict.flatten()==rcsaRes.flatten()).reshape(nrof_batches, nrow)
rauthrate = np.sum(rauth_result, axis=1)/nrow

rsuccauth = rauthrate[rauthrate > auth_pass_thres]
rsuccauth_rate = len(rsuccauth)/nrof_batches
print("successful authentication rate (random): {}".format(rsuccauth_rate))


pdb.set_trace()
#=====================
