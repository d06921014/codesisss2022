import apuf_lib as ap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import pdb
from keras.models import load_model

#mChs: (batch, nrow, dim), apufRes: (batch, nrow)
def server_expected_res(mChs, apufRes1, apufRes2):
    nrow = mChs.shape[1]
    nrof_batches = mChs.shape[0]
    hres1 = utils.softResToHard(apufRes1)
    hres2 = utils.softResToHard(apufRes2)
    response = np.zeros((nrof_batches, nrow))
    for i in range(nrof_batches):
        response[i] = server_expected_CSAres(mChs[i], hres1[i], hres2[i])
    return response

def server_expected_CSAres(mChs, hres1, hres2):
    return (mChs.dot(hres1)+hres2)%2
            
#dim = (batch, nrow, ndim), ndim=1 for all respones, ndim=128 for challengees 
def server_calibration_2bts(fchs, auth_pass_thres, nrof_batches, nrow, fcsaRes, swapufRes1, swapufRes2):
    auth_calibrat_result = np.zeros(nrof_batches)
    cal_1bt_success = np.zeros(nrof_batches)
    cal_2bt_success = np.zeros(nrof_batches)
    
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
                cal_1bt_success[i]=1
                break
        #1bit cal failed, try 2bit cal
        if auth_calibrat_result[i]!=1:
            #print("1-bit cal failed, try 2-bit cal.")
            for j in np.arange(nrow-1):
                res1 = hres1.copy()
                res1[j] = 1 if hres1[j] == 0 else 0
                for k in np.arange(j+1, nrow):
                    res1_second = res1.copy()
                    res1_second[k] = 1 if hres1[k] == 0 else 0
                    csa_pred = server_expected_CSAres(fchs[i], res1_second, hres2)
                    auth_result = (csa_pred==csaRes)
                    if (auth_result.sum()/nrow) > auth_pass_thres:
                        auth_calibrat_result[i]=1
                        break
                if auth_calibrat_result[i]==1:
                    cal_2bt_success[i]=1
                    break

    print("number of failed authentication: {}".format(nrof_batches))
    print("server recover rate: {}".format(auth_calibrat_result.sum()/nrof_batches))
    print("server recover 2bit rate: {}".format(cal_2bt_success.sum()/nrof_batches))
    return auth_calibrat_result, cal_2bt_success, cal_1bt_success
    

analysis = True

dim = 128
nrow = 128

#load data
trained_nsig = 1.5 
env_nsig = 2.2 #2.2
auth_pass_thres = 0.75 #0.95

print("===================================================")
print("Setting: train-nsig={}, env_nsig={}, auth_pass={}".format(trained_nsig, env_nsig, auth_pass_thres))
print("===================================================\n")

pretrain = 20000
batches = 23437
nchs = batches*nrow
xDir = "../dataset/dataset-128bit-challenge/dataset-3m.csv"
parity = pd.read_csv(xDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values
chsDir =  "../dataset/dataset-128bit-challenge/dataset-challenge-3m.csv"
chs = pd.read_csv(chsDir,header=None, dtype=np.int8).iloc[pretrain:pretrain + nchs, :].values

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

#Get expected CSAPUF Resopnse from reliable challenges
server_predict = server_expected_res(stb_chs, mRes1, mRes2)

#===========================================================
#Perform Random Guess to Evaluate False Acceptance
randres = np.random.choice(2,nrof_batches*nrow).reshape(nrof_batches, nrow)
randguess = (server_predict.flatten()==randres.flatten()).reshape(nrof_batches, nrow)
rdg_acc = np.sum(randguess, axis=1)/nrow
succmsk = rdg_acc >= auth_pass_thres
#===========================================================
#[w/o calibration] calculate authentication successful rate for reliabile challenges
succauth_rate = succmsk.sum()/nrof_batches
print("False Acceptance rate (reliable, without calibration): {}".format(succauth_rate))

if succauth_rate != 1:
    #[ w calibration ] calculate authentication successful rate for reliabile challenges
    fchs = stb_chs[~succmsk]
    fswapuf1 = mRes1[~succmsk]
    fswapuf2 = mRes2[~succmsk]
    frandres = randres[~succmsk]
    #Apply Random Guess Response to the Server
    calibrated, cal_2bts, cal_1bt = server_calibration_2bts(fchs, auth_pass_thres, len(fchs), nrow, frandres, fswapuf1, fswapuf2)
    print("False Acceptance rate rate (reliable, with calibration): {}".format((succmsk.sum()+calibrated.sum())/nrof_batches))

faCount = succmsk.sum() + calibrated.sum()
pfa = faCount/nrof_batches
pfa_1bt = (cal_1bt.sum()+succmsk.sum())/nrof_batches
print("CSAPUF P_FA (w 2-bit cal) = {}, fa_count: {}".format(pfa, faCount))
print("CSAPUF P_FA (w 1-bit cal) = {}, fa_count: {}".format(pfa_1bt, cal_1bt.sum()+succmsk.sum()))
print("Direct Pass = {}, Pass after cal = {}, Pass after 2-bit cal: {}".format(succmsk.sum(), calibrated.sum(), cal_2bts.sum()))


pdb.set_trace()
#=====================
