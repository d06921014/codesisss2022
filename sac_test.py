import numpy as np
import pandas as pd
import apuf_lib as ap
import utils
import pdb
import matplotlib.pyplot as plt

debug = False
save = False

def getnfCSAPUFResponse(stb_chs, nrof_batched_chs, nrow, dim, csapuf):
    if len(stb_chs.shape) < 3:
        stb_chs = stb_chs.reshape(-1, nrow, dim)

    gt_csaRes = np.zeros(((nrof_batched_chs, nrow)))
    gt_csapuf1 = np.zeros((nrof_batched_chs, nrow))
    gt_csapuf2 = np.zeros((nrof_batched_chs, nrow))
    for i in range(nrof_batched_chs):
        gt_csaRes[i], gt_csapuf1[i], gt_csapuf2[i] = csapuf.getPufResponse(stb_chs[i].reshape(nrow, dim), noisefree=True)
    return gt_csaRes, gt_csapuf1, gt_csapuf2
'''
def getCSAPUFResponse(stb_chs, nrof_batched_chs, nrow, dim, csapuf):
    if len(stb_chs.shape)<3:
        stb_chs.reshape(-1, row, dim)
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
    return csaRes, csapuf1, csapuf2, gt_csaRes, gt_csapuf1, gt_csapuf2 
'''
def showPlt(length, trans):
    plt.ylim(0,1)
    plt.plot(np.arange(length), trans)
    plt.show()

if __name__ == "__main__":
    
    nrow = 128
    length = 128
    nchal = 1280

    # Init csapuf
    ap1_fname = "apuf-1-{}-res-n0.5".format(length)
    ap2_fname = "apuf-2-{}-res-n0.5".format(length)
    ap1_param = np.load('PUFs/{}.npy'.format(ap1_fname))
    ap2_param = np.load('PUFs/{}.npy'.format(ap2_fname))
    ap1 = ap.APUF(length, nSigma=ap1_param[-1], delay=ap1_param[:-1])
    ap2 = ap.APUF(length, nSigma=ap2_param[-1], delay=ap2_param[:-1])

    csapuf = ap.CSAPUF(ap1, ap2, length, nrow)

    np.random.seed(17)
    chs = utils.genNChallenges(length, nchal)
    chflip = np.tile(chs, (length, 1, 1))

    # for each index in filp array, for each challenge, flip the index-th bit
    for lidx, cs in enumerate(chflip):
        for nidx, c in enumerate(cs):
            chflip[lidx, nidx, lidx] = not c[lidx]
    #pdb.set_trace()

    nrof_batches = int(nchal/nrow)

    sac_csaRes = np.zeros((length, nrof_batches, nrow))
    csapufTrans = np.zeros((length))
    csaRes, _, _ = getnfCSAPUFResponse(chs, nrof_batches, nrow, length, csapuf)
    for i in range(length):
        sac_csaRes[i], _, _  = getnfCSAPUFResponse(chflip[i], nrof_batches, nrow, length, csapuf)
        #pdb.set_trace()
        csapufTrans[i] = (sac_csaRes[i].flatten() != csaRes.flatten()).sum()
        if i % 8==0:
            print("idx - : {} ".format(i))
    

    pdb.set_trace()
    
