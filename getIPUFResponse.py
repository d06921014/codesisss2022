import numpy as np
import pandas as pd
import apuf_lib as ap
import os
import pdb
import sys
import matplotlib.pyplot as plt
from apuf_lib import APUF, XORAPUF, CSAPUF, IPUF

debug = True
saveInstances = False
   
def dualFlip(pufres, chs, nchal, length, flipIdx):
    pufres_flip = pufres.copy()
    print("flipidx: {}".format(flipIdx))

    for i, c in enumerate(chs):
        flip = np.logical_xor.reduce(c[flipIdx])
        if flip:
            pufres_flip[i] = not pufres_flip[i]

    print("% of flipped response: {}".format(np.logical_xor.reduce(chs[:, flipIdx], axis=1).mean()))
    print("pufres-mean={}".format(np.mean(pufres)))
    print("pufres_flipped-mean={}".format(np.mean(pufres_flip)))
    return pufres_flip

def initIPUF(length, x, y):
    print("init pufs...")
    # fix random seed for reproducibility
    np.random.seed(29)
    '''
    # Init CSAPUF
    ap1 = APUF(length, nSigma=nsigma)
    ap2 = APUF(length, nSigma=nsigma)
    csapuf = CSAPUF(ap1, ap2, length, length)
    '''
    # Init IPUF
    xxorpuf = XORAPUF(length, x)
    yxorpuf = XORAPUF(length+1, y)
    ipuf = IPUF(xxorpuf, yxorpuf, length)
    if saveInstances:
        #csapuf.savePUFInstance("")
        ipuf.savePUFInstance("")
    return ipuf

def getIPUFResponse(chs, nchal, resDir):
    length = 128
    x = 4
    y = 3
    resfname = '{}_{}_Ipuf-{}-hres-nf-{}.csv'.format(str(x), str(y), length, nchal)

    ipuf = initIPUF(length, x, y)

    print("get responses...")
    nfreeRes = ipuf.getPufResponse(chs)

    print("save hard responses...")
    np.savetxt(resDir+"{}/{}".format("IPUF", resfname), nfreeRes, fmt="%d", delimiter=',')
    #np.savetxt(resDir+"{}/{}".format("xorpuf", "3-xorpuf-128-hres-nf-3m"), txorres, fmt="%d", delimiter=',')

def get_df_ipuf_response(chs, nchal, resDir, length = 128):
    x = 4
    y = 3
    flipIdx = np.arange(0, length, 16)
    resfname = "{}_{}_Ipuf-{}-hres-nf-{}.csv".format(str(x), str(y), length, nchal)
    pufresDir = resDir + "{}/{}".format("IPUF", resfname)
    ipufres = pd.read_csv(pufresDir,header=None, dtype=np.int8).iloc[:nchal, :].values
    print("get responses...")
    dfipres = dualFlip(ipufres, chs, nchal, length, flipIdx)
    print("save hard responses...")
    np.savetxt(resDir+"{}/{}".format("IPUF", "df_" + resfname), dfipres, fmt="%d", delimiter=',')
    pdb.set_trace()

if __name__ == "__main__":

    length = 128
    nchal = 3000000
    chsDir = "../dataset/dataset-{}bit-challenge/dataset-challenge-3m.csv".format(length)
    resDir = "response/seed42/"

    print("load data...")
    chs = pd.read_csv(chsDir,header=None, dtype=np.int8).iloc[:nchal, :].values

    getIPUFResponse(chs, nchal, resDir)
    
    get_df_ipuf_response(chs, nchal, resDir)
    
    pdb.set_trace()
