import numpy as np
import pandas as pd
import apuf_lib as ap
import os
import pdb
import sys
import matplotlib.pyplot as plt
from apuf_lib import APUF

debug = True

def measureSoftRes(chs, nrofchs, puf, counter = 100, fname = ""):
    #calculate soft response
    print("get responses...")
    softRes = puf.getPufResponse(chs, nrofchs)

    for i in range(counter-1):
        softRes = apuf.getPufResponse(chs, nrofchs)+softRes
        if i % 1000 == 0:
            print("count : {}".format(i))
    softRes = softRes/counter

    # save response
    print("save responses...")
    np.savetxt(fname, softRes, fmt="%f", delimiter=',')
    
    return softRes
   

if __name__ == "__main__":

    no = 2
    length = 128
    nsigma = 1

    nchal = 500000
    counter = 100000
    dataDir = "../dataset/dataset-{}bit-challenge/dataset-3m.csv".format(length)
    resDir = "response/seed42/"
    softResDir = resDir+"softRes/"
    
    resfname = 'apuf-{}-{}-noisyres-n{}-{}.csv'.format(no, length, nsigma, nchal)

    print("init pufs...")
    # fix random seed for reproducibility
    # np.random.seed(42)
    # apuf = APUF(length)
    ap_fname = "apuf-{}-{}".format(no, length)
    ap_param = np.load('PUFs/{}.npy'.format(ap_fname))
    apuf = APUF(length, nSigma=nsigma, delay=ap_param[:-1])

    print("load data...")
    chs = pd.read_csv(dataDir,header=None, dtype=np.int8).iloc[:nchal, :].values

    print("get responses...")
    #measure soft response
    sresfile = softResDir+resfname
    softRes = measureSoftRes(chs, nchal, apuf, counter = counter, fname = sresfile)


    # noisy/noise-free hard response
    nfreeRes = apuf.getPufResponse(chs, nchal, noisefree=True)
    hardRes = apuf.getPufResponse(chs, nchal)

    pdb.set_trace()

    #print("save hard responses...")
    #np.savetxt('response/seed42/apuf-{}-{}-res-noisy-n{}-{}.csv'.format(no, length, nsigma, nchal), nfreeRes, fmt="%d", delimiter=',')

    
    #print("save soft responses...")
    #np.savetxt('response/seed42/softRes/apuf-{}-{}-softres-noisy-n{}-{}.csv'.format(no, length, nsigma, nchal), softRes, fmt="%d", delimiter=',')

    #plt.hist(softRes, bins=np.arange(0, 1.05, 0.05))
    #plt.show()
    pdb.set_trace()
