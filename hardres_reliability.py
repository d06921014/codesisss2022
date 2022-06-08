import numpy as np
import pandas as pd
import apuf_lib as ap
import utils
import pdb
import matplotlib.pyplot as plt

debug = False

no = 1
length = 128
nchal = 500000

#initialize apuf
ap_fname = "apuf-{}-{}".format(no, length)
ap_param = np.load('PUFs/{}.npy'.format(ap_fname))
apuf = APUF(length, nSigma=0.65, delay=ap_param[:-1])

#load challenges
dataDir = "../dataset/dataset-{}bit-challenge/dataset-3m.csv".format(length)
chs = pd.read_csv(dataDir,header=None, dtype=np.int8).iloc[:nchal, :].values

nfreeRes = apuf.getPufResponse(chs, nchal, noisefree=True)
hardRes = apuf.getPufResponse(chs, nchal)

reliability = np.sum(nfreeRes == hardRes)/nchal
print("overall reliability = {}".format(reliability))
