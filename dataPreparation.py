import os
import pdb
import numpy as np
import apuf_lib as ap
import utils
   
length = 128
seed = 42
nchal = 50000

dataDir = os.path.join(os.getcwd(), "dataset/dataset-{}bit-challenge".format(length))

print(dataDir)
if not os.path.exists(dataDir):
    os.makedirs(dataDir)

filename = "dataset-challenge-{}.csv".format(nchal)

chs = utils.genNChallenges(length, nchal)

print("save challenges...")
np.savetxt(os.path.join(dataDir, filename), chs, fmt="%d", delimiter=',')

#pdb.set_trace()
