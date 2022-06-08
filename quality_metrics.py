import numpy as np
import pandas as pd
import pdb
import utils
from apuf_lib import APUF, CSAPUF
import matplotlib.pyplot as plt

debug = False
save = False

    
def chHD(chs, nchal, length):
    hdDist = np.zeros(length)
    for i in range(nchal):
        for j in range(i+1, nchal):
            hd = np.sum(chs[i]!=chs[j])
            hdDist[hd] = hdDist[hd] + 1
    return hdDist

def getPUFResponse(csapuf, chs, batch_size):
    nchs = chs.shape[0]
    dims = chs.shape[1]
    nbtches = nchs//batch_size
    batched_chs = chs.reshape(nbtches, batch_size, dims)
    pufRes = np.zeros((nbtches, batch_size))
    secRes = np.zeros((nbtches, batch_size))
    errRes = np.zeros((nbtches, batch_size))

    for i in range(nbtches):
        pufRes[i], secRes[i], errRes[i] = csapuf.getPufResponse(batched_chs[i], isParity=False, noisefree = True)
    return pufRes.flatten()

# shape of responses = n instances, k response bits
def uniqueness(responses, nInstances, length):
    hdDist = np.zeros(length+1)
    sum_hd = 0
    for i in range(nInstances):
        for j in range(i+1, nInstances):
            hd = np.sum(responses[i]!=responses[j])
            hdDist[hd] = hdDist[hd] + 1
            sum_hd = sum_hd + hd
    nrof_cmp = (nInstances*(nInstances-1))/2
    print("Uniqueness test for {} instances: {} % ".format(nInstances, sum_hd/nrof_cmp/length))
    return hdDist
# m = n = length
length = 128
rounds = 100
nchal = length*rounds
nhd = length+1
nUniform = 100
nUnique = 100

exp = "cppuf"

pufreslist = np.zeros((rounds, nUniform, length))
ufResSum = np.zeros((rounds, nhd))

np.random.seed(53)
chs = utils.genNChallenges(length, nchal).reshape(rounds, length, length)
#print("calculate hamming distance between challenges...")
#hdch = chHD(chs, nchal, length)
#hdchPercent = (hdch/np.sum(hdch))*100
#hdDist = np.concatenate([hdch, hdchPercent]).reshape(-1,hdch.shape[0])
    
#np.savetxt('1-128_challenge_HD_dist.csv', hdDist, fmt="%d", delimiter=',')

print("init puf list...")
puflist = np.array([CSAPUF(APUF(length), APUF(length), length, length) for _ in range(nUniform)])

print("Uniformity test - {} instances x {} bit response ...".format(nUniform, nchal))
for j in range(rounds):
    for i in range(nUniform):
        if i % 50 == 0:
            print("{}rounds, {} th".format(j, i))
        pufreslist[j, i, :] = getPUFResponse(puflist[i], chs[j], length)
        idx = int(pufreslist[j, i].sum())
        ufResSum[j, idx] = ufResSum[j, idx] + 1

percent_ufResSum = (ufResSum.sum(axis=0))/(nUniform*rounds)

#np.savetxt('{}_pufreslist-{}pufs-{}bits.csv'.format(exp, nUniform, length), pufreslist, fmt="%d", delimiter=',')

#np.savetxt('{}_uniformity-{}pufs-{}bits.csv'.format(exp, nUniform, length), ufResSum, fmt="%d", delimiter=',')
print("Uniformity test for {} instances: {} % ".format(nUniform, percent_ufResSum))
    
plt.bar(np.arange(nhd), percent_ufResSum)
plt.xlabel('Hamming Weight')
plt.ylabel('% of responses')
plt.savefig('{}_uniformity-1kpufs-{}bit.png'.format(exp, length))
plt.show()    

uniqueDist = np.zeros((rounds, length+1))
nrof_cmp = (nUnique*(nUnique-1))/2
for i in range(rounds):
    print("Uniqueness test - {} rounds, {} instances x {} bit response ...".format(i, nUnique, length))
    uniqueDist[i, :] = uniqueness(pufreslist[i, :nUnique, :], nUnique, length)
#np.savetxt('{}_uniqueness-{}pufs-{}bits.csv'.format(exp, nUnique, length), uniqueDist, fmt="%d", delimiter=',')

perent_uniqueDist = uniqueDist.sum(axis=0)/(nrof_cmp*rounds)

#pdb.set_trace()
plt.bar(np.arange(nhd), perent_uniqueDist)
plt.xlabel('Hamming Distance')
plt.ylabel('% of responses')
plt.savefig('{}_uniqueness-1kpufs-{}bit.png'.format(exp, length))    
plt.show()

pdb.set_trace()
    
