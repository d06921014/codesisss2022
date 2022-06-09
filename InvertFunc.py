import numpy as np
from decimal import *
import math
import pdb

debug = False

class InvertFunctions:

    def __init__(self, challenges):
        self.challenges = challenges
        self.length = challenges.shape[-1]

    # idx: CNF style input
    # e.g. idx = [[c1, c2], [c3, c4]]] means c1*c2 + c3*c4
    def ANDOR(self, idx):
        if len(idx.shape) != 2 and idx.shape[1] != 2:
            print("[AND_OR] Invalid argument: index.shape={}".format(index.shape))
        tmp = []
        for pair in idx:
            a = np.logical_and.reduce(self.challenges[:, pair], axis=1)
            tmp.append(a)
        inv = np.logical_or.reduce(np.array(tmp).T, axis=1)
        return inv

    def XOR(self, idx):
        return np.logical_xor.reduce(self.challenges[:, idx], axis=1)

    def AND(self, idx):
        return np.logical_or.reduce(self.challenges[:, idx], axis=1)

    def TFF_AND(self, idx):
        nrof_idx = len(idx)
        asl = int(math.pow(2, nrof_idx))
        chs = self.challenges.copy()

        if len(chs) % asl != 0:
           print("[TFF_AND] Invalid size of challenges: challenges.shape={}".format(self.challenges.shape))
           return
        else:
           chs = chs.reshape(-1, asl, self.length)
           inv = np.zeros((len(chs), asl))
           # each batch
           for i in range(len(chs)):
               bch = chs[i]
               t = np.logical_and.reduce(bch[:, idx], axis=1)
               q = np.zeros(len(t))
               for j in range(len(t)):
                   if j < len(t)-1:
                       q[j+1] = np.logical_xor(t[j], q[j])
               inv[i] = q
               if debug:
                   print("idx={}".format(idx))
                   print("chs[idx]=\n{}".format(bch[:, idx]))
                   print("t={} \nq={}".format(t, q))
               #pdb.set_trace()
           return inv
        
#unit test

def unit_test_invf():
    import utils
    length = 8
    nchals = 32

    idx=np.arange(2)
    challenges = utils.genNChallenges(length, nchals)
    f = InvertFunctions(challenges)
    inv = f.TFF_AND(np.arange(2))
    print(inv)
    pdb.set_trace()

if __name__ == "__main__":

    unit_test_invf()

