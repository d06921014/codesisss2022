import numpy as np
from utils import genDelayVector
import pdb

debug = False

# return n for n-xor puf challenges [nrows, length]
#def xorpufGenchallenge(length=8, nxor=1, nrows=1):
#    c = np.random.choice(2,length*nrows).reshape(nrows, length)
#    #c[c==0]=-1
#    c = np.tile(c, nxor).reshape(nrows, nxor,length)
#    return c

# return n challenge vectors that encoded challenges [nrows, length+1]

def challengeTransform(achallenge, chalSize, nrows=1):
    aphi = np.ones([nrows,chalSize+1])
    for i in range(nrows):
        for j in range(chalSize):
            if np.logical_xor.reduce(achallenge[i,j:]):
                aphi[i,j] = -1
            else:
                aphi[i,j] = 1
    return aphi
'''
def xorChallengeTransform(achallenge, chalSize, nxor, nrows=1):
    aphi = challengeTransform(achallenge, chalSize, nrows)
    aphi = np.tile(aphi, nxor).reshape(nrows, nxor, chalSize+1)
    return aphi
'''
#==========================================================================
# ignore the above functions if read generated challenges from csv directly

# mode change mechanism design
# divide into div=4 part, get the result by xor the first bit from each part
# expected challenge:[0,1]
#============================================================================

# Noise free version
class XORAPUF:
    def __init__(self, nStage, nxor, delays = np.array([])):
        self.nstage = nStage
        self.nxor = nxor
        self.apufs = delays
        if not delays.any():
            self.apufs = genDelayVector(nStage = nStage, nrows=nxor)
        if len(self.apufs.shape) < 2:
            print("XORPUF: invalid length for APUFs: {} ".format(len(self.apufs.shape)))

    def getPufResponse(self, aphis, nchs=1, isParity=True):
        if not isParity:
            aphis = challengeTransform(aphis, aphis.shape[1], nrows = nchs)
        if len(aphis.shape) == 2:
            aphis = np.tile(aphis, self.nxor).reshape(nchs, self.nxor, self.nstage+1)
        XORAResponse = np.ones((nchs))
        for i in range(nchs):
            #pdb.set_trace()
            culDelay = np.sum(self.apufs*aphis[i],axis=1)
            #print("culDelay, before sign\n {}".format(culDelay))
            culDelay[culDelay > 0] = 1
            culDelay[culDelay <= 0] = 0
            XORAResponse[i] = np.logical_xor.reduce(culDelay).astype(int)
            if debug:
                #pdb.set_trace()
                print("cul_delay (after sign):{}, \nXORAResponse[{}]=\n{}".format(culDelay, i, XORAResponse[i]))
        return XORAResponse

    # Save delay vector and noise
    def savePUFInstance(self, path):
        np.save(path, self.apufs)
        print("save instance - path={}".format(path))
        return


class APUF:
    def __init__(self, nStage, mu=0, sigma=1, nSigma = 0.5, delay = np.array([])):
        self.nstage = nStage
        self.delayVec = delay
        if not delay.any():
            self.delayVec = genDelayVector(mu = mu, sigma = sigma, nStage = nStage)
        self.nsigma = nSigma

    # Expected input is a parity vector.
    # return 1-bit responses of one apuf, response.shape = [nchallenges, 1] 
    def getPufResponse(self, challenge, nchs=1, isParity=True, noisefree = False):
        if not isParity:
            challenge = challengeTransform(challenge, challenge.shape[1], nrows = nchs)
        nFreeResponse = self.rawDelay(challenge)
        noise = np.random.normal(0, self.nsigma, nchs)

        AResponse = nFreeResponse + noise
        if noisefree:
            AResponse = nFreeResponse.copy()
        nFreeResponse[nFreeResponse > 0] = 0
        nFreeResponse[nFreeResponse < 0] = 1
        AResponse[AResponse > 0] = 0
        AResponse[AResponse < 0] = 1

        #print("Reliability = {} %".format((np.sum(AResponse==nFreeResponse)/nchs)*100))
        return AResponse

    def rawDelay(self, parity):
        return np.add.reduce(self.delayVec*parity, 1)

    # Save delay vector and noise
    def savePUFInstance(self, path):
        tmp = np.concatenate([self.delayVec.flatten(), np.array([self.nsigma])])
        np.save(path, tmp)
        print("save instance - path={}".format(path))
        return

class CSAPUF:
    def __init__(self, apuf1, apuf2, dim, nrof_row):
        if nrof_row < dim:
            print("CSAPUF build failed: m cannot smaller than n.")
            return
        self.n = dim
        self.m = nrof_row
        self.puf_secret = apuf1
        self.puf_error = apuf2

        mask = np.zeros(self.m)
        mask[:self.n]=1
        np.random.shuffle(mask)
        self.trapdoor = mask.astype(bool)#currently no trapdoor design, set to all 1
        
    def getPufResponse(self, challenge, isParity=False, noisefree = False):
        if self.m != challenge.shape[0]:
            print("CSAPUF - require {} challenges per batch".format(self.m))
            return
        schal = challenge[self.trapdoor]
        secret = self.puf_secret.getPufResponse(schal, self.n, isParity, noisefree)
        error = self.puf_error.getPufResponse(challenge, self.m, isParity, noisefree)
        response = (challenge.dot(secret)+error)%2
        '''
        # for debug
        for i in range(self.m):
            ads = np.logical_and(challenge[i], secret)
            adsp = np.logical_xor.reduce(ads)
            adspe = np.logical_xor(adsp, error[i])
            #pdb.set_trace()
            if adspe != response[i]:
                print("csapuf getresponse {} - answer is not correct!".format(i))
        '''
        return response, secret, error

    # Save delay vector and noise
    def savePUFInstance(self, path):
        pufIns = np.concatenate([[self.puf_secret], [self.puf_error]])
        np.save(path, tmp)
        print("save instance - path={}".format(path))
        return

    def initializeIPUF(self, path, nsig):
        params = np.load(path)
        self.n = len(params[0])
        self.m = len(params[1])
        self.puf_secret = APUF(stage = self.n, delay = params[0], nSigma = nsig)
        self.puf_error = APUF(stage = self.n, delay = params[1], nSigma = nsig)
        print("Initialize CSAPUF instance - path={}".format(path))
        return

class IPUF:
    def __init__(self, x, y, dim, index=-1):
        if dim <= 0 or x.nstage != dim or y.nstage != dim+1:
            print("IPUF - Invalid length = {}".format(dim))
        self.idx = index
        if index < 0:
            self.idx = dim//2
        self.xpuf = x
        self.ypuf = y
    # Raw challenge(s) as input
    def getPufResponse(self, chals):
        nchs = chals.shape[0]
        #xchals = chals.copy()

        aphisx = challengeTransform(chals, chals.shape[1], nrows = nchs)
        xres = self.xpuf.getPufResponse(aphisx, nchs=nchs)
        xchals = np.insert(chals, self.idx, xres, axis=1)
        aphisy = challengeTransform(xchals, xchals.shape[1], nrows = nchs)
        yres = self.ypuf.getPufResponse(aphisy, nchs=nchs)
        return yres
    def savePUFInstance(self, path):
        pufIns = np.concatenate([[self.xpuf], [self.ypuf]])
        np.save(path, tmp)
        print("save instance - path={}".format(path))
        return
        
    def initializeIPUF(self, path ,index=-1):
        params = np.load(path)
        self.xpuf = params[0]
        self.ypuf = params[1]
        self.dim = xpuf.shape[1]
        if index < 0:
            self.idx = dim//2
        print("Initialize IPUF instance - path={}".format(path))
        return

        
