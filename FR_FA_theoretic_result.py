import apuf_lib as ap
import numpy as np
import math
from utils import *
from decimal import *
import pdb

n = 128
au_thres = 0.85

noise=[1.5, 1.8, 2.2, 2.8, 3, 3.5]
p1 = [0.999998477, 0.999994669, 0.999904809, 0.999109014, 0.998497508, 0.995995894]
p2 = [0.997668206, 0.994774405, 0.98929903, 0.978003308, 0.973923049, 0.962733941]
pfr=[]

#false_accept(n, au_thres)
'''
for j in np.arange(0,4,1):
    for i in np.arange(0.5, 1, 0.05):
        false_accept(n, i, ebit=j)
'''
length = len(p1)
for k in np.arange(length):
    print("noise = {} ==================================================".format(noise[k]))
    for j in np.arange(0,4,1):
        print("{}-bit EC ***********************************".format(j))
        for i in np.arange(0.85, 0.86, 0.05):
            print("authentication threshold: {}".format(i))
            pfr.append(false_rejected(n, p1[k], p2[k], i, ebit=j))

'''
while(True):
    p1, p2 = map(float, input("Enter p1, p2: ").split()) 
    false_rejected(n, p1, p2, au_thres)
'''
pdb.set_trace()

numpy.savetxt("pfr.csv", a, delimiter=",")
#=====================
