import math
import numpy as np
from heuristicSolver import *
from numba import njit
from Instances import importInstance
import pandas as pd


# the following is our own code based on the idea of 'lifted lower bounds' presented in     Haouari, Mohamed & Gharbi, Anis & Mahdi, Jemmali. (2006). Tight bounds for the identical parallel machine scheduling problem. International Transactions in Operational Research. 13. 529 - 548. 10.1111/j.1475-3995.2006.00562.x. 

# computes a lower bound for subset of jobs J_l^{k} defined as in the paper, using the bound L_{TV} in the paper 
@njit
def lJ(psort, pcumsum, l, k, m):
    lmbdakl = int(k*np.floor(l/m) + min(k, l - np.floor(l/m)*m))
    average = (pcumsum[-l:][lmbdakl-1] - pcumsum[-l] + psort[-l])/k # equivalent to sum of J_l^{k}, defined in paper, divided by k, except uses cumulative sums
    # average = np.sum(psort[-l:][:lmbdakl])/k
    # assert(average == np.sum(psort[-l:][:lmbdakl])/k)
    n = len(psort[-l:][:lmbdakl])
    jlk = psort[-l:][:lmbdakl] # subset of jobs J_k^{l}
    lbTV = max(jlk[-1], jlk[n-k-1] + jlk[n-k], average) # a lower bound for optimal, using the last l elements of the sorted p and the smallest lmbdakl of these 
    return lbTV

# computes max_{1 <= k <= m} (max_{m+1 <= l <= n} (L(J_^{l}))), the stronger 'lifted lower bound'
@njit
def liftedLB(psort, pcumsum, m):
    n = len(psort)
    maxK = -1 # as maximum is calulated over non-negative values
    for k in range(1, m+1):
        maxL = -1
        for l in range(m+1, n+1): # don't need to compute over l <= m, as per the paper
            val = lJ(psort, pcumsum, l, k, m)
            if val > maxL:
                maxL = val
        if maxL > maxK:
            maxK = maxL
    return maxK


# generate lifted lower bounds for each instance
def generateLiftedLowerBounds(dataname):
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    results = []
    progress = 0
    for nmagnitude in range(1, 5+1): ########################
        for mfrac in range(1, 4+1):
            for kind in kinds:
                for i in range(1, 10+1):
                    filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
                    p, m, allowableTime, sortedOrder = importInstance("instances/" + filename)
                    psort = np.sort(p)
                    pcumsum = np.cumsum(psort)
                    liftedLowerBound = liftedLB(psort, pcumsum, m)
                    results.append([kind, len(p), m, max(np.sum(p)/m, np.max(p)), liftedLowerBound])
                    progress += 1
                    print(progress)
    df = pd.DataFrame(results, columns = ["jobDist", "n", "m", "lowerBound", "liftedLowerBound"])
    df.to_csv(dataname)
    return df