import numpy as np
import time
import matplotlib.pyplot as plt
import random
import re
from itertools import compress
import math
import copy
from heuristicSolver import *
import pandas as pd

def experimentalStudy():
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    ns, ms, jobDist, costs, greedyBenchmark, optLowerBound, rho, start, neighbourhood = [], [], [], [], [], [], [], [], []
    count = 0
    for nmagnitude in range(1, 4+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                for i in range(1, 10+1):
                    filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
                    X = solver()
                    p, m, allowableTime, sortedOrder = X.importInstance("instances/" + filename)
                    ns.append(len(p))
                    ms.append(m/len(p)) # m as a fraction of the number of jobs
                    jobDist.append(kind)
                    optLowerBound.append(max(np.sum(p)/m, np.max(p)))
                    greedy = agentBase(time.time(), allowableTime)
                    greedy.generateGreedyAllocation(m, p, sortedOrder)
                    greedyBenchmark.append(greedy.cost)
                    start.append(np.random.choice(["random", "greedy"]))
                    neighbourhood.append(np.random.choice(["swap", "move"]))
                    rho.append(np.random.beta(1/2, 1/2))
                    sol = X.solve("instances/" + filename, rho[-1], start[-1], neighbourhood[-1])
                    costs.append(sol[1])
                    count += 1
                    print(count)
    df = pd.DataFrame({'n':ns, 'm':ms, 'jobDist':jobDist, 'cost':costs, 'greedy':greedyBenchmark, 'optLower':optLowerBound, 'rho':rho, 'start':start, 'neighbourhood':neighbourhood})
    return df