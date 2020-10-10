import numpy as np
import re

def generateInstance(allowableTime, m, n, lowerJobSize, upperJobSize, alpha, beta):
    p = lowerJobSize + np.random.binomial(upperJobSize - lowerJobSize,
     np.random.beta(alpha, beta, size = n)) # job processing times
    instance = np.append([allowableTime, m], p).astype('int')
    return instance

# from GLS_parameter_tuning.ipynb in AAH project folder
def createInstance(inp, filename):
    txter = open(filename, "w")
    for item in inp:
        txter.write(str(item))
        txter.write('\n')
    txter.close()

# from GLS_parameter_tuning.ipynb in AAH project folder
def importInstance(filename):
    '''
    imports a text file instance, converts it to an array and then allocates it to p and m, 
    where p are the jobs and m is the number of machines
    '''
    inst = list(map(int, re.findall('\d+', str([line.rstrip('\n') for line in open(filename)]))))
    global p, m, allowableTime, sortedOrder
    p = inst[2:]
    sortedOrder = np.argsort(p)[::-1] # get index order of (decreasing) sorted array p
    m = inst[1]
    allowableTime = inst[0]
    return p, m, allowableTime, sortedOrder



def generate():
    allowableTime = 10
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    params = {"discreteUniform" : (1, 1), "largeJobs" : (10, 2), "extremes" : (1/2, 1/2), "centred" : (4, 4)}

    for nmagnitude in range(1, 4+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                for i in range(1, 10+1):
                    filename = str(allowableTime) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
                    alpha, beta = params[kind]
                    n = 10**nmagnitude
                    m = (mfrac/5)*n
                    inst = generateInstance(allowableTime, m, n, 1, 40000, alpha, beta)
                    createInstance(inst, filename)
