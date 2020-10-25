import numpy as np
import time
import random
import re
import matplotlib.pyplot as plt
import heapq
from operator import itemgetter
from heuristicSolver import *
import math

from sklearn.utils.random import sample_without_replacement #########################

import pandas as pd

from Instances import importInstance
import greedy_1

# converts rho from a expression in terms of n to a number
def rhoToNumber(rhoString, n):
    if rhoString == "1":
        rho = 1
    elif rhoString == "n^(-1/2)":
        rho = math.pow(n, -1/2)
    elif rhoString == "n^(-1)":
        rho = math.pow(n, -1)
    elif rhoString == "n^(-3/2)":
        rho = math.pow(n, -3/2)
    elif rhoString == "0":
        rho = 0
    else:
        raise ValueError("Not a valid input for rho.")
    return rho

# tune the parameters of the heuristic for a particular instance
def tuneForInstance(i, filename, jobDistribution):
    X = solver()
    X.importInstance(filename)
    n = len(X.p)
    m = X.m
    paramTuningResults = []
    for start in ["random", "greedy"]:
        for neighbourhood in ["move", "swap"]:
            for rho in ["1", "n^(-1/2)", "n^(-1)", "n^(-3/2)", "0"]:
                sol = X.solve(rhoToNumber(rho, n), start, neighbourhood)
                paramTuningResults.append([i, jobDistribution, n, m/n,  X.lowerBound, sol[1], rho, start, neighbourhood])
    return paramTuningResults

def generateTuningData(dataname):
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    paramTuningResults = []
    i = 0
    for nmagnitude in range(1, 5+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(1) + ".txt"
                paramTuningResults.extend(tuneForInstance(i, "instances/" + filename, kind))
                i += 1

    df = pd.DataFrame(paramTuningResults, columns = ["id", "jobDist", "n", "mfracn", "lowerBound", "cost", "rho", "start", "neighbourhood"])
    df.to_csv(dataname)
    return df


def generateTestDataForFixedTime(dataname):
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    results = []
    progress = 0
    for nmagnitude in range(1, 5+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                for i in range(1, 10+1):
                    filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
                    X = solver()
                    X.importInstance("instances/" + filename)
                    n = len(X.p)
                    m = X.m
                    X.allowableTime = 40 # give 40 seconds for testing on each instance
                    sol = X.solve(math.pow(n, -3/2), "greedy", "swap")
                    results.append([kind, n, m, X.lowerBound, sol[1], sol[2], sol[3]])
                    progress += 1
                    print("%.2f%% complete. Estimated time remaining: %.2f minutes." % (100*progress/(5*4*4*10), X.allowableTime*(5*4*4*10 - progress)/60)) # print progress in percentage complete
                    print(results[-1])
    df = pd.DataFrame(results, columns = ["jobDist", "n", "m", "lowerBound", "cost", "time", "timeToBestCost"])
    df.to_csv(dataname)
    return df


def generateTestDataForFixedTimeGLS(dataname):
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    results = []
    progress = 0
    for nmagnitude in range(1, 5+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                for i in range(1, 10+1):
                    filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
                    p, m, allowableTime, sortedOrder = importInstance("instances/" + filename)
                    allowableTime = 40
                    A = greedy_1.agent()
                    m, n, k, cost, time_taken, approximation_ratio = A.greedySearch(allowableTime, 2, m, p, sortedOrder)
                    results.append([kind, n, m, max(np.sum(p)/m, np.max(p)), cost, time_taken])
                    progress += 1
                    print("%.2f%% complete. Estimated time remaining: %.2f minutes." % (100*progress/(5*4*4*10), allowableTime*(5*4*4*10 - progress)/60)) # print progress in percentage complete
    df = pd.DataFrame(results, columns = ["jobDist", "n", "m", "lowerBound", "cost", "time"])
    df.to_csv(dataname)
    return df



# parameter tuning for brute force GLS

from bruteForceGreedyLocalSearch import *

# tune the parameters of the brute force GLS (abbreviated here as BFGLS)
def tuneForInstanceBFGLS(i, filename, jobDistribution):
    p, m, allowableTime, sortedOrder = importInstance(filename)
    allowableTime = 40 # give 40 seconds for tuning, as takes time to scan neighbourhood
    n = len(p)
    paramTuningResults = []
    for start in ["random", "greedy"]:
        for k in [1, 2, 3, 4]:
            A = agentBruteForceGLS(time.time(), allowableTime)
            A.generateInitialFeasibleSolution(m, p, sortedOrder, start)
            A.bruteForceGLS(m, p, k)
            paramTuningResults.append([i, jobDistribution, n, m/n,  max(np.sum(p)/m, np.max(p)), A.cost, time.time() - A.initialTime, start, k])
            print(paramTuningResults[-1])
    return paramTuningResults

def generateTuningDataBFGLS(dataname):
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    paramTuningResults = []
    i = 0
    for nmagnitude in range(1, 5+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(1) + ".txt"
                paramTuningResults.extend(tuneForInstanceBFGLS(i, "instances/" + filename, kind))
                i += 1

    df = pd.DataFrame(paramTuningResults, columns = ["id", "jobDist", "n", "mfracn", "lowerBound", "cost", "time", "start", "k"])
    df.to_csv(dataname)
    return df



def generateTestDataForFixedTimeBFGLS(dataname):
    kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
    results = []
    progress = 0
    for nmagnitude in range(1, 5+1):
        for mfrac in range(1, 4+1):
            for kind in kinds:
                for i in range(1, 10+1):
                    filename = str(10) + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
                    p, m, allowableTime, sortedOrder = importInstance("instances/" + filename)
                    allowableTime = 40
                    A = agentBruteForceGLS(time.time(), allowableTime)
                    A.generateInitialFeasibleSolution(m, p, sortedOrder, "greedy")
                    A.bruteForceGLS(m, p, 2)
                    results.append([kind, len(p), m, max(np.sum(p)/m, np.max(p)), A.cost, time.time() - A.initialTime])
                    progress += 1
                    print(results[-1])
                    print("%.2f%% complete. Estimated time remaining: %.2f minutes." % (100*progress/(5*4*4*10), allowableTime*(5*4*4*10 - progress)/60)) # print progress in percentage complete
    df = pd.DataFrame(results, columns = ["jobDist", "n", "m", "lowerBound", "cost", "time"])
    df.to_csv(dataname)
    return df






# COMPUTE MEAN OF COST/LOWERBOUND
# df = pd.read_csv("heuristicTuning.csv")
# table = df.loc[df['n'] < 10**5].groupby(["rho", "start", "neighbourhood"]).agg({"cost" : "mean"}).sort_values("cost")
# print(table.to_latex())

# df = pd.read_csv("heuristicTuning.csv")
# table = df.loc[df['n'] >= 10**5].groupby(["rho", "start", "neighbourhood"]).agg({"cost" : "mean"}).sort_values("cost")
# print(table.to_latex())


# BFGLS
# df = pd.read_csv("BFGLStuningData.csv")

# table = df.loc[df['n'] <= 100].groupby(["start", "k"]).agg({"cost" : "mean", "time" : "mean"}).sort_values("cost")
# print(table.to_latex())

# table = df.loc[df['n'] > 100].groupby(["start", "k"]).agg({"cost" : "mean", "time" : "mean"}).sort_values("cost")
# print(table.to_latex())

