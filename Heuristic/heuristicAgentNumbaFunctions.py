import numpy as np
from numba import njit

@njit
def greedySwap(searchStarts, jobsOfMaxMachineArray, workload, assignedMachine, increasingSortedOrder, p, costAlpha):
    minimum = costAlpha
    bestJob = jobsOfMaxMachineArray[0]
    bestK = 0
    for k in range(len(jobsOfMaxMachineArray)):
        initial = searchStarts[k]
        jobK = jobsOfMaxMachineArray[k]
        for i in range(initial, 0 - 1, -1):
            job = increasingSortedOrder[i]
            if assignedMachine[jobK] == assignedMachine[job]:
                val = costAlpha
            else:
                val = max(costAlpha + p[job] - p[jobK], workload[assignedMachine[job]] + p[jobK] - p[job])
            if val < minimum:
                minimum = val
                bestJob = job
                bestK = k
    return (bestK, bestJob, minimum)

@njit
def greedyMove(costAlpha, jobsOfMaxMachineArray, p, workload, minMachine):
    minimum = costAlpha
    bestJob = jobsOfMaxMachineArray[0]
    for k in range(len(jobsOfMaxMachineArray)):
        jobK = jobsOfMaxMachineArray[k]
        val = max(costAlpha - p[jobK], workload[minMachine] + p[jobK])
        if val < minimum:
            minimum = val
            bestJob = jobK
    return (bestJob, minimum)