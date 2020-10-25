from heuristicSolver import agentBase
from Instances import importInstance
import time
import numpy as np
import itertools

# p, m, allowableTime, sortedOrder = importInstance("testInstance.txt")


# go to best neighbour in k-move neighbourhood in each iteration and stop when gain <= 0 or time runs out
class agentBruteForceGLS(agentBase):
    def __init__(self, initialTime, allowableTime):
        super().__init__(initialTime, allowableTime)
    def generateInitialFeasibleSolution(self, m, p, sortedOrder, startMethod):
        if startMethod == "greedy":
            self.generateGreedyAllocation(m, p, sortedOrder)
        elif startMethod == "random":
            self.generateRandomAllocation(m, p)
        else:
            raise ValueError("Please select either 'greedy' or 'random' for parameter startMethod.")
    def iterationBruteForceGLS(self, m, p, k):
        n = len(p)
        maxGain = 0 # can do nothing for gain 0
        maxJobs = []
        maxMachines = []
        maxL = 0
        for l in range(0, k+1): # consider strict 0-moves, then 1-moves, up to k-moves
            for jobs in itertools.combinations(range(n), l): # for each combination of jobs to move to some machines
                for machines in itertools.product(*[range(m) for q in range(len(jobs))]): # for each possible selection of machines to move these jobs to
                    initialMachines = [self.assignedMachine[job] for job in jobs]
                    # do the local transformation of moving jobs to machines
                    for (job, machine) in zip(jobs, machines):
                        self.switchMachine(p, job, machine)
                    costBeta = np.max(self.workload) # find cost after moving jobs to machines
                    gain = self.cost - costBeta
                    if gain > maxGain:
                        maxJobs = jobs
                        maxMachines = machines
                        maxGain = gain
                        maxL = l
                    if time.time() - self.initialTime >= 0.985*self.allowableTime or self.cost - maxGain <= self.lowerBound: # if time is about to run out or optimal is found, do the best local transformation found so far and terminate
                        # reverse the local transformation
                        for (job, machine) in zip(jobs, initialMachines):
                            self.switchMachine(p, job, machine)
                        # do the best local transformation found
                        for (job, machine) in zip(maxJobs, maxMachines):
                            self.switchMachine(p, job, machine)
                        self.cost = np.max(self.workload)
                        return (max(0, maxGain), maxJobs, maxMachines, maxL) # return max(0, maxGain) as greedyGain even if not so GLS terminates
                    # reverse the local transformation
                    for (job, machine) in zip(jobs, initialMachines):
                        self.switchMachine(p, job, machine)
        for (job, machine) in zip(maxJobs, maxMachines):
            self.switchMachine(p, job, machine)
        self.cost = np.max(self.workload)
        return (maxGain, maxJobs, maxMachines, maxL)
    def bruteForceGLS(self, m, p, k):
        greedyGain = np.inf
        while greedyGain > 0: 
            greedyGain, maxJobs, maxMachines, l = self.iterationBruteForceGLS(m, p, k)
            # print(greedyGain, maxJobs, maxMachines, l)


# k = 3
# allowableTime = 20
# startMethod = "greedy"

# A = agentBruteForceGLS(time.time(), allowableTime)
# A.generateInitialFeasibleSolution(m, p, sortedOrder, startMethod)
# print(A.cost)
# A.bruteForceGLS(m, p, k)
# print(A.cost)
# sol = A.solution(m, p)
# print(A.solution(m, p))


# # determines whether the solution found is indeed feasible
# def verifyFeasibleSolution(p, m, solution, cost):
#     n = len(p)
#     setOfJobs = set(range(n)) # set of all jobs {0, ..., number of jobs - 1}
#     for i in range(m):
#         assert(set.issubset(solution[i], setOfJobs)) # assert set i is a subset of the jobs
#     assert(set.union(*solution) == setOfJobs) # assert that the union of all the sets is the set of all jobs

#     # check that each job appears no more than once in the sets, which implies the sets are pairwise disjoint
#     count = [0 for i in range(n)] # number of occurrences of each job in the sets
#     for machine in solution:
#         for job in machine:
#             count[job] += 1
#             assert(count[job] <= 1)

#     assert(cost == max([sum([p[i] for i in solution[machine]]) for machine in range(m)])) # assert cost is consistent with objective value of solution


# verifyFeasibleSolution(p, m, sol[0], sol[1])



                



