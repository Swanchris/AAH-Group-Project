import numpy as np
import time
import random
import re
# import matplotlib.pyplot as plt # uncomment for plots
import heapq
from operator import itemgetter
from heuristicAgentNumbaFunctions import greedySwap, greedyMove
import math


from sklearn.utils.random import sample_without_replacement #########################


class agentBase:
    def __init__(self, initialTime, allowableTime): 
        ### Does this need to be __init__(self, machine, workload) ? Or do we only want self because different methods take different attributes ?
        self.assignedMachine = None # list of length len(p), where self.assignedMachine[job] = machine assigned to job
        self.workload = None # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        # self.costTrajectory = None # list of cost of feasible solution found in each step
        self.jobsOfMachine = None
        self.bestCost = None
        self.bestSolutionCopy = None
        self.initialTime = initialTime
        self.allowableTime = allowableTime
        self.timeToBestCost = None
        self.lowerBound = None
    # generates a random initial feasible solution
    def generateRandomAllocation(self, m, p):
        self.workload = np.zeros(m)
        self.cost = None
        # self.costTrajectory = []
        self.jobsOfMachine = [[] for machine in range(m)]
        self.assignedMachine = np.random.randint(0, m, size = len(p))
        np.add.at(self.workload, self.assignedMachine, p)
        for job in range(len(p)):
            self.jobsOfMachine[self.assignedMachine[job]].append(job)
        self.cost = np.max(self.workload)
        self.bestCost = self.cost
        self.timeToBestCost = time.time() - self.initialTime
        self.bestSolutionCopy = self.assignedMachine.copy()
        self.lowerBound = max(np.sum(p)/m, np.max(p))
        # self.costTrajectory.append(self.cost)
    # generates a greedy initial feasible solution, uses a heap data structure to keep track of minimum workload across machines

    # note: ADD CHECKS TO ENSURE THE ALGORITHM RUNS WITHIN THE ALLOWED TIME

    def generateGreedyAllocation(self, m, p, sortedOrder):
        self.assignedMachine = list(-np.ones(len(p), dtype = int))
        self.workload = [0 for i in range(m)]
        self.jobsOfMachine = [[] for machine in range(m)]
        # self.costTrajectory = []
        for i in range(m):
            job = sortedOrder[i]
            self.assignedMachine[job] = i # assign machine 'i' to 'job'
            self.workload[i] += p[job]
            self.jobsOfMachine[i].append(job)
        self.workload = list(zip(self.workload, range(len(self.workload)))) # contains tuples (workload, machine number) so heap can pop minimum workload
        heapq.heapify(self.workload) # convert self.workload into a min-heap
        for i in range(m,len(p)):
            job = sortedOrder[i]
            minWorkload, minMachine = heapq.heappop(self.workload)
            self.assignedMachine[job] = minMachine # assign machine 'minMachine' to 'job'
            heapq.heappush(self.workload, (minWorkload + p[job], minMachine))
            self.jobsOfMachine[minMachine].append(job)
        self.workload.sort(key = itemgetter(1)) # reorder so that machine workloads are in increasing order of machine number
        self.workload = np.array(list(map(itemgetter(0), self.workload))) # convert self.workload into np.array of integer workloads
        self.cost = np.max(self.workload)
        # self.costTrajectory.append(self.cost)
        self.bestCost = self.cost
        self.timeToBestCost = time.time() - self.initialTime
        self.assignedMachine = np.array(self.assignedMachine)
        self.bestSolutionCopy = self.assignedMachine.copy()
        self.lowerBound = max(np.sum(p)/m, np.max(p))
    # switch assigned machine of 'job' to 'toMachine'
    def switchMachine(self, p, job, toMachine):
        prevMachine = self.assignedMachine[job]
        self.workload[prevMachine] += -p[job]
        self.jobsOfMachine[prevMachine].remove(job)
        self.assignedMachine[job] = toMachine
        self.jobsOfMachine[toMachine].append(job)
        self.workload[toMachine] += p[job]
    # returns a solution as a list of sets, a cost value, and the total time elapsed
    # note that this does not compute that solution
    def solution(self, m, p):
        if self.cost <= self.bestCost:
            feasibleSolution = list(map(set, self.jobsOfMachine))
            cost = self.cost
        else:
            feasibleSolution = [set() for i in range(m)] # solution is a list of subsets of the jobs
            for i in range(len(p)):
                feasibleSolution[self.bestSolutionCopy[i]].add(i) # add job i to the set corresponding to the machine it is allocated to
            cost = max([sum([p[job] for job in feasibleSolution[machine]]) for machine in range(m)])
        return(feasibleSolution, cost, time.time() - self.initialTime)


class agent(agentBase):
    def __init__(self, initialTime, allowableTime):
        super().__init__(initialTime, allowableTime)
    def generateInitialFeasibleSolution(self, m, p, sortedOrder, startMethod):
        if startMethod == "greedy":
            self.generateGreedyAllocation(m, p, sortedOrder)
        elif startMethod == "random":
            self.generateRandomAllocation(m, p)
        else:
            raise ValueError("Please select either 'greedy' or 'random' for parameter startMethod.")
    # performs an iteration of a kind of local search that iteratively balances the workloads of machines
    def searchIteration(self, m, p, increasingSortedOrder, neighbourhood):
        maxMachine = np.argmax(self.workload)
        self.cost = self.workload[maxMachine]
        # self.costTrajectory.append(self.cost)
        jobsOfMaxMachine = np.array(self.jobsOfMachine[maxMachine])

        minMachine = np.argmin(self.workload)
        jobMove, minimumMoveValue = greedyMove(self.cost, jobsOfMaxMachine, p, self.workload, minMachine)

        if neighbourhood == "swap":
            searchStarts = np.searchsorted(p, [p[job] for job in jobsOfMaxMachine], side = "left", sorter = increasingSortedOrder)
            bestK, jobSwap, minimumSwapValue = greedySwap(searchStarts, jobsOfMaxMachine, self.workload, self.assignedMachine, increasingSortedOrder, p, self.cost)
        else:
            minimumSwapValue = minimumMoveValue

        if min(minimumSwapValue, minimumMoveValue) >= self.cost:
            balanced = True
        else:
            balanced = False

        if not balanced:
            # do a greedy action
            if minimumSwapValue < minimumMoveValue:
                self.switchMachine(p, jobsOfMaxMachine[bestK], self.assignedMachine[jobSwap])
                self.switchMachine(p, jobSwap, maxMachine)
            else:
                self.switchMachine(p, jobMove, minMachine)
        return balanced
    # given that the probability of any particular element of the solution vector mutating is tau, mutate accordingly
    def mutate(self, m, p, tau):
        k = np.random.binomial(len(p), tau)
        jobsToMutate = list(sample_without_replacement(len(p), k))
        toMachines = [int(m*random.random()) for i in range(k)]
        for i in range(k):
            self.switchMachine(p, jobsToMutate[i], toMachines[i])
    
    # search for better solutions, mutating when we have reached a local optimum and decreasing the mutation probability with time
    def search(self, m, p, increasingSortedOrder, rho, neighbourhood):
        while time.time() - self.initialTime < 0.985*self.allowableTime: ############# CHANGE TIME LIMIT
            balanced = self.searchIteration(m, p, increasingSortedOrder, neighbourhood)
            if balanced:
                if self.cost < self.bestCost:
                    self.bestCost = self.cost
                    self.timeToBestCost = time.time() - self.initialTime
                    self.bestSolutionCopy = self.assignedMachine.copy()
                if self.cost == math.ceil(self.lowerBound):    # if cost is definitely optimal, stop search
                    break
                if rho > 0:
                    tau = math.exp((time.time() - self.initialTime)/self.allowableTime * math.log(rho))
                else:
                    tau = 0
                self.mutate(m, p, tau)

class solver:
    def __init__(self):
        self.p = None
        self.m = None
        self.allowableTime = None
        self.sortedOrder = None
        self.lowerBound = None
    def importInstance(self, filename):
        '''
        imports a text file instance, converts it to an array and then allocates it to p and m, 
        where p are the jobs and m is the number of machines
        '''
        inst = list(map(int, re.findall('\d+', str([line.rstrip('\n') for line in open(filename)]))))
        self.p = np.array(inst[2:]).astype('int64')
        self.sortedOrder = np.argsort(self.p)[::-1] # get index order of (decreasing) sorted array p
        self.m = inst[1]
        self.allowableTime = inst[0]

    # determines whether the solution found is indeed feasible
    # input: m is number of machines, p is array of job sizes, solution is a list of sets of job indexes, cost is objective value of solution
    def verifyFeasibleSolution(self, solution, cost):
        n = len(self.p)
        setOfJobs = set(range(n)) # set of all jobs {0, ..., number of jobs - 1}
        for i in range(self.m):
            assert(set.issubset(solution[i], setOfJobs)) # assert set i is a subset of the jobs
        assert(set.union(*solution) == setOfJobs) # assert that the union of all the sets is the set of all jobs

        # check that each job appears no more than once in the sets, which implies the sets are pairwise disjoint
        count = [0 for i in range(n)] # number of occurrences of each job in the sets
        for machine in solution:
            for job in machine:
                count[job] += 1
                assert(count[job] <= 1)

        # directly check that the sets are pairwise disjoint, slower
        # for i in range(self.m-1):
        #     for j in range(i+1, self.m):
        #         assert(set.intersection(solution[i], solution[j]) == set()) # assert pair of sets i and j, for i != j, are disjoint

        assert(cost == max([sum([self.p[i] for i in solution[machine]]) for machine in range(self.m)])) # assert cost is consistent with objective value of solution


    def solve(self, rho, startMethod, neighbourhood):
        self.initialTime = time.time()

        increasingSortedOrder = self.sortedOrder[::-1]
        self.lowerBound = max(np.sum(self.p)/self.m, np.max(self.p)) # a lower bound for the optimal solution

        A = agent(self.initialTime, self.allowableTime)
        A.generateInitialFeasibleSolution(self.m, self.p, self.sortedOrder, startMethod)
        if time.time() - self.initialTime > self.allowableTime:
            return A.solution(self.m, self.p)

        A.search(self.m, self.p, increasingSortedOrder, rho, neighbourhood)
        A.cost = np.max(A.workload)

        # get time to solution found with best cost
        if A.cost < A.bestCost:
            timeToBestCost = time.time() - A.initialTime
        else:
            timeToBestCost = A.timeToBestCost

        solution, cost, timeElapsed = A.solution(self.m, self.p)

        ############################
        # self.verifyFeasibleSolution(solution, cost) ########## only used for testing
        # print(cost)
        # print(self.lowerBound)
        # approximationRatio = A.bestCost/self.lowerBound
        # print(approximationRatio)
        # plt.style.use("seaborn-dark")
        # plt.xlabel("iteration")
        # plt.ylabel("cost of feasible solution")
        # plt.plot(A.costTrajectory)
        # plt.axhline(y = self.lowerBound, color = "gold") # lower bound for global minimum
        #########################################
        return (solution, cost, timeElapsed, timeToBestCost)

def solveInstance(filename):
    X = solver()
    X.importInstance(filename)
    sol = X.solve(math.pow(len(X.p), -3/2), "greedy", "swap")
    print("Objective value: ", sol[1])
    print("Lower bound for optimal: ", X.lowerBound)
    print("Total time elapsed: ", sol[2], "seconds")
    print("Time to best solution found: ", sol[3], "seconds\n")

    print("Verifying feasible solution and cost...")
    X.verifyFeasibleSolution(sol[0], sol[1]) ##################
    print("Verified.\n") # otherwise an assertion is not met in X.verifyFeasibleSolution, which causes the program to exit

    print("Feasible solution printed to file: group 5 solution.txt. Overwritten if this file already exists.\n")

    # printFeasible = input("Print full feasible solution? Enter 'yes' or 'no'. Case-sensitive.\n")
    # while not (printFeasible in ['yes', 'no']):
    #     printFeasible = input("Print full feasible solution? Enter 'yes' or 'no'. Case-sensitive.\n")
    # if printFeasible == 'yes':
    #     print("Feasible solution (jobs indexed from 0 to n-1): ", sol[0])
    # else:
    #     pass
    return sol


# sol = solveInstance("instances/" + "10 1 4 centred 4.txt")
