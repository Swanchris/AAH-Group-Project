import numpy as np
import time
import matplotlib.pyplot as plt
import random
import re
from itertools import compress


allowableTime = 2 # time allowed in seconds
p = 1 + np.random.randint(1000, size = 612) # job processing times
p = np.append(p, 1 + np.random.randint(2001, size = 132))
p = np.append(p, 1 + 1 + np.random.negative_binomial(n = 1500, p = 0.5, size = 201))
m = 201 # number of machines
inst = np.append(allowableTime,np.append(m,p))

### creating an example txt instance

with open("instance.txt", "w") as f:
    print(inst, file = f)
    
### Given an instance will be of the form (𝑝1,𝑝2,⋯,𝑝𝑛,𝑚) we need to read the txt file in this way. We can then chop it up for the agent to manipulate

def import_inst(filename):
    '''
    imports a text file instance, converts it to an array and then allocates it to p and m, 
    where p are the jobs and m is the number of machines
    '''
    inst = list(map(int, re.findall('\d+', str([line.rstrip('\n') for line in open(filename)]))))
    global p, m, allowableTime
    p =  inst[2:]
    m = inst[1]
    allowableTime = inst[0]

import_inst("instance.txt")


# this defines an 'agent' object which would implement a heuristic to solve the makespan problem
# note: for neatness, this should later be moved to its own file
class agent:
    def __init__(self): 
        self.initialTime = time.time()
        self.allocation = {x: [] for x in range(m)} ### A dictionary where we can keep track of which job is assigned to which machine ie: allocation[0] = [k_1, ...] are the jobs assigned to the first machine
        self.workload = np.zeros(m) # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
        self.big_candidate = None
        self.small_candidate = None
        
    # generates a greedy initial feasible solution
    def generateGreedySolution(self):
        ### couple of different ideas about how to sort this array - 
        ### see "https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order"
        ### to begin with, i'll go with an inplace sorting, but whether this is efficent we'll discuss later
        
        p[::-1].sort()
        for i in range(len(p)):
            worker = np.argmin(self.workload)
            self.allocation[worker].append(p[i])
            self.workload[worker] += p[i]
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
        
        
    # switch assigned machine of 'job' to 'toMachine'
    def switchMachine(self,  job, fromMachine, toMachine):
        self.workload[fromMachine] += - job
        self.allocation[fromMachine].remove(job)
        self.workload[toMachine] += job
        self.allocation[toMachine].append(job)
    
    def Swap(self, Big, big_candidate, Small, small_candidate):
        self.switchMachine(Big, big_candidate, small_candidate)
        self.switchMachine(Small, small_candidate, big_candidate)
    
    def greedySearchIteration(self):
        self.big_candidate = np.argmax(self.workload)
        self.small_candidate = np.argmin(self.workload)
        Big = np.max(self.allocation[self.big_candidate])
        Small = np.max(list(compress(self.allocation[self.small_candidate],
                                     [i < Big for i in self.allocation[self.small_candidate]])))
        self.Swap(Big, self.big_candidate, Small, self.small_candidate)
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
        print(self.cost)
    
    def print_results(self):
        plt.bar(range(m),self.workload)
        plt.show()
        print('best workload found:', self.workload)
        print('best allocation found:', self.allocation)
        print('neighbours visited:', self.costTrajectory)
        print('approximation ratio:',  self.cost/np.average(self.workload))
    
    def greedySearch(self,totalTime):
        self.generateGreedySolution()
        while time.time() - self.initialTime < totalTime - 0.1:
            self.greedySearchIteration()
            if self.cost > self.costTrajectory[-2]:
                self.Swap(self.allocation[self.small_candidate][-1], self.small_candidate, 
                          self.allocation[self.big_candidate][-1], self.big_candidate) ### swap back to last neighbour
                self.cost = np.max(self.workload)
                self.costTrajectory.append(self.cost)
                print(self.cost)
                self.print_results()
                return
        print('*****ALLOCATED TIME EXPIRED!*****')
        print('BEST RESULT:')
        self.print_results()
    
    def verifyFeasibleSolution(self):
        # check that each job is assigned to exactly one machine
        assert(sum([len(self.allocation[i]) for i in range(m)]) == len(p))
        # check that there are at most m machines that have jobs assigned to them
        assert(len(self.allocation) <= m)

        # check that the workloads are as indicated in A.workload
        for i in range(m):
            assert(self.workload[i] == sum(self.allocation[i]))

        # check that the maximum of the workloads (i.e. the cost) is as indicated in A.cost
        assert(np.isclose(self.cost, np.max(self.workload)))
            
    
            
A = agent()
A.greedySearch(allowableTime)
A.verifyFeasibleSolution()


### Greedy makespan allocation algo is on pg 262 of text