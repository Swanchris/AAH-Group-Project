import numpy as np
from numpy.random import randint as rint
import time
import matplotlib.pyplot as plt
import random
import re
from itertools import compress
from copy import deepcopy

### creating an example txt instance

def create_instance(p_length_min, p_length_max, allowable_time, filename):
    p = (1 + np.random.binomial(70000, np.random.beta(50, 4, size = np.random.randint(p_length_min,p_length_max)))).astype('int64') 
    m = rint(0.4*len(p), 0.7*len(p))
    allowableTime = allowable_time
    inst = np.append(allowableTime,np.append(m,p))
    txter = open(filename, "w")
    for item in inst:
        txter.write(str(item))
        txter.write('\n')
    txter.close()

def set_instance(p_length_min, p_length_max, allowable_time):
    '''
    takes an instance, converts it to an array and then allocates it to p and m, 
    where p are the jobs and m is the number of machines
    '''
    global p, m, allowableTime
    p = (1 + np.random.binomial(70000, np.random.beta(50, 4, size = np.random.randint(p_length_min,p_length_max)))).astype('int64') 
    m = rint(0.4*len(p), 0.7*len(p))
    allowableTime = allowable_time
    inst = np.append(allowableTime,np.append(m,p))
    
    p = inst[2:]
    p.sort()
    p = p[::-1]
    m = inst[1]
    allowableTime = inst[0]
    return p, m, allowableTime

### Given an instance will be of the form (𝑝1,𝑝2,⋯,𝑝𝑛,𝑚) we need to read the txt file in this way. We can then chop it up for the agent to manipulate

def import_instance(filename):
    '''
    imports a text file instance, converts it to an array and then allocates it to p and m, 
    where p are the jobs and m is the number of machines
    '''
    inst = list(map(int, re.findall('\d+', str([line.rstrip('\n') for line in open(filename)]))))
    global p, m, allowableTime
    p =  inst[2:]
    p.sort()
    p = p[::-1]
    m = inst[1]
    allowableTime = inst[0]
    return p, m, allowableTime




class agent():
    def __init__(self): 
        self.initialTime = time.time()
        self.allocation = {} # a dict for tracking  the allocated jobs to machines
        self.workload = np.zeros(m) # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
        self.workload_so_far = []
        self.allocation_so_far = []

        
    # generates a greedy initial feasible solution
    def generateGreedySolution(self):
        for i in range(m):
            self.allocation[i] = [p[i]]
            self.workload[i] += p[i]
        for i in range(m,len(p)):
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
    
    def greedySearchIteration(self,k):
        ind = np.argsort(self.workload)
        for i in range(k):
            big_candidate = ind[-(i+1)]
            small_candidate = ind[i]
            Big = np.max(self.allocation[big_candidate])
            if not list(compress(self.allocation[small_candidate], [i < Big for i in self.allocation[small_candidate]])):
                self.switchMachine(Big, big_candidate, small_candidate)
            else: 
                Small = np.max(list(compress(self.allocation[small_candidate], [i < Big for i in self.allocation[small_candidate]])))
                self.Swap(Big, big_candidate, Small, small_candidate)
                
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
        self.workload_so_far.append(deepcopy(self.workload))
        self.allocation_so_far.append(deepcopy(self.allocation))
    
    def print_results(self):
        plt.bar(range(m),self.workload)
        plt.hlines(np.average(self.workload),0,len(self.workload), colors= 'y')
        plt.show()
#         print('best workload found:', self.workload)
#         print('best allocation found:', self.allocation)
        print('neighbours visited:', self.costTrajectory)
        print('approximation ratio:',  self.cost/np.average(self.workload))
        print('time taken:', time.time() - self.initialTime)
    
    def greedySearch(self,totalTime,k):
        self.generateGreedySolution()
        self.workload_so_far.append(deepcopy(self.workload))
        self.allocation_so_far.append(deepcopy(self.allocation))
        while time.time() - self.initialTime < totalTime - 0.31:
            self.greedySearchIteration(k)
            if self.cost > self.costTrajectory[-2]:
                self.workload = deepcopy(self.workload_so_far[-2])
                self.allocation = deepcopy(self.allocation_so_far[-2])
                self.cost = np.max(self.workload)
                self.costTrajectory.append(self.cost) 
                self.print_results()
                return
            if self.cost == self.costTrajectory[-2]:
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
            
    

### Greedy makespan allocation algo is on pg 262 of text