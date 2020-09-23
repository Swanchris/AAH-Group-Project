### I've added my comments with triple # ~ Chris

# This is just something to start with. If there's anything you wish to change, feel free to do so.
# If there's anything that needs clarifying please indicate it as a comment with #### prefixed.

import numpy as np
import time
import matplotlib.pyplot as plt
import random


# note: instances should be read from text file instead of as defined below

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
    m =  int(inst[2:])
    p = inst[1]
    allowableTime = inst[0]

# this defines an 'agent' object which would implement a heuristic to solve the makespan problem
# note: for neatness, this should later be moved to its own file
class agent:
    def __init__(self): 
        self.initialTime = time.time()
        self.allocation = {x: [] for x in range(m)} ### A dictionary where we can keep track of which job is assigned to which machine ie: allocation[0] = [k_1, ...] are the jobs assigned to the first machine
        self.workload = np.array([]) # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
        
    # generates a greedy initial feasible solution
    def generateGreedySolution(self):
        ### couple of different ideas about how to sort this array - 
        ### see "https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order"
        ### to begin with, i'll go with an inplace sorting, but whether this is efficent we'll discuss later
        self.workload = np.zeros(m)
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
        big_candidate = np.argmax(self.workload)
        small_candidate = np.argmin(self.workload)
        Big = np.max(self.allocation[big_candidate])
        Small = np.min(self.allocation[small_candidate])
        self.Swap(Big, big_candidate, Small, small_candidate)
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
        print(self.cost)
    
    def greedySearch(self,totalTime):
        self.generateGreedySolution()
        while time.time() - self.initialTime < totalTime:
            self.greedySearchIteration()
            if self.costTrajectory[-1] <= self.costTrajectory[-2]:
                return print(self.costTrajectory[-2])
            
    
            
    

    ### the switch machine function will work fine for the greedy search
### Greedy makespan algo is on pg 262 of text