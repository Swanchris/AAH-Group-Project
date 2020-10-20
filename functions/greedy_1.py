import numpy as np
from numpy.random import randint as rint
import time
import matplotlib.pyplot as plt
import random
import re
from itertools import compress
from copy import deepcopy

### Note that function names have been updated for consistent nomenclature
class agent():
    def __init__(self): 
        self.initialTime = time.time()
        # key is the index in p of job 'k' (from 0 to n-1) and element is the machine job 'k' is assigned to
        self.assignedMachine = {} 
        # Note that the index of p relates to the index of assignedMachine
        # index is machine and element is the sum of processing times of jobs assigned to machine
        self.workload = None 
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
        self.workloadTrajectory = [] # list of workloads of feasible solution found in each step
        self.assignedMachineTrajectory = [] # list of allocations found in each step
    
    def greedySearch(self,totalTime,k, m,  p, sortedOrder):
        self.generateGreedySolution(m, p , sortedOrder) # run function 
        while time.time() - self.initialTime < totalTime - 0.31: # keeps algorithm within alocated time
            try:
                self.greedySearchIteration(k, m, p, sortedOrder) # performs k-exchange
            except ValueError:
                return [m, len(p), k, self.cost, time.time() - self.initialTime, self.cost/np.average(self.workload)]
            if self.cost > self.costTrajectory[-2]: # if local minimum is reached, return algorithm. attributes update
                self.workload = deepcopy(self.workloadTrajectory[-2])
                self.assignedMachine = deepcopy(self.assignedMachineTrajectory[-2])
                self.cost = np.max(self.workload)
                self.costTrajectory.append(self.cost) 
#                 self.verifyFeasibleSolution(m, p, sortedOrder) # independently checks that result meet constraints
#                 self.print_results(m)
                return [m, len(p), k, self.cost, time.time() - self.initialTime, self.cost/np.average(self.workload)]
            if self.cost == self.costTrajectory[-2]: # if neighbour is same as last, return
#                 self.verifyFeasibleSolution(m, p, sortedOrder)
#                 self.print_results(m)
                return [m, len(p), k, self.cost, time.time() - self.initialTime, self.cost/np.average(self.workload)]
#         print('*****ALLOCATED TIME EXPIRED!*****')
#         print('BEST RESULT:')
#         self.verifyFeasibleSolution(m, p, sortedOrder)
#         self.print_results(m)
        return [m, len(p), k, self.cost, time.time() - self.initialTime, self.cost/np.average(self.workload)]
    
    def generateGreedySolution(self, m, p, sortedOrder): # generates a greedy initial feasible solution
        self.workload = np.zeros(m)
        for i in range(m): # initial allocation of m largest jobs
            self.workload[i] += p[sortedOrder[i]] # adds job i in sortedOrder (ref to p) to the workload of machine i
            self.assignedMachine[sortedOrder[i]] = i # assigns machine to relevant job
        for i in range(m,len(p)): # distributed rest of jobs sequentially to machine with smallest workload
            worker = np.argmin(self.workload) # finds machine with smallest makespan
            self.assignedMachine[sortedOrder[i]] = worker # allocates next job (i) to that machine
            self.workload[worker] += p[sortedOrder[i]] # updates workload
        self.cost = np.max(self.workload) # updates current cost
        self.costTrajectory.append(self.cost) # updates list of cost (neighbours visited)
        self.workloadTrajectory.append(deepcopy(self.workload)) # saves current workload
        self.assignedMachineTrajectory.append(deepcopy(self.assignedMachine)) # saves current allocation
    
    def greedySearchIteration(self,k, m, p, sortedOrder):
        ind = np.argsort(self.workload) # sorts workload indices
        # k-exchange - swaps the biggest element 'Big' in machine with largest workload with the biggest element 'Small'
        # in machine with smallest workload such that 'Big' > 'Small'
        if len(ind) > 0:
        	for i in range(k):
	            if i < len(ind):
	                big_candidate = ind[-(i+1)] # finds machine with biggest workload
	                small_candidate = ind[i] # finds machine with smallest workload

	                # finds biggest/smallest index of job allocated to candidates by making a list of machine workloads
	                big_machine_workload = [x for j, (x,y) in enumerate(self.assignedMachine.items()) if y == big_candidate]
	                small_machine_workload = [x for j, (x,y) in enumerate(self.assignedMachine.items()) if y == small_candidate]

	                Big = np.max([p[i] for i in big_machine_workload]) # largest job in 'big_machine_workload'
	                Big_index = np.argmax([p[i] for i in big_machine_workload]) # index in 'p' of largest job in 'big_machine_workload'

	                # list of all elements in 'small_machine_workload' that are smaller than 'Big'
	                small_list = list(compress( [p[i] for i in small_machine_workload] , [p[i] < Big for i in small_machine_workload]))
	                # indices in 'p' of the above
	                small_list_ind = list(compress( small_machine_workload , [p[i] < Big for i in small_machine_workload]))
	                if not small_list: # if no element is smaller, just moves 'Big' to 'small_candidate'
	                    self.switchMachine(Big, Big_index, big_candidate, small_candidate)
	                else: # swaps biggest element in 'small_list' with biggest element in 'big_candidate'
	                    Small = np.max(small_list)
	                    Small_index = small_list_ind[np.argmax(small_list)]
	                    self.Swap(Big, Big_index, big_candidate, Small, Small_index, small_candidate)
	        self.cost = np.max(self.workload) # updates 'cost'
	        self.costTrajectory.append(self.cost) # updates 'costTrajectory'
	        self.workloadTrajectory.append(deepcopy(self.workload)) # updates workloadTrajectory
	        self.assignedMachineTrajectory.append(deepcopy(self.assignedMachine)) # updates assignedMachineTrajectory
    
    def Swap(self, Big, Big_index, big_candidate, Small, Small_index, small_candidate):
        self.switchMachine(Big, Big_index, big_candidate, small_candidate)
        self.switchMachine(Small, Small_index, small_candidate, big_candidate)
        
    # switch assigned machine of 'job' to 'toMachine'
    def switchMachine(self,  job, job_index, fromMachine, toMachine):
        self.workload[fromMachine] += - job
        self.workload[toMachine] += job
        self.assignedMachine[job_index] = toMachine
            
    def print_results(self, m):
        plt.bar(range(m),self.workload)
        plt.hlines(np.average(self.workload),0,len(self.workload), colors= 'y')
        plt.show()
        print('neighbours visited:', self.costTrajectory)
        print('approximation ratio:',  self.cost/np.average(self.workload))
        print('time taken:', time.time() - self.initialTime)
    
        # determines whether the solution found by A is indeed feasible, input: A is an 'agent' object
    # and returns a feasible solution as a set of subsets of the jobs
    def verifyFeasibleSolution(self, m, p, sortedOrder):
        solution = [set({}) for i in range(m)] # solution is a list of subsets of the jobs
        for i in range(len(p)):
            solution[self.assignedMachine[i]].add(i) # add job i to the set corresponding to the machine it is allocated to
        setOfJobs = set(range(len(p))) # set of all jobs {0, ..., number of jobs - 1}
        for i in range(m):
            assert(set.issubset(solution[i], setOfJobs)) # assert set i is a subset of the jobs
        assert(set.union(*solution) == setOfJobs) # assert that the union of all the sets is the set of all jobs
        for i in range(m-1):
            for j in range(i+1, m):
                assert(set.intersection(solution[i], solution[j]) == set()) # assert pair of sets i and j, for i != j, are disjoint
        cost = max([sum([p[i] for i in solution[machine]]) for machine in range(m)])
        assert(self.cost == cost) # assert self.cost is consistent with objective value of solution
        return (solution, cost)