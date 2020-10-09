### I've added my comments with triple # ~ Chris

# This is just something to start with. If there's anything you wish to change, feel free to do so.
# If there's anything that needs clarifying please indicate it as a comment with #### prefixed.

import numpy as np
import time
import matplotlib.pyplot as plt
import random
import re
from itertools import compress




from scipy.special import softmax
from scipy.stats import zscore
import math
import copy


overallInitialTime = time.time()


# note: instances should be read from text file instead of as defined below

allowableTime = 4 # time allowed in seconds




#########################
p = 1 + np.random.binomial(70000, np.random.beta(50, 4, size = 2020)) # job processing times
p = p.astype('int64')
#########################

sortedOrder = np.argsort(p)[::-1] # get index order of (decreasing) sorted array p



# p = np.append(p, 1 + np.random.randint(2001, size = 132))
# p = np.append(p, 1 + 1 + np.random.negative_binomial(n = 1500, p = 0.5, size = 201))
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






# import_inst("instance.txt")










# makespan instance from Charl's lecture notes (part 3)
# allowableTime = 1
# m = 4
# p = [3, 2, 4, 1, 3, 3, 6]

# this defines an 'agent' object which would implement a heuristic to solve the makespan problem
# note: for neatness, this should later be moved to its own file
class agent:
    def __init__(self): 
        ### Does this need to be __init__(self, machine, workload) ? Or do we only want self because different methods take different attributes ?
        self.assignedMachine = [] # list of length len(p), where self.assignedMachine[job] = machine assigned to job
        self.workload = np.array([]) # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
    # generates a random initial feasible solution
    def generateInitialSolution(self):
        self.assignedMachine = []
        self.workload = np.zeros(m)
        for job in range(len(p)):
            machine = np.random.randint(m) # randomly select machine to assign job to 
            ### should we have different variable name given that 'machine' is already an attribute ? suggest "worker"
            self.assignedMachine.append(machine) # assign 'job' to 'machine'
            self.workload[machine] += p[job] # add job processing time to workload of 'machine'
        self.cost = np.max(self.workload)
    # switch assigned machine of 'job' to 'toMachine'
    def switchMachine(self, job, toMachine):
        self.workload[self.assignedMachine[job]] += -p[job]
        self.assignedMachine[job] = toMachine
        self.workload[toMachine] += p[job]
    # Note that this is just a (random) local search implementation, not greedy local search as question 1 asks
    def localSearchIteration(self, k):
        costAlpha = self.cost
        jobs = [int(len(p)*random.random()) for i in range(k)] # randomly choose k jobs to reassign
        machines = [int(m*random.random()) for i in range(k)] # randomly choose k machines to assign these jobs to
        initialMachines = [self.assignedMachine[i] for i in jobs]
        for (job, machine) in zip(jobs, machines):
            self.switchMachine(job, machine)
        costBeta = self.workload[np.argmax(self.workload)] # find max workload across machines, faster than np.max(self.workload) 
        # if new feasible solution is worse then go back, otherwise stay
        if costBeta > costAlpha:
            for (job, machine) in zip(jobs, initialMachines):
                self.switchMachine(job, machine)
            self.costTrajectory.append(costAlpha)
        else:
            self.cost = costBeta
            self.costTrajectory.append(costBeta)
    # k defines the size of the neighbourhood and totalTime determines how much time the function is allowed to run
    def localSearch(self, k, totalTime):
        initialTime = time.time()
        # note: algorithm may run later than cut-off if iteration takes too long!
        while time.time() - initialTime < totalTime:
            self.localSearchIteration(k)
    # modified from GLS_parameter_tuning.ipynb in AAH-Group-Project to work here
    # generates a greedy initial feasible solution
    def generateGreedySolution(self):
        self.assignedMachine = [-1 for i in range(len(p))]
        self.workload = np.zeros(m)
        for i in range(m):
            job = sortedOrder[i]
            self.assignedMachine[job] = i # assign machine 'i' to 'job'
            self.workload[i] += p[job]
        for i in range(m,len(p)):
            job = sortedOrder[i]
            worker = np.argmin(self.workload)
            self.assignedMachine[job] = worker # assign machine 'worker' to 'job'
            self.workload[worker] += p[job]
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
    
        
# A = agent()
# A.generateInitialSolution()
# A.localSearch(4, allowableTime)

# plt.plot(A.costTrajectory)
# plt.xlabel("iteration")
# plt.ylabel("cost of feasible solution")
# plt.axhline(y = max(sum(p)/m, np.max(p)), color = "gold") # lower bound for global minimum

# approximationRatio = A.cost/max(sum(p)/m, np.max(p))
# print(approximationRatio)


# determines whether the solution found by A is indeed feasible, input: A is an 'agent' object
# and returns a feasible solution as a set of subsets of the jobs
def verifyFeasibleSolution(X):
    solution = [set({}) for i in range(m)] # solution is a list of subsets of the jobs
    for i in range(len(p)):
        solution[X.assignedMachine[i]].add(i) # add job i to the set corresponding to the machine it is allocated to
    setOfJobs = set(range(len(p))) # set of all jobs {0, ..., number of jobs - 1}
    for i in range(m):
	    assert(set.issubset(solution[i], setOfJobs)) # assert set i is a subset of the jobs
    assert(set.union(*solution) == setOfJobs) # assert that the union of all the sets is the set of all jobs
    for i in range(m-1):
	    for j in range(i+1, m):
		    assert(set.intersection(solution[i], solution[j]) == set()) # assert pair of sets i and j, for i != j, are disjoint
    cost = max([sum([p[i] for i in solution[machine]]) for machine in range(m)])
    assert(X.cost == cost) # assert X.cost is consistent with objective value of solution
    return (solution, cost)


# solution = verifyFeasibleSolution(A)


class geneticAgent:
    def __init__(self): 
        ### Does this need to be __init__(self, machine, workload) ? Or do we only want self because different methods take different attributes ?
        self.assignedMachine = -np.ones(len(p), dtype = int) # list of length len(p), where self.assignedMachine[job] = machine assigned to job
        self.workload = np.zeros(m) # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
    # modified from GLS_parameter_tuning.ipynb in AAH-Group-Project to work here
    # generates a greedy initial feasible solution
    def generateGreedySolution(self):
        self.assignedMachine = -np.ones(len(p), dtype = int)
        self.workload = np.zeros(m)
        for i in range(m):
            job = sortedOrder[i]
            self.assignedMachine[job] = i # assign machine 'i' to 'job'
            self.workload[i] += p[job]
        for i in range(m,len(p)):
            job = sortedOrder[i]
            worker = np.argmin(self.workload)
            self.assignedMachine[job] = worker # assign machine 'worker' to 'job'
            self.workload[worker] += p[job]
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
    # switch assigned machines of 'jobs' to 'toMachines'
    def switchMachine(self, jobs, toMachines):
        np.add.at(self.workload, self.assignedMachine[jobs], -p[jobs])
        self.assignedMachine[jobs] = toMachines
        np.add.at(self.workload, self.assignedMachine[jobs], p[jobs])
    def mutate(self, prob):
        mutations = np.random.random(len(p))
        jobs = np.where(mutations < prob)[0]
        machines = np.random.randint(0, m, size = len(jobs))
        self.switchMachine(jobs, machines)
        self.cost = self.workload[np.argmax(self.workload)]


class geneticAlgorithm:
    def __init__(self, popSize, mutationRate, initialMutationRate, allowableTime):
        self.population = None
        self.bestCost = None
        self.costTrajectory = []
        self.popSize = popSize
        assert self.popSize % 2 == 0, "Population size must be even."
        self.mutationRate = mutationRate
        self.allowableTime = allowableTime
        self.initialTime = time.time()
        self.progenitorCost = None
        self.initialMutationRate = initialMutationRate
    def crossover(self, X, Y):
        Z = geneticAgent()
        randoms = np.random.random(len(p))
        jobsFromX = np.where(randoms < 0.5)[0]
        jobsFromY = np.where(randoms >= 0.5)[0]
        Z.assignedMachine[jobsFromX] = X.assignedMachine[jobsFromX]
        Z.assignedMachine[jobsFromY] = Y.assignedMachine[jobsFromY]
        np.add.at(Z.workload, Z.assignedMachine, p)
        Z.cost = np.max(Z.workload)
        return Z
    def generateInitialPopulation(self):
        geneticAgents = []
        progenitor = geneticAgent()
        progenitor.generateGreedySolution()
        self.progenitorCost = progenitor.cost
        for i in range(self.popSize):
            A = geneticAgent()
            A.assignedMachine = copy.deepcopy(progenitor.assignedMachine)
            np.add.at(A.workload, A.assignedMachine, p)
            A.cost = np.max(A.workload)
            A.mutate(self.initialMutationRate*i/(self.popSize-1))
            geneticAgents.append(A)
        self.population = geneticAgents
        # compute best cost here?
    def computeFitness(self):
        fitnesses = []
        for i in range(len(self.population)):
            fitnesses.append(self.population[i].cost)
        fitnesses = -zscore(fitnesses)
        if math.isnan(fitnesses[0]):
            fitnesses = np.ones(len(fitnesses))
        distribution = softmax(fitnesses)
        return(fitnesses, distribution)
    def generateNewPopulation(self):
        fitnesses, distribution = self.computeFitness()
        X = np.random.choice(range(self.popSize), size = self.popSize, p = distribution).reshape((self.popSize//2, 2)) # note: REPLACE IS TRUE, ALLOWING SELF-CROSSOVER
        newGeneticAgents = []
        for i in range(self.popSize//2):
            newAgent = self.crossover(self.population[X[i, 0]], self.population[X[i, 1]])
            newAgent.mutate(self.mutationRate)
            newGeneticAgents.append(newAgent)
        self.population.extend(newGeneticAgents)
        # apply mutation to all individuals in population or just the new ones? #
        fitnesses, distribution = self.computeFitness()
        sortedFitnesses = np.argsort(fitnesses)[::-1]
        populationCopy = self.population # does this only copy pointers?
        for i in range(self.popSize):
            self.population[i] = populationCopy[sortedFitnesses[i]]
        for i in range(self.popSize//2):
            self.population.pop()
        self.bestCost = self.population[0].cost
        self.costTrajectory.append(self.bestCost)
    def optimise(self):
        self.generateInitialPopulation()
        while time.time() - self.initialTime < self.allowableTime - 0.08:
            self.generateNewPopulation()

# genAlg = geneticAlgorithm(popSize = 30, mutationRate = 8/len(p), initialMutationRate = 1/10, allowableTime = allowableTime)
# genAlg.optimise()

# plt.plot(genAlg.costTrajectory)
# plt.xlabel("iteration")
# plt.ylabel("cost of feasible solution")
# plt.axhline(y = max(sum(p)/m, np.max(p)), color = "gold") # lower bound for global minimum

# approximationRatio = genAlg.bestCost/max(sum(p)/m, np.max(p))
# print(approximationRatio)

# solution = verifyFeasibleSolution(genAlg.population[0])











class fastAnnealingAgent:
    def __init__(self): 
        ### Does this need to be __init__(self, machine, workload) ? Or do we only want self because different methods take different attributes ?
        self.assignedMachine = [] # list of length len(p), where self.assignedMachine[job] = machine assigned to job
        self.workload = np.array([]) # np.array of length m, where self.workload[machine] = sum of processing times of jobs assigned to machine
        self.cost = None # cost of current feasible solution
        self.costTrajectory = [] # list of cost of feasible solution found in each step
        self.jobsOfMachine = {machine : [] for machine in range(m)}
    # generates a random initial feasible solution
    def generateRandomInitialSolution(self):
        self.assignedMachine = []
        self.workload = np.zeros(m)
        for job in range(len(p)):
            machine = np.random.randint(m) # randomly select machine to assign job to 
            ### should we have different variable name given that 'machine' is already an attribute ? suggest "worker"
            self.assignedMachine.append(machine) # assign 'job' to 'machine'
            self.workload[machine] += p[job] # add job processing time to workload of 'machine'
            self.jobsOfMachine[machine].append(job)
        self.cost = np.max(self.workload)
    # modified from GLS_parameter_tuning.ipynb in AAH-Group-Project to work here
    # generates a greedy initial feasible solution
    def generateGreedySolution(self):
        self.assignedMachine = list(-np.ones(len(p), dtype = int))
        self.workload = np.zeros(m)
        for i in range(m):
            job = sortedOrder[i]
            self.assignedMachine[job] = i # assign machine 'i' to 'job'
            self.workload[i] += p[job]
            self.jobsOfMachine[i].append(job)
        for i in range(m,len(p)):
            job = sortedOrder[i]
            worker = np.argmin(self.workload)
            self.assignedMachine[job] = worker # assign machine 'worker' to 'job'
            self.workload[worker] += p[job]
            self.jobsOfMachine[worker].append(job)
        self.cost = np.max(self.workload)
        self.costTrajectory.append(self.cost)
    # switch assigned machine of 'job' to 'toMachine'
    def switchMachine(self, job, toMachine):
        prevMachine = self.assignedMachine[job]
        self.workload[prevMachine] += -p[job]
        self.jobsOfMachine[prevMachine].remove(job)
        self.assignedMachine[job] = toMachine
        self.jobsOfMachine[toMachine].append(job)
        self.workload[toMachine] += p[job]
    def localSearchIteration(self, k, coolness):
        costAlpha = self.cost
        maxMachine = np.argmax(self.workload)
        maxJobs = self.jobsOfMachine[maxMachine] # [i for i in range(len(p)) if self.machine[i] == maxMachine]
        processingTimes = np.array([p[i] for i in maxJobs])
        minJ = maxJobs[np.argmin(processingTimes)]
        minMachine = np.argmin(self.workload)
        self.switchMachine(minJ, minMachine)
        prob = math.exp(-coolness)
        if random.random() < prob:
            for i in range(k):
                self.switchMachine(int(len(p)*random.random()), int(m*random.random()))
        costBeta = self.workload[np.argmax(self.workload)]
        # if costBeta < self.bestCost and prob < 0.5:
        #     self.bestCost = costBeta
        #     self.bestSolutionCopy = copy.deepcopy(self.machine)
        self.cost = costBeta
        self.costTrajectory.append(costBeta)
    # k defines the size of the neighbourhood and totalTime determines how much time the function is allowed to run
    def localSearch(self, k, totalTime, targetProbability):
        initialTime = time.time()
        # note: algorithm may run later than cut-off if iteration takes too long!
        while time.time() - initialTime < totalTime - 0.15:
            coolness = 0 + (-math.log(targetProbability))*(time.time() - initialTime)/allowableTime
            self.localSearchIteration(k, coolness)


greedy = fastAnnealingAgent()
greedy.generateGreedySolution()
greedyCost = greedy.cost

B = fastAnnealingAgent()
B.generateRandomInitialSolution()
B.localSearch(3, allowableTime, 1e-05)
plt.style.use("seaborn-dark")
plt.plot(B.costTrajectory)
# plt.plot(np.minimum.accumulate(B.costTrajectory))
plt.xlabel("iteration")
plt.ylabel("cost of feasible solution")
plt.axhline(y = max(sum(p)/m, np.max(p)), color = "gold") # lower bound for global minimum
plt.axhline(y = greedyCost, color = "purple") # greedy solution cost

verifyFeasibleSolution(B)























timeElapsed = time.time() - overallInitialTime
print(timeElapsed)