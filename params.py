import matplotlib.pyplot as plt
import Instances.Instances as gen
import functions.greedy_1 as G
import pandas as pd
import numpy as np

# p, m, allowableTime, sortedOrder = gen.importInstance('Instances/TestInstances/10 4 4 largeJobs 9.txt')
# A = G.agent()
# A.greedySearch(allowableTime, 1, m, p , sortedOrder)


# Parameter_data = pd.DataFrame(columns = ["m", "p", "k", "workload", "time taken", "approximation ratio"])
# kinds = ["discreteUniform", "largeJobs", "extremes", "centred"]
# params = {"discreteUniform" : (1, 1), "largeJobs" : (10, 2), "extremes" : (1/2, 1/2), "centred" : (4, 4)}
# for nmagnitude in range(1, 4+1):
#     for mfrac in range(1, 4+1):
#         for kind in kinds:
#             for i in range(1, 10+1):
#                 filename = "tests/60" + " " + str(mfrac) + " " + str(nmagnitude) + " " + kind + " " + str(i) + ".txt"
#                 p, m, allowableTime, sortedOrder = gen.importInstance(filename)
#                 for k in range(1,11):
#                     A = G.agent()
#                     search = A.greedySearch(allowableTime, k, m, p , sortedOrder)
#                     Parameter_data = Parameter_data.append(dict(zip(Parameter_data.columns, search)), ignore_index=True)
# Parameter_data.to_csv('testing.csv',index=False)

Parameter_data= pd.read_csv("testing.csv")
k_data = Parameter_data.set_index(['k'])
k = {}
for i in range(1,11):
    k[i] = k_data.loc[i]
    k[i] = k[i].reset_index(drop = "true")
t = range(len(T))
data1 = A
data2 = T

fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('k-exchanges per iteration')
ax1.set_ylabel('average approximation ratio', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('average time taken', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('k_ex.jpg')
