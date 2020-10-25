import sys
from heuristicSolver import solveInstance

filename = sys.argv[1]
sol = solveInstance(filename)

outputFile = open("group 5 solution.txt", "w")
outputFile.write("Objective value: " + str(sol[1]) + "\n")
outputFile.write("Total time elapsed: " + str(sol[2]) + " seconds\n")
outputFile.write("Time to best solution found: " + str(sol[3]) + " seconds\n\n")
outputFile.write("Feasible solution (jobs indexed from 0 to n-1):\n" + str(sol[0]))
outputFile.close()

# # what if no \n at end of file?

# for item in inp:
#     txter.write(print("Objective value: ", sol[1]))
#     txter.write('\n')
#     txter.close()