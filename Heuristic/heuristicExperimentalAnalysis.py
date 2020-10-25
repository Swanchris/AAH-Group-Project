import pandas as pd
from plotnine import *
import math
import numpy as np


df1 = pd.read_csv("testDataNew.csv", index_col = 0)
df2 = pd.read_csv("testDataGreedy.csv", index_col = 0)

df = pd.concat([df1, df2.rename(columns = {"time" : "timeGLS", "cost" : "costGLS"})[["costGLS", "timeGLS"]]], axis = 1)


# ggplot(df, aes('costGLS/lowerBound - 1', 'cost/lowerBound - 1')) + geom_point(aes(shape = 'jobDist', colour = 'factor(m/n)'), alpha = 0.2) + geom_abline(slope = 1, intercept = 0) + facet_grid('~ n')
# ggplot(df, aes('costGLS/lowerBound - 1', 'cost/lowerBound - 1')) + geom_point(aes(shape = 'jobDist', colour = 'factor(n)'), alpha = 0.2) + geom_abline(slope = 1, intercept = 0) + facet_grid('~ m/n')


# performance relative to greedy local search
plot1 = ggplot(df, aes('jobDist', 'cost/costGLS')) + geom_boxplot(aes(colour = 'jobDist'), alpha = 0.2) + facet_grid('m/n ~ n', labeller = "label_both") + theme(axis_text_x = element_blank()) + ggtitle("boxplots: costOfHeuristic/costOfGreedyLocalSearch")

# absolute performance relative to lower bound for optimal
plot2 = ggplot(df, aes('jobDist', 'cost/lowerBound')) + geom_boxplot(aes(colour = 'jobDist'), alpha = 0.2) + facet_grid('m/n ~ n', labeller = "label_both") + theme(axis_text_x = element_blank()) + ggtitle("boxplots: cost/lowerBoundForOptimal")
# plot3 = ggplot(df, aes('jobDist', 'np.log10(cost/lowerBound - 1)')) + geom_boxplot(aes(colour = 'jobDist'), alpha = 0.2) + facet_grid('m/n ~ n', labeller = "label_both") + theme(axis_text_x = element_blank()) + ggtitle("boxplots: cost/lowerBoundForOptimal - 1 (log10 scale)") + labs(y = "log_10(cost/lowerBound - 1)")


dfLB = pd.read_csv("liftedLowerBoundsNew.csv")
df = pd.concat([df, dfLB["liftedLowerBound"]], axis = 1)

plot3 = ggplot(df, aes('jobDist', 'np.log10(cost/liftedLowerBound - 1)')) + geom_boxplot(aes(colour = 'jobDist'), alpha = 0.2) + facet_grid('m/n ~ n', labeller = "label_both") + theme(axis_text_x = element_blank()) + ggtitle("boxplots: cost/lowerBoundForOptimal - 1 (log10 scale)") + labs(y = "log_10(cost/liftedLowerBound - 1)")

plot4 = ggplot(df, aes('jobDist', 'timeToBestCost')) + geom_boxplot(aes(colour = 'jobDist'), alpha = 0.2) + facet_grid('m/n ~ n', labeller = "label_both") + theme(axis_text_x = element_blank()) + ggtitle("boxplots: time to best cost") + labs(y = "timeToBestCost")

ggsave(plot1, "Q2comparison.pdf")
ggsave(plot2, "Q2performance.pdf")
ggsave(plot3, "Q2performanceLogScale.pdf")
ggsave(plot4, "Q2timeToBestCost.pdf")