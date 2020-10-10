-Interpretation of File Name Format-

The filename describes the parameters of the test instance, including the allowable time, number of machines, number of jobs, and distribution of job sizes.

Example:

'10 2 4 centred 1.txt' means
  10 = allowableTime in seconds
  2 = mfrac which is defined by number of machines = (mfrac/5) * number of jobs
  4 = nmagnitude which is defined by number of jobs = 10^nmagnitude
  'centred' means the distribution of jobs has many medium-sized jobs, few large jobs and few small jobs 
  1 is the replicate number of this configuration of test instance parameters. There are 10 replicates for each configuration of test instance parameters.
  
Each job size distribution is modelled with a beta-binomial distribution. The kinds of job size distributions are described as follows, with job sizes integer in the range [1, 40000]:

"discreteUniform" : job sizes are uniformly distributed between 1 and 40000
"largeJobs" : many large jobs, few small jobs
"extremes" : many large jobs, many small jobs, moderate number of medium-sized jobs
"centred" : many medium-sized jobs, few large jobs and few small jobs

If you have any suggestions for new job size distributions, they can be added without much additional work.
