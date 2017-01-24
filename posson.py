import random
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
def cdf(y, x):
	prob = np.zeros((len(x)))
	for i in range(len(x)):
		if(i == 0):
			prob[i] = y[i]
		prob[i] = y[i] + prob[i-1]
	return prob
def poisson_probability(actual, mean):
    data = stats.binom.rvs(n=10, p = 0.2, size=1000)
    print (data)
    print(np.mean(data))
   	
poisson_probability(100, 10)

