# MVRSM demo
# By Laurens Bliek, 16-03-2020
# Supported functions: 'func2C', 'func3C', 'dim10Rosenbrock',
# 'linearmivabo', 'dim53Rosenbrock', 'dim53Ackley', 'dim238Rosenbrock'
# Example: python demo.py -f dim10Rosenbrock  -n 10 -tl 4
# Here, -f is the function to be optimised, -n is the number of iterations, and -tl is the total number of runs.
# Afterward, use plot_result.py for visualisation.

import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import numpy as np
import pickle
import time
import synth_functions
import mvrsm
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial
from scipy.optimize import rosen
import matplotlib.pyplot as plt



if __name__ == '__main__':

    ff = synth_functions.dim10Rosenbrock
    d = 10 # Total number of variables
    lb = -2*np.ones(d).astype(int) # Lower bound
    ub = 2*np.ones(d).astype(int) # Upper bound
    num_int = 3 # number of integer variables
    lb[0:num_int] = 0
    ub[0:num_int] = num_int+1
	
    x0 = np.zeros(d) # Initial guess
    x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
    x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	
    rand_evals = 100 # Number of random iterations, same as initN above (24)
    n_itrs = 500
    n_trials = 10
    max_evals = n_itrs+rand_evals # Maximum number of MVRSM iterations, the first <rand_evals> are random
	
	
	###########
	## MVRSM ##
	###########
	
    def obj_MVRSM(x):
        #print(x[0:num_int])
        h = np.copy(x[0:num_int]).astype(int)
        result = ff(h,x[num_int:])
        return result

    def run_MVRSM():
        solX, solY, model, logfile = mvrsm.MVRSM_minimize(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals)		
        print("Solution found: ")
        print(f"X = {solX}")
        print(f"Y = {solY}")
        print(solX)
        print(solY)
        print()

        #mvrsm.plot_results(logfile)
        return solX, solY

    fobj_vec = []
    x_vec = []
    for i in range(n_trials):
        print(f"Testing MVRSM on the {d}-dimensional Rosenbrock function with integer constraints.")
        print("The known global minimum is f(1,1,...,1)=0")
        xs, ys = run_MVRSM()
        # fobj_vec.append(run_MVRSM())
        x_vec.append(xs)
        fobj_vec.append(ys)
        print(i)

    plt.plot(np.arange(n_trials), fobj_vec)
    plt.show()
		
		
	

		