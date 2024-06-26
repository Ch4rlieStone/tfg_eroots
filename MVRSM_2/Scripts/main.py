import sys
import argparse
import os
import numpy as np
import pickle
import time
import mvrsm as MVRSM
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial
import wind_offshore
import matplotlib.pyplot as plt


if __name__ == '__main__':

	# Read arguments
	
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-f', '--func', help='Objective function',
                        default='wind_offshore', type=str)   
    parser.add_argument('-mix', '--kernel_mix',
                        help='Mixture weight for production and summation kernel. Default = 0.0', default=0.5,
                        type=float)
    parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 100',
                        default=100, type=int)
    parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 1',
                        default=1, type=int)
    parser.add_argument('-b', '--batch', help='Batch size (>1 for batch CoCaBO and =1 for sequential CoCaBO). Default = 1',
                        default=1, type=int)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    obj_func = args.func
    kernel_mix = args.kernel_mix
    n_itrs = args.max_itr
    n_trials = args.trials
    batch = args.batch
    
    n_trials = 1
	
    if obj_func == 'wind_offshore':
        ff = wind_offshore.costac_2
        d = 13
        lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 400e6])  # Lower bound
        ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 600e6])  # Upper bound
        num_int = 7
    else:
        raise NotImplementedError
	
    x0 =np.zeros(d) # Initial guess
    x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
    x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	
	
    rand_evals = 24 # Number of random iterations, same as initN above (24)
    n_itrs = 400 # Number of MVRSM iterations (200)
    max_evals = n_itrs+rand_evals # Maximum number of MVRSM iterations, the first <rand_evals> are random
	
	
	###########
	## MVRSM ##
	###########
    
    def obj_MVRSM(x):
        result = ff(x)
        return result
    def run_MVRSM():
        logfile = 'MVRSM_log.txt'
        solX, solY, bests_ys,   model = MVRSM.MVRSM_minimize(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals)		
        print("Solution found: ")
        
        print(f"X = {solX}")
        print(f"Y = {solY}")


        # MVRSM.plot_results(logfile)
        return solX, solY, bests_ys
        

    fobj_vec = []
    x_vec = []
    for i in range(n_trials):
        if obj_func == 'dim10Rosenbrock' or obj_func == 'dim53Rosenbrock' or obj_func == 'dim238Rosenbrock':
            print(f"Testing MVRSM on the {d}-dimensional Rosenbrock function with integer constraints.")
            print("The known global minimum is f(1,1,...,1)=0")
        else:
            print("Start MVRSM trials")
        
        xs , ys , listys = run_MVRSM()
        x_vec.append(xs)
        fobj_vec.append(ys)
        #print(listys)
        #print(len(listys))

    for i in range(max_evals):

        plt.scatter(i, listys[i], color='blue')
    plt.ylim(0,1000)
    plt.xlabel('Iteration')
    plt.ylabel('Objective function value')
    plt.show()
    
    np.savetxt("listys.csv", listys, delimiter=",")

