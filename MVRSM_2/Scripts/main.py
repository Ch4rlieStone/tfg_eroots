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


if __name__ == '__main__':

	# Read arguments
	
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-f', '--func', help='Objective function',
                        default='wind_offshore', type=str)   
    parser.add_argument('-mix', '--kernel_mix',
                        help='Mixture weight for production and summation kernel. Default = 0.0', default=0.5,
                        type=float)
    parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 100',
                        default=10, type=int)
    parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 20',
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
	
    if obj_func == 'wind_offshore':
        ff = wind_offshore.costac_2
        d = 13
        lb = np.array([3, 3, 1, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 500e6])  # Lower bound
        ub = np.array([3, 3, 1, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 500e6])  # Upper bound
        num_int = 7
    else:
        raise NotImplementedError
	
    x0 =np.zeros(d) # Initial guess
    x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
    x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	
	
    rand_evals = 24 # Number of random iterations, same as initN above (24)
    max_evals = n_itrs+rand_evals # Maximum number of MVRSM iterations, the first <rand_evals> are random
	
	
	###########
	## MVRSM ##
	###########

    def obj_MVRSM(x):
        result = ff(x)
        return result
    def run_MVRSM():
        solX, solY, model = MVRSM.MVRSM_minimize(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals)		
        print("Solution found: ")
        print(f"X = {solX}")
        print(f"Y = {solY}")
    for i in range(n_trials):
        if obj_func == 'dim10Rosenbrock' or obj_func == 'dim53Rosenbrock' or obj_func == 'dim238Rosenbrock':
            print(f"Testing MVRSM on the {d}-dimensional Rosenbrock function with integer constraints.")
            print("The known global minimum is f(1,1,...,1)=0")
        else:
            print("Start MVRSM trials")
        run_MVRSM()
