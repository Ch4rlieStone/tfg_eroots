import numpy as np
import costac_2
import mvrsm
import MVRSM_mo_scaled
import matplotlib.pyplot as plt



if __name__ == '__main__':

    #  ff = synth_functions.dim3constrRosenbrock  #dim2constrRosenbrock 
    ff = costac_2.costac_2
    d = 13 # Total number of variables
    lb = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 450e6])  # Lower bound
    ub = np.array([3, 3, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 2000e6])  # Upper bound
    #lb = np.array([-1.5, -0.5]) # Lower bound
    #ub = np.array([1.5, 2.5])
    #lb = np.array([-1.5, -1.5]) # Lower bound
    #ub = np.array([1.5, 1.5])
    #lb = np.array([0, -1.5, -1.5]) #Lower bound
    #ub = np.array([1, 1.5, 1.5])
    num_int = 7 # number of integer variables
    #lb[0:num_int] = 0
    #ub[0:num_int] = num_int+1
	
    x0 = np.zeros(d) # Initial guess
    x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
    x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	
    rand_evals = 200 # Number of random iterations, same as initN above (24)
    n_itrs = 200
    n_trials = 1
    max_evals = n_itrs+rand_evals # Maximum number of MVRSM iterations, the first <rand_evals> are random
	
	
	###########
	## MVRSM ##
	###########

    def obj_MVRSM_ori(x):
        #print(x[0:num_int])
        h = np.copy(x[0:num_int]).astype(int)
        result, cost_inv, cost_tech = ff(h,x[num_int:])
        return result
    
    def obj_MVRSM(x):
        #print(x[0:num_int])
        h = np.copy(x[0:num_int]).astype(int)
        cost_inv, cost_tech = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x[7],x[8],x[9],x[10],x[11],x[12])
        return cost_inv, cost_tech
    
    # cmap = plt.get_cmap()
    def run_MVRSM():
        # solX, solY, model, logfile = MVRSM_mo_scaled.MVRSM_mo_scaled(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals, args=(), n_objectives=2)	
        ysol, xsol, ypop, xpop, fpop = MVRSM_mo_scaled.MVRSM_mo_scaled(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals, args=(), n_objectives=2)
        
        
        print("Solution found: ")
        print(f"solution = {xsol}")
        print(f"f(x,y) = {ysol}")
        #print(solX)
        #print(solY)
        print()

        #  mvrsm.plot_results(logfile)
        return xsol, ysol, xpop, ypop, fpop


    fobj_vec = []
    x_vec = []
    for i in range(n_trials):
        print(f"Testing MVRSM on the HVAC cost function with integer constraints.")
        # print("The known global minimum is f(1,1,...,1)=0")
        xs, ys, xp, yp, fp = run_MVRSM()
        # fobj_vec.append(run_MVRSM())
        x_vec.append(xs)
        fobj_vec.append(ys)
        print(i)

    # plt.plot(np.arange(max_evals), fp)
    # plt.show()
    cmap = plt.get_cmap('viridis', max_evals)
    for i in range(max_evals):
        plt.scatter(yp[i, 0], yp[i, 1], color=cmap(i))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_evals-1))
    plt.colorbar(sm, label='Point index')
    #plt.ylim(0, 50)
    #plt.xlim(0, 50)
    # plt.scatter(yp[:,0], yp[:,1])
    plt.show()

    """
    sorted_y_, sorted_x_, y_population_ = MVRSM_new.MVRSM_minimize(obj_MVRSM,
                                                               x0=x0,
                                                               lb=lb,
                                                               ub=ub,
                                                               num_int=num_int,
                                                               max_evals=400,
                                                               rand_evals=100,
                                                               args=())

    print("Best solutions:")
    print(sorted_y_)
    print(sorted_x_)

    plt.scatter(sorted_y_[:, 0], sorted_y_[:, 1], 1, )
    plt.plot(y_population_[:, 0])
    plt.show()
    """