import numpy as np
import costac_2
import Altres.mvrsm
import MVRSM_mo_scaled
import matplotlib.pyplot as plt



if __name__ == '__main__':

    ff = costac_2.costac_2
    d = 13 # Total number of variables
    lb = np.array([3, 3, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 500e6])  # Lower bound
    ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 800e6])  # Upper bound
    # lb = np.array([1, 1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 500e6])  # Lower bound
    # ub = np.array([3, 3, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 800e6])  # Upper bound

    num_int = 7 # number of integer variables
    x0 = np.zeros(d) # Initial guess
    x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
    x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	
    rand_evals = 500 # Number of random iterations
    n_itrs = 500
    n_trials = 1
    max_evals = n_itrs+rand_evals # Maximum number of MVRSM iterations, the first <rand_evals> are random
	
	
	###########
	## MVRSM ##
	###########

    # try original mvrsm if that does not work fine

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

    def obj_MVRSM_josep(x):
        h = np.copy(x[0:num_int]).astype(int)
        cost_inv, cost_tech = ff(*h, *x[num_int:])
        return cost_inv, cost_tech
    
    # cmap = plt.get_cmap()
    def run_MVRSM():
        # solX, solY, model, logfile = MVRSM_mo_scaled.MVRSM_mo_scaled(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals, args=(), n_objectives=2)	
        # ysol, xsol, ypop, xpop, fpop = MVRSM_mo_scaled.MVRSM_mo_scaled(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals, args=(), n_objectives=2)
        ysol, xsol, ypop, xpop, fpop = MVRSM_mo_scaled.MVRSM_mo_scaled(obj_MVRSM_josep, x0, lb, ub, num_int, max_evals, rand_evals, args=(), n_objectives=2)
        
        
        print("Solution found: ")
        print(f"solution = {xsol}")
        print(f"f(x,y) = {ysol}")
        #print(solX)
        #print(solY)
        print()

        # mvrsm.plot_results(logfile)
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
    #     plt.subplot(1,2,2)
        plt.scatter(yp[i, 0], yp[i, 1], color=cmap(i))
        # plt.scatter(ys[i, 0], ys[i, 1], color=cmap(i))
    # for i in range(len(ys)):
    #     plt.subplot(1,2,1)
        # plt.scatter(ys[i, 0], ys[i, 1], color=cmap(i))
        

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_evals-1))
    plt.colorbar(sm, label='Point index')
    plt.ylim(0, 1500)
    plt.xlim(0, 500)
    # plt.scatter(yp[:,0], yp[:,1])
    plt.show()
    print(xs)
    print(ys)
