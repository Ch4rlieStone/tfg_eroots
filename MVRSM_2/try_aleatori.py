import numpy as np
import matplotlib.pyplot as plt
import costac_2



trials = 1000
ff = costac_2.costac_2
random_check = np.zeros((trials,2))
d = 13
num_int = 7
lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 200e6])  # Lower bound
ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 800e6])  # Upper bound

for i in range(trials):
        
        x0 = np.zeros(d) # Initial guess
        x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
        x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
        h = np.copy(x0[0:num_int]).astype(int)
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
        cost_invest, cost_tech = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr)
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_check[i,:] = [cost_invest, cost_tech]
        
print(random_check)
plt.scatter(random_check[:,0], random_check[:,1], color='blue')
plt.ylim(0,500)
plt.xlim(0,500)
plt.show()

# np.savetxt("random_check.csv", random_check, delimiter=",")