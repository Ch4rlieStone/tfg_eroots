import numpy as np
import matplotlib.pyplot as plt
import costac_2



trials = 1000
ff = costac_2.costac_2
random_check = np.zeros((trials,6))
d = 13
num_int = 7
lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 450e6])  # Lower bound
ub = np.array([3, 2, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])  # Upper bound

p_owflist = np.linspace(1, 10, trials)
x_history = np.zeros((trials, d))

p_owf = 5
for i in range(trials):
        
        x0 = np.zeros(d) # Initial guess
        x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
        x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
        x_history[i,:] = x0
        h = np.copy(x0[0:num_int]).astype(int)
        x_history[i,:] = x0
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
        """
        x0 = np.array([2,3,1,1,1,1,1,0.0,0.0,0.8,0.0,0.0,2000e6])
        x_history[i,:] = x0
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
        p_owf = p_owflist[i]
        """
        cost_invest, cost_tech, cost_full = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_check[i,:] = [cost_invest, cost_tech, cost_full[10], cost_full[2], cost_full[3], cost_full[11]]
        
        
     
# print(random_check)
plt.scatter(random_check[:,0], random_check[:,1], color='blue')
plt.ylim(0,1000)
plt.xlim(0,500)
# Find the index of the row with the smallest sum
min_sum_row_index = np.argmin(np.sum(random_check, axis=1))

#print(random_check)
# Print the row
#print(random_check[min_sum_row_index])
#print(x_history[min_sum_row_index,:])

plt.show()
"""  
normalized_cost_volover = random_check[:,2] / np.max(random_check[:,2])
normalized_cost_losses = random_check[:,3] / np.max(random_check[:,3])
normalized_cost_react = random_check[:,4] / np.max(random_check[:,4])
normalized_cost_volunder = random_check[:,5] / np.max(random_check[:,5])

plt.plot(p_owflist, normalized_cost_volover, label='Overvoltage Cost')
plt.plot(p_owflist, normalized_cost_losses, label='Losses Cost')
plt.plot(p_owflist, normalized_cost_react, label='Reactive Power grid Cost')
plt.plot(p_owflist, normalized_cost_volunder, label='Undervoltage Cost')
plt.xlabel('Power Injection')
plt.ylabel('Normalized Costs')
plt.title('Evolution of  Costs for Different Power Injections')
plt.legend()
plt.show()

# np.savetxt("random_check.csv", random_check, delimiter=",")
"""