import numpy as np
import wind_offshore
import matplotlib.pyplot as plt



trials = 400
ff = wind_offshore.costac_2
random_check = np.zeros((trials,1))
d = 13
num_int = 7
lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 400e6])  # Lower bound
ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 600e6])  # Upper bound

for i in range(trials):
        
        x0 = np.zeros(d) # Initial guess
        x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
        x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
        h = np.copy(x0[0:num_int]).astype(int)
        result = ff(x0)
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_check[i] = result
        
print(random_check)
plt.scatter(range(trials), random_check, color='blue')
plt.ylim(0,2000)
plt.show()

np.savetxt("random_check.csv", random_check, delimiter=",")