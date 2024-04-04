import numpy as np
d = 13 # Number of variables
lb = np.array([2, 1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 200e6])  # Lower bound
ub = np.array([3, 3, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 800e6])  # Upper bound

num_int = 7 # number of integer variables
x0 = np.zeros(d) # Initial guess
x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)

print(x0)