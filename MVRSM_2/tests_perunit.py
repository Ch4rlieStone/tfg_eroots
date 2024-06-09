import numpy as np
import costac_2

x0 = np.array([2,2,1,1,1,1,1,0.0,0.0,0.8,0.0,0.0,5000e6])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
p_owf = 5

# Call the run_p_ function with the fixed set of variables
ff = costac_2.costac_2
cost_invest, cost_tech, cost_full = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)


