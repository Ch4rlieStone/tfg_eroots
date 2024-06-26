# This is a simple Newton-Raphson algorithm to solve the power flow problem in a 5-bus system
# Carles Roca Reverter, 2024
import numpy as np

# INPUT UNKNOWS TO SOLVE
u_i = 220e3  # transmission voltage
n_cables = 1  # number of cables
#  




# 1. Define grid

# 1.1 Global grid
Sbase = 100e6  # VA
V_ref = 220e3  # V
f = 50  # Hz
Y_ref = Sbase / V_ref**2  # 1 / ohm

# 1.2 Trafo

# Trafo parameters
U_rtr = u_i  # V
P_Cu = 60e3  # W
P_Fe = 40e3  # W
u_k = 0.18  # p.u.
i_o = 0.012  # p.u.
S_rtr = 500e6  # VA

# Computation of Y parallel
G_tri = (P_Fe / U_rtr**2)
B_tri = - (i_o * (S_rtr / U_rtr**2))
Y_tr = (G_tri + 1j * B_tri) / Y_ref

# Computation of Y series
R_tr = P_Cu / S_rtr
# X_tr = np.sqrt(u_k**2 - R_tr**2)
X_tr = np.sqrt((u_k * (U_rtr**2 / S_rtr))**2 - R_tr**2)
Z_tr = R_tr + 1j * X_tr
Y_trserie = (1 / Z_tr) / Y_ref

# 1.3 Cables
l = 100  # km

R = 0.0072  # ohm/km
Cap = 0.17e-6  # F/km
L = 0.40e-3   # H/km
Y = 1j * (2 * np.pi * f * Cap / 2)
Z = R + 1j * (2 * np.pi * f * L)
theta = l / 2 * np.sqrt(Z * Y)
Y_pi = n_cables * (Y * l / 4 * np.tanh(theta / 2) / (theta / 2)) / Y_ref
G_pi = np.real(Y_pi)
B_pi = np.imag(Y_pi)
Z_piserie = (Z * l / 2 * np.sinh(theta) / theta)
Y_piserie = n_cables * (1 / Z_piserie)  / Y_ref

# 1.4 Compensator
Y_l1 = 0 + 1j * ( B_pi)
Y_l2 = 0 + 1j * ( B_pi)
Y_l3 = 0 + 1j * ( B_pi)
Y_l4 = 0 + 1j * ( B_pi)
Y_l5 = 0 + 1j * ( B_pi)
#  Y_l = (0 + 1 / (1j * 2 * np.pi * f * l_r)) / Y_ref  # explicit formula to use when using l_r (reactor inductance) as unknown
#  Y_l = 0 
#  (WE HAVE TO CHECK THAT SINCE NO COMPENSATION IS YIELDING SMALLER LOSSES) josep says it might make sense since we have overvoltages that reduce losses

# 1.5 Grid connection
Y_g = (4.95 - 0.49j) / Y_ref # p.u.


# 1.6 Wind farm
p_owf = 5  # p.u (this value is P_owf/100e6  S_base=100e6) In this case, we consider 500 MW owf
q_owf = 0 # p.u

Y_bus = np.array([[Y_trserie + Y_tr + Y_l1, -Y_trserie, 0, 0, 0, 0],
                  [-Y_trserie, Y_piserie + Y_pi  + Y_trserie, - Y_piserie, 0, 0, 0],
                  [0, -Y_piserie, 2 * Y_piserie + 2 * Y_pi + Y_l3 , -Y_piserie, 0, 0],
                  [0, 0, -Y_piserie, Y_piserie + Y_pi  + Y_trserie, -Y_trserie, 0],
                  [0, 0, 0, -Y_trserie, Y_trserie + Y_tr + Y_g + Y_l5, -Y_g],
                  [0, 0, 0, 0, -Y_g, Y_g]])


# 2. Run algorithm
epsilon = 1
k = 0


###
 
nbus = 6

V_slack = 1.0
angle_slack = 0.0

V = np.ones(nbus - 1, dtype=float)
V_wslack = np.empty(nbus, dtype=float)
V_wslack[:nbus - 1] = V
V_wslack[nbus - 1] = V_slack

angles = np.zeros(nbus - 1, dtype=float)
angle_wslack = np.empty(nbus, dtype=float)
angle_wslack[:nbus - 1] = angles
angle_wslack[nbus - 1] = angle_slack

x0 = np.concatenate([angles, V])
x = x0

P_obj = np.array([p_owf, 0, 0, 0, 0])
Q_obj = np.array([q_owf, 0, 0, 0, 0])
PQ_obj = np.concatenate([P_obj, Q_obj])

while epsilon > 1e-5 and k < 100:

    k = k + 1
    x = np.concatenate((angles, V))
    P = np.zeros(nbus - 1)
    Q = np.zeros(nbus - 1)
    

    # Compute power mismatch function
    for i in range(nbus - 1):
        for j in range(nbus):
            P[i] = P[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.cos(
                angle_wslack[i]-angle_wslack[j]) + np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]))

    for i in range(nbus - 1):
        for j in range(nbus):
            Q[i] = Q[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.sin(
                angle_wslack[i]-angle_wslack[j]) - np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]))

    # compute error in mismatch function
    PQ = np.concatenate((P, Q))
    deltaPQ = (PQ_obj - PQ)
    #print("mismatch =", deltaPQ)

    # Now we will build the Jacobian
    J11 = np.zeros((nbus - 1, nbus - 1))  # P wrt angle
    J12 = np.zeros((nbus - 1, nbus - 1))  # P wrt V
    J21 = np.zeros((nbus - 1, nbus - 1))  # Q wrt angle
    J22 = np.zeros((nbus - 1, nbus - 1))  # Q wrt V

    for i in range(nbus - 1):
        for j in range(nbus - 1):
            
            if j == i:
                
                J11[i, j] = - Q[i] - (V[i]**2) * np.imag(Y_bus[i, i])

            else:
                J11[i, j] = abs(V[i]) * abs(V[j]) * (np.real(Y_bus[i, j]) * np.sin(
                    angles[i] - angles[j]) - np.imag(Y_bus[i, j]) * np.cos(angles[i]-angles[j]))

    

    for i in range(nbus - 1):
        
        for j in range(nbus - 1):
            
            if j == i: 
                
                J12[i, j] = P[i] / abs(V[i]) + \
                    np.real(Y_bus[i, i]) * abs(V[i])
            else:
                J12[i, j] = abs(V[i]) * (np.real(Y_bus[i, j]) * np.cos(angles[i] -
                    angles[j]) + np.imag(Y_bus[i,j]) * np.sin(angles[i]-angles[j]))

    for i in range(nbus - 1):
        
        for j in range(nbus - 1):
            
            if j == i:
                
                J21[i, j] = P[i]-(V[i,])**2 * np.real(Y_bus[i, i])
            else:
                J21[i, j] = - abs(V[i])*abs(V[j]) * (np.real(Y_bus[i, j]) * np.cos(
                    angles[i]-angles[j]) + np.imag(Y_bus[i, j]) * np.sin(angles[i]-angles[j]))

    for i in range(nbus - 1):
        
        for j in range(nbus - 1):
            
            if j == i:
                
                J22[i, j] = Q[i]/abs(V[i]) - \
                    np.imag(Y_bus[i, i]) * abs(V[i])
            else:
                J22[i, j] = abs(V[i]) * (np.real(Y_bus[i, j]) * np.sin(angles[i] -
                                                                  angles[j]) - Y_bus[i, j].imag*np.cos(angles[i]-angles[j]))

    J = np.concatenate((np.concatenate((J11, J12), axis=1),
                       np.concatenate((J21, J22), axis=1)), axis=0)
   

    # now we have to solve the system

    delta_x = np.linalg.solve(J, deltaPQ)
    
    x_new = x + delta_x  # we have updated value of angles and V [1X10] matrix (note this vector does not include slack!)
    
    angles = x_new[0:5]
    V = x_new[5:10]
    angle_wslack[:nbus - 1] = angles
    V_wslack[:nbus - 1] = V

    # we check error value
    epsilon = (max(abs(deltaPQ)))
    

print("x_final =" , x_new)
print("error =", epsilon)
print("iterations =", k)

#  Now we can get the P ad Q values of the slack node 6 (note that the slack node is the last one of the vectors)

p_wslack = np.zeros(nbus)
for i in range(nbus):
        for j in range(nbus):
            p_wslack[i] = p_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.cos(
                angle_wslack[i]-angle_wslack[j]) + np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]))
            
print("p_final =" , p_wslack)

q_wslack = np.zeros(nbus)
for i in range(nbus):
        for j in range(nbus):
            q_wslack[i] = q_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.sin(
                angle_wslack[i]-angle_wslack[j]) - np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]))
            
print("q_final =" , q_wslack)

# Now we compute the injected currents at each node

curr_inj = np.zeros(nbus, dtype = 'cfloat')
for i in range(nbus):
    for j in range(nbus):
        curr_inj[i] = curr_inj[i] + Y_bus[i,j] * (V_wslack[j] * (np.cos(angle_wslack[j]) +
                                                    np.sin(angle_wslack[j]) *1j ))
        

print("currents injections =", curr_inj)

#  When it comes to overcurrents, we are interested in line currents, not node injection currents.
#  We compute now line currents

i_21 = (V[1] - V[0]) * Y_trserie
i_32 = (V[2] - V[1]) * Y_piserie
i_43 = (V[3] - V[2]) * Y_piserie
i_54 = (V[4] - V[3]) * Y_trserie


#  We compute the AC power losses

p_lossac = Sbase * (p_owf + p_wslack[5]) * 1e-3  # KW

print ("ac_losses =", p_lossac,"KW")