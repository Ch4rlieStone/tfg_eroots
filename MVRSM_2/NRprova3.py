# This is a simple Newton-Raphson algorithm to solve the power flow problem in a 5-bus system
# Carles Roca Reverter, 2024
import numpy as np


# 1. Define grid

# 1.1 Global grid
Sbase = 100  # MVA
V_ref = 220  # kV
f = 50  # Hz
Y_ref = Sbase / V_ref**2  # 1 / ohm

# 1.2 Trafo

# Trafo parameters
U_rtr = 220e3  # V
P_Cu = 60e3  # W
P_Fe = 40e3  # W
u_k = 0.18  # p.u.
i_o = 0.012  # p.u.
S_rtr = 500e6  # VA

# Computation of Y parallel
G_tri = (P_Fe / U_rtr**2)
B_tri = -(i_o * (S_rtr / U_rtr**2))
Y_tr = (G_tri + 1j * B_tri) / Y_ref

# Computation of Y series
R_tr = P_Cu / S_rtr
X_tr = np.sqrt(u_k**2 - R_tr**2)
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
Y_pi = (Y * l / 4 * np.tanh(theta / 2) / (theta / 2)) / Y_ref
G_pi = np.real(Y_pi)
B_pi = np.imag(Y_pi)
Z_piserie = (Z * l / 2 * np.sinh(theta) / theta)
Y_piserie = (1 / Z_piserie)  / Y_ref

# 1.4 Compensator
Y_l = 0 + 1j * (-B_pi)

# 1.5 Grid connection
Y_g = (4.95 - 0.49j)  # p.u.

# 1.6 Wind farm
p_owf = 5  # p.u
q_owf = 0  # p.u

Y_bus = np.array([[Y_trserie + Y_tr + Y_l, -Y_trserie, 0, 0, 0, 0],
                  [-Y_trserie, Y_piserie + Y_pi + Y_l + Y_trserie, - Y_piserie, 0, 0, 0],
                  [0, -Y_piserie, 2 * Y_piserie + 2 * Y_pi + Y_l, -Y_piserie, 0, 0],
                  [0, 0, -Y_piserie, Y_piserie + Y_pi + Y_l + Y_trserie, -Y_trserie, 0],
                  [0, 0, 0, -Y_trserie, Y_trserie + Y_tr + Y_l + Y_g, -Y_g],
                  [0, 0, 0, 0, -Y_g, Y_g]])

# 2. Run algorithm
epsilon = 1
k = 0


###
# (this value is P_owf/100e6  S_base=100e6) In this case, we consider 500 MW owf
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

while epsilon > 1e-6 and k < 100:

    k = k + 1
    x = np.concatenate((angles, V))
    P = np.zeros(nbus - 1)
    Q = np.zeros(nbus - 1)
    angle_wslack[:nbus - 1] = angles

    V_wslack[:nbus - 1] = V

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
    print("mismatch =", deltaPQ)

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
                
                J12[i, j] = P[i, 0]/abs(V[i, 0]) + \
                    Y_bus[i, i].real*abs(V[i, 0])
            else:
                J12[i, j] = abs(V[i, 0])*(Y_bus[i, j].real*np.cos(angles[i, 0] -
                                                                  angles[j, 0]) + Y_bus[i, j].imag*np.sin(angles[i, 0]-angles[j, 0]))

    for i in range(nbus - 1):
        
        for j in range(nbus - 1):
            
            if j == i:
                
                J21[i, j] = P[i, 0]-(V[i, 0])**2*Y_bus[i, i].real
            else:
                J21[i, j] = - abs(V[i, 0])*abs(V[j, 0])*(Y_bus[i, j].real*np.cos(
                    angles[i, 0]-angles[j, 0]) + Y_bus[i, j].imag*np.sin(angles[i, 0]-angles[j, 0]))

    for i in range(nbus - 1):
        
        for j in range(nbus - 1):
            
            if j == i:
                
                J22[i, j] = Q[i, 0]/abs(V[i, 0]) - \
                    Y_bus[i, i].imag*abs(V[i, 0])
            else:
                J22[i, j] = abs(V[i, 0])*(Y_bus[i, j].real*np.sin(angles[i, 0] -
                                                                  angles[j, 0]) - Y_bus[i, j].imag*np.cos(angles[i, 0]-angles[j, 0]))

    J = np.concatenate((np.concatenate((J11, J12), axis=1),
                       np.concatenate((J21, J22), axis=1)), axis=0)
    # print(J)
    # print("determinantJ =", np.linalg.det(J))

    # now we have to solve the system
    # print(J)
    # print("determinantJ =", np.linalg.det(J))

    # now we have to solve the system

    # WE WILL TRY JACOBIAN WITH SMALL VARIATIONS OF X VALUES#

    delta_x = np.linalg.solve(J, deltaPQ)
    # print("delta_x =",delta_x)
    x_new = x + delta_x  # we have updated value of angles and V [10X1] matrix
    # print("x =",x)
    # (note this vector does not include slack!)
    # print("xnew =",x_new)
    angles = x_new[0:5]
    V = x_new[5:10]

    # we check error value
    epsilon = (max(abs(deltaPQ[:, 0])))
    # print("error =", epsilon)

print(x_new)
print(epsilon)
print("iterations =", k)
