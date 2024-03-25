import numpy as np
import cmath
import random

def costac(vol , n_cables , react_bi , react_val , S_rtr):

    # 1. Define grid

    # 1.1 Global grid
    Sbase = 100e6  # VA
    V_ref = 220e3  # V
    I_ref = Sbase / (np.sqrt(3) * V_ref)
    f = 50  # Hz
    Y_ref = Sbase / V_ref**2  # 1 / ohm

    # WPPT TO EVALUATE parameters

    l = 100  #  distance to shore in km
    p_owf = 5  # p.u (this value is P_owf/100e6  S_base=100e6) In this case, we consider 500 MW owf
    q_owf = 0 # p.u  We assume no reactive power is generated at plant

    #  Integer
#  vol = 3
#  transmission voltage (110 [1],150 [2],220 [3]) kV
    if vol == 3:
        u_i = 220e3
        R = 0.0067  # ohm/km
        Cap = 0.17e-6  # F/km
        L = 0.40e-3   # H/km
        A = 3.181
        B = 0.11
        C = 0.0116
        D = 17e4
        E = 8.98
        I_rated = 540  # how we get this value? 

    if vol == 2:
        u_i = 150e3
        R = 0.0067  # ohm/km
        Cap = 0.19e-6  # F/km
        L = 0.38e-3   # H/km
        A = 1.971
        B = 0.209
        C = 0.0166
        D = 17e4
        E = 8.98
        I_rated = 500

    if vol == 1:
        u_i = 110e3
        R = 0.0067  # ohm/km
        Cap = 0.24e-6  # F/km
        L = 0.36e-3   # H/km
        A = 1.3295
        B = 0.417
        C = 0.01855
        D = 17e4
        E = 8.98
        I_rated= 470

    # 1.2 Trafo

    # Trafo parameters
    U_rtr = u_i  # V
    P_Cu = 60e3  # W
    P_Fe = 40e3  # W
    u_k = 0.18  # p.u.
    i_o = 0.012  # p.u.

    # Computation of Y parallel
    G_tri = (P_Fe / U_rtr**2)
    B_tri = - (i_o * (S_rtr / U_rtr**2))
    Y_tr = (G_tri + 1j * B_tri) / Y_ref

    # Computation of Y series
    R_tr = P_Cu / S_rtr
    #X_tr = np.sqrt(u_k**2 - R_tr**2)
    X_tr = np.sqrt((u_k * (U_rtr**2 / S_rtr))**2 - R_tr**2)
    Z_tr = R_tr + 1j * X_tr
    Y_trserie =  (1 / Z_tr) / Y_ref

    # 1.3 Cables
    l = 100  # km

    R = 0.0067  # ohm/km
    Cap = 0.17e-6  # F/km
    L = 0.40e-3   # H/km
    Y = 1j * (2 * np.pi * f * Cap / 2)
    Z = R + 1j * (2 * np.pi * f * L)
    theta = l / 2 * np.sqrt(Z * Y)
    Y_pi = n_cables * (Y * l / 4 * np.tanh(theta / 4) / (theta / 4)) / Y_ref
    G_pi = np.real(Y_pi)
    B_pi = np.imag(Y_pi)
    Z_piserie = (Z * l / 2 * np.sinh(theta /2) / (theta/2)) 
    Y_piserie = n_cables * (1 / Z_piserie)  / Y_ref 

    # 1.4 Compensator
    if react_bi[0] == 1:
        Y_l1 = react_val[0]
    else:
        Y_l1 = 0

    if react_bi[1] == 1:
        Y_l2 = react_val[1]
    else:
        Y_l2 = 0

    if react_bi[2] == 1:
        Y_l3 = react_val[2]
    else:
        Y_l3 = 0

    if react_bi[3] == 1:
        Y_l4 = react_val[3]
    else:
        Y_l4 = 0
    
    if react_bi[4] == 1:
        Y_l5 = react_val[4]
    else:
        Y_l5 = 0

    # 1.5 Grid connection
    Y_g = (4.950 - 0.495j) / Y_ref # p.u.  Computed from SCR = 5 and Xg/Rg = 10

    Y_bus = np.array([[Y_trserie + Y_tr + Y_l1, -Y_trserie, 0, 0, 0, 0],
                  [-Y_trserie, Y_piserie + Y_pi + Y_l2 + Y_trserie, - Y_piserie, 0, 0, 0],
                  [0, -Y_piserie, 2 * Y_piserie + 2 * Y_pi + Y_l3, -Y_piserie, 0, 0],
                  [0, 0, -Y_piserie, Y_piserie + Y_pi + Y_l4 + Y_trserie, -Y_trserie, 0],
                  [0, 0, 0, -Y_trserie, Y_trserie + Y_tr + Y_l5 + Y_g, -Y_g],
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

    max_iter = 100

    while epsilon > 1e-6 and k < max_iter:

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
                    
                    J22[i, j] = Q[i] / abs(V[i]) - \
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

    #print("sol =" , x_new)
    #print("error =", epsilon)
    #print("iterations =", k)

    if k == max_iter:
        print("NO SOLUTION !!!")

    #  Now we can get the P ad Q values of the slack node 6 (note that the slack node is the last one of the vectors)

    p_wslack = np.zeros(nbus)
    for i in range(nbus):
            for j in range(nbus):
                p_wslack[i] = p_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.cos(
                    angle_wslack[i]-angle_wslack[j]) + np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]))
                
    #print("p_final =" , p_wslack)

    q_wslack = np.zeros(nbus)
    for i in range(nbus):
            for j in range(nbus):
                q_wslack[i] = q_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.sin(
                    angle_wslack[i]-angle_wslack[j]) - np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]))
                
    #print("q_final =" , q_wslack)

    # Now we compute the injected currents at each node

    curr_inj = np.zeros(nbus, dtype = 'cfloat')
    for i in range(nbus):
        for j in range(nbus):
            curr_inj[i] = curr_inj[i] + Y_bus[i,j] * cmath.rect(V_wslack[j],angle_wslack[j])

    #print("currents injections =", curr_inj)


    #  When it comes to overcurrents, we are interested in line currents, not node injection currents.
    #  We compute now line currents. Note they are normalized to the power of the plant

    i_21 = abs((cmath.rect(V[0],angles[0]) - cmath.rect(V[1],angles[1])) * Y_trserie) / p_owf
    i_32 = abs((cmath.rect(V[1],angles[1]) - cmath.rect(V[2],angles[2])) * Y_piserie) / p_owf
    i_43 = abs((cmath.rect(V[2],angles[2]) - cmath.rect(V[3],angles[3])) * Y_piserie) / p_owf
    i_54 = abs((cmath.rect(V[3],angles[3]) - cmath.rect(V[4],angles[4])) * Y_trserie) / p_owf

    curr = np.array([i_21, i_32, i_43, i_54])
    #print("currents =", curr)

    #  i_g5 = abs((V_wslack[5] - V[4]) * Y_trserie)

    
    


    #  We compute the AC power losses

    p_lossac = Sbase * (p_owf + p_wslack[5]) * 1e-6  # MW

    #print ("ac_losses =", p_lossac,"MW")

    #  NOW WE CREATE THE COST FUNCTION THAT WILL ALLOWAS TO IMPLEMENT THE MVRSM ALGORITHM

    #  Cable cost

    c_cab = ((A + B * np.exp(C * (np.sqrt(3) * u_i * I_rated) * 1e-8) + D) * (9 * n_cables + 1) * l) / (10 * E)  #  to get  A,B,C,D and I max parameters we will use
                                                                                            #  manufacturer data (110,150,220) S_rcb in MVA

    #  Cost switchgears
    c_gis = (0.0017 * u_i * 1e-3 + 0.0231)  # u_i in kV

    # Cost susbstation
    c_ss = 2.534 + 0.0887 * p_owf * 100  # p-owf in MW

    # Cost power losses
    t_owf = 30  # lie time in years
    c_ey = 100  # eu/MWh, cost of energy lost 
    c_losses = 8760 * t_owf * c_ey * p_lossac # losses in MW , 8760 since 1 year is 8760 h

    # Cost transformers
    c_tr = 0.0427 * (S_rtr * 1e-6)**0.7513  # S_rtr in MVA

    # Cost reactors
    k_on = 0.01049
    k_mid = 0.01576
    k_off = 0.01576
    p_on = 0.8312
    p_mid = 1.244
    p_off = 1.244

    c_r1 = k_off * (abs(Y_l1) * (u_i*1e-3)**2) + p_off
    c_r2 = k_off * (abs(Y_l2) *(u_i*1e-3)**2) + p_off
    c_r3 = k_mid * (abs(Y_l3) * (u_i*1e-3)**2) + p_mid
    c_r4 = k_on * (abs(Y_l4) * (u_i*1e-3)**2) + p_on
    c_r5 = k_on * (abs(Y_l5) * (u_i*1e-3)**2) + p_on
    c_reac = c_r1 + c_r2 + c_r3 + c_r4 + c_r5 #Total cost

    c_total = c_cab + c_gis + c_ss + c_losses + c_tr + c_reac
    #print("total cost =", c_total)

    #  We have to include here the penalizations of voltages, current and reactive power
    #  deviations since the are the inequality constraints. 

    # over or below voltages
    c_vol = 0
    for i in range(nbus-1):
        if V[i] > 1.1 or V[i] < 0.9:
            c_vol = c_vol + abs(V[i] - 1) *1e3

    #  print("overvoltage =", c_vol)

    #overcurrents
    c_curr = 0
    for i in range(4):
        if curr[i] > 1.1:
            c_curr = c_curr + abs(V[i] - 1) * 1e3

    #  print("overcurrent =", c_curr)

    # we want reactive power delivered to the grid to be as close as possible to 0
    c_react = 0
    if q_wslack[nbus-1] != 0:
            c_react = abs(q_wslack[nbus-1]) * 1e2

    #print("reactivetogrid =", c_react)

    costs = np.array([c_total * 1e-8, c_vol, c_curr, c_react])
    #print("costs =", costs)

    return costs



vol = 3  # Corresponds to 220 kV
n_cables = 1
react_bi = np.array([1, 0, 1, 0, 1])
#  react_val = np.array([0, 0.1615, 0.323, 0.1615, 0])


trials = 30
s_rtr = np.zeros(trials)
for i in range(trials):
    s_rtr[i] = random.uniform(450e6,700e6)
# print("s_rtr = ",s_rtr)

react_val = np.zeros((trials,5))
for i in range(trials):
    react_val[i,:] = random.uniform(0,0.5)
# print("react =", react_val)

results = np.zeros((trials, 4))
for i in range(trials):
    pow_tr = s_rtr[i]
    react_valpres = react_val[i]
    y = costac( vol , n_cables , react_bi , react_valpres , pow_tr)
    results[i,:] = y

total = results.sum(axis = 1)

print(results)
print ("sum =", total)

#  Now we should study how we normalized the economic costs and technical deviations
#  to be able to asses correctly the optimization
   