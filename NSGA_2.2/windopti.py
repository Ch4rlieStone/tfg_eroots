from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
import numpy as np
import cmath
import pymoo.gradient.toolbox as anp
import time
class MixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "react1_bi": Binary(),
            "react2_bi": Binary(),
            "react3_bi": Binary(),
            "react4_bi": Binary(),
            "react5_bi": Binary(),
            #"vol_level": Choice(options=["vol132","vol220"]),
            "vol_level": Choice(options=["vol220"]),
            #"vol_level": Choice(options=["vol132"]),
            "n_cables": Integer(bounds=(2, 2)),
            "S_rtr": Real(bounds=(657e6, 1000e6)),
            "react1": Real(bounds=(0.0, 1.0)),
            "react2": Real(bounds=(0.0, 1.0)),
            "react3": Real(bounds=(0.0, 1.0)),
            "react4": Real(bounds=(0.0, 1.0)),
            "react5": Real(bounds=(0.0, 1.0)),
        }
        super().__init__(vars=vars, n_obj=2, **kwargs)
        #super().__init__(vars=vars, n_obj=2, n_ieq_constr = 10, **kwargs)
        

    def _evaluate(self, X, out, *args, **kwargs):

        react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, vol, n_cables, S_rtr, react1, react2, react3, react4, react5 = \
        X["react1_bi"], X["react2_bi"], X["react3_bi"], X["react4_bi"], X["react5_bi"], X["vol_level"], X["n_cables"], X["S_rtr"],\
        X["react1"], X["react2"], X["react3"], X["react4"], X["react5"]

        def build_grid_data(Sbase, f, l, p_owf, q_owf, vol, S_rtr, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val, react4_val, react5_val):
            """
            Build the admittance matrix of the grid and define the general data.
            :param Sbase: Base power of the grid
            :param f: Frequency of the grid
            :param l: Length of the cables
            :param p_owf: Active power of the offshore wind farm
            :param q_owf: Reactive power of the offshore wind farm
            :param vol: Voltage level of the grid
            :param S_rtr: Rated power of the transformer
            :param n_cables: Number of cables
            :param react1_bi: Binary value for the first compensator
            :param react2_bi: Binary value for the second compensator
            :param react3_bi: Binary value for the third compensator
            :param react4_bi: Binary value for the fourth compensator
            :param react5_bi: Binary value for the fifth compensator
            :param react1_val: Value of the first compensator
            :param react2_val: Value of the second compensator
            :param react3_val: Value of the third compensator
            :param react4_val: Value of the fourth compensator
            :param react5_val: Value of the fifth compensator
            :return Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie
            """

            if vol == "vol132":
                u_i = 132e3  # V
                R = 0.0067  # ohm/km
                Cap = 0.19e-6  # F/km
                L = 0.38e-3   # H/km
                A = 1.971e6
                B = 0.209e6
                C = 1.66
                I_rated = 500  # A

            if vol == "vol220":
                u_i = 220e3  # V
                R = 0.0067  # ohm/km
                Cap = 0.17e-6  # F/km
                L = 0.40e-3   # H/km
                A = 3.181e6
                B = 0.11e6
                C = 1.16
                I_rated = 540  # A

            Y_ref = Sbase / u_i**2  # 1 / ohm
            V_ref = u_i

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
            # R_tr = P_Cu / S_rtr
            R_tr = P_Cu / (3 * (S_rtr / (np.sqrt(3) * u_i))**2) #new corrected version
            #  R_tr = P_Cu * (U_rtr / S_rtr)**2  new version to try
            X_tr = np.sqrt((u_k * (U_rtr**2 / S_rtr))**2 - R_tr**2)
            #X_tr = np.sqrt(((u_k * u_i / np.sqrt(3) / (S_rtr / (np.sqrt(3) * u_i)))**2 - R_tr**2))
            Z_tr = R_tr + 1j * X_tr
            Y_trserie =  (1 / Z_tr) / Y_ref
            
            # Per unit parameters for OPF
            g_tr = np.real(Y_tr)
            b_tr = np.imag(Y_tr)
            r_tr = np.real(Z_tr * Y_ref)  # multiply by Y_ref to have it in p.u.
            x_tr = np.imag(Z_tr * Y_ref)

            # 1.3 Cables
            R = 0.0067  # ohm/km
            Cap = 0.17e-6  # F/km
            L = 0.40e-3   # H/km
            Y = 1j * (2 * np.pi * f * Cap / 2)
            Z = R + 1j * (2 * np.pi * f * L)
            #theta = l / 2 * np.sqrt(Z * Y)
            theta = l * np.sqrt(Z * Y)
            Y_pi = n_cables * (Y * l / 2 * np.tanh(theta / 2) / (theta / 2)) / Y_ref
            G_pi = np.real(Y_pi)
            B_pi = np.imag(Y_pi)
            Z_piserie = (Z * l  * np.sinh(theta / 1) / (theta / 1)) / n_cables
            #Y_piserie = n_cables * (1 / Z_piserie)  / Y_ref 
            Y_piserie =  (1 / Z_piserie)  / Y_ref

            # Per unit parameters for OPF: also remove the 2
            # b_unit2 = 2 * np.imag(Y_pi)
            b_unit = np.imag(Y_pi)
            r_unit = np.real(Z_piserie * Y_ref)
            x_unit = np.imag(Z_piserie * Y_ref)
            '''
            # Cables new
            R = 0.0067  # ohm/km
            Cap = 0.17e-6  # F/km
            L = 0.40e-3   # H/km
            Y = 1j * (2 * np.pi * f * Cap / 2)
            Z = R + 1j * (2 * np.pi * f * L)
            theta = l * np.sqrt(Z * Y)
            Y_pi = (n_cables * Y * (l / 2) * np.tanh(theta / 2) / (theta / 2)) / Y_ref
            G_pi = np.real(Y_pi)
            B_pi = np.imag(Y_pi)
            Z_piserie = (Z * l * np.sinh(theta / 1) / (theta / 1)) 
            Y_piserie = n_cables * (1 / Z_piserie)  / Y_ref 
            '''
            # 1.4 Compensator
            if react1_bi:
                Y_l1 =  - 1j * react1_val
            else:
                Y_l1 = 0
                react1_bi = 0.0

            if react2_bi:
                Y_l2 =  - 1j * react2_val
                
            else:
                Y_l2 = 0
                react2_bi = 0.0

            if react3_bi:
                Y_l3 =  - 1j * react3_val
            else:
                Y_l3 = 0
                react3_bi = 0.0

            if react4_bi:
                Y_l4 =  - 1j * react4_val
            else:
                Y_l4 = 0
                react1_bi = 0.0

            if react5_bi:
                Y_l5 =  - 1j * react5_val
            else:
                Y_l5 = 0
                react5_bi = 0.0

            # 1.5 Grid connection
            scr = 50  # which value should we put here 5 or 50?
            xrr = 10
            V_grid = 220e3  # V
            #zgridm = V_ref**2 / (scr * p_owf * Sbase)
            zgridm = V_grid**2 / (scr * p_owf * Sbase)
            rgrid = np.sqrt(zgridm**2 / (xrr**2 + 1))
            xgrid = xrr * rgrid
            zgrid = rgrid + 1j * xgrid
            Y_g = (1 / zgrid) / Y_ref

            Y_bus = np.array([[Y_trserie + Y_tr + Y_l1, -Y_trserie, 0, 0, 0, 0],
                        [-Y_trserie, Y_piserie + Y_pi + Y_l2 + Y_trserie, - Y_piserie, 0, 0, 0],
                        [0, -Y_piserie, 2 * Y_piserie + 2 * Y_pi + Y_l3, -Y_piserie, 0, 0],
                        [0, 0, -Y_piserie, Y_piserie + Y_pi + Y_l4 + Y_trserie, -Y_trserie, 0],
                        [0, 0, 0, -Y_trserie, Y_trserie + Y_tr + Y_l5 + Y_g, -Y_g],
                        [0, 0, 0, 0, -Y_g, Y_g]])
        
            return Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie, Y_ref

        def run_pf(p_owf: float=0.0,
               q_owf: float=0.0,
               Y_bus: np.ndarray=None,
               nbus: int=1,
               V_slack: float=1.0,
               angle_slack: float=0.0,
               max_iter: int=20,
               eps: float=1e-6,
               y_trserie: float=0.0,
               y_piserie: float=0.0,
               S_rtr = S_rtr,
               n_cables = n_cables,
               vol = vol):

            start_time = time.time()
            """
            Run the power flow algorithm to find the voltages and angles of the nodes in the grid.
            :param p_owf: Active power of the offshore wind farm
            :param q_owf: Reactive power of the offshore wind farm
            :param Y_bus: Admittance matrix of the grid
            :param nbus: Number of buses in the grid
            :param V_slack: Voltage of the slack bus
            :param angle_slack: Angle of the slack bus
            :param max_iter: Maximum number of iterations
            :param eps: Error tolerance
            :return: [V, V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found]
            """
    
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

            epsilon = 1e10
            k = 0

            while epsilon > eps and k < max_iter:

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

                # we have updated value of angles and V [1X10] matrix (note this vector does not include slack!)
                x_new = x + delta_x  
                
                angles = x_new[0:5]
                V = x_new[5:10]
                angle_wslack[:nbus - 1] = angles
                V_wslack[:nbus - 1] = V

                # we check error value
                epsilon = (max(abs(deltaPQ)))

            solution_found = False
            p_wslack = np.zeros(nbus)
            q_wslack = np.zeros(nbus)
            curr_inj = np.zeros((nbus),dtype = "complex")
            curr = np.zeros(nbus - 2)
            if k + 1 < max_iter:
                end_time = time.time()
                #print(f"Time taken to compute power flow: {end_time - start_time} seconds")                
                solution_found = True

                Iinj = Y_bus @ (V_wslack * np.exp(1j * angle_wslack))
                Sinj = (V_wslack * np.exp(1j * angle_wslack)) * np.conj(Iinj)
                #  Now we can get the P ad Q values of the slack node 6 (note that the slack node is the last one of the vectors)
                for i in range(nbus):
                        for j in range(nbus):
                            p_wslack[i] = p_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.cos(
                                angle_wslack[i]-angle_wslack[j]) + np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]))
                            
                for i in range(nbus):
                        for j in range(nbus):
                            q_wslack[i] = q_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.sin(
                                angle_wslack[i]-angle_wslack[j]) - np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]))
                            

                # Now we compute the injected currents at each node
                for i in range(nbus):
                    for j in range(nbus):
                        curr_inj[i] += Y_bus[i,j] * cmath.rect(V_wslack[j],angle_wslack[j])

                #  When it comes to overcurrents, we are interested in line currents, not node injection currents.
                #  We compute now line currents. Note they are normalized to the power of the plant
                #i_21 = abs((cmath.rect(V[0],angles[0]) - cmath.rect(V[1],angles[1])) * y_trserie) / p_owf
                #i_32 = abs((cmath.rect(V[1],angles[1]) - cmath.rect(V[2],angles[2])) * y_piserie) / p_owf
                #i_43 = abs((cmath.rect(V[2],angles[2]) - cmath.rect(V[3],angles[3])) * y_piserie) / p_owf
                #i_54 = abs((cmath.rect(V[3],angles[3]) - cmath.rect(V[4],angles[4])) * y_trserie) / p_owf

                i_21 = abs((cmath.rect(V[0],angles[0]) - cmath.rect(V[1],angles[1])) * y_trserie) 
                i_32 = abs((cmath.rect(V[1],angles[1]) - cmath.rect(V[2],angles[2])) * y_piserie) 
                i_43 = abs((cmath.rect(V[2],angles[2]) - cmath.rect(V[3],angles[3])) * y_piserie) 
                i_54 = abs((cmath.rect(V[3],angles[3]) - cmath.rect(V[4],angles[4])) * y_trserie) 
                curr = np.array([i_21, i_32, i_43, i_54])

            else:
                print("No solution found")
                pass

            return V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found
        
        def compute_costs(p_owf: float=0.0, p_wslack: float=0.0, q_wslack: float=0.0, V: np.ndarray=None,
                      curr: np.ndarray=None, nbus: int=1, n_cables: int=1, u_i: float=220, I_rated: float=1.0,
                      S_rtr: float=500, react1_bi: bool=False, react2_bi: bool=False, react3_bi: bool=False,
                      react4_bi: bool=False, react5_bi: bool=False, Y_l1: float=0.0, Y_l2: float=0.0, Y_l3: float=0.0,
                      Y_l4: bool=False, Y_l5: bool=False, Y_ref: float=0.0, solution_found: bool=False):
            """
            Compute all the costs
            :param p_owf: Active power of the offshore wind farm
            :param p_wslack: Active power at each node
            :param q_wslack: Reactive power at each node
            :param V: Voltage at each node
            :param curr: Current at each line
            :param nbus: Number of buses in the grid
            :param n_cables: Number of cables
            :param u_i: Rated voltage of the cables
            :param I_rated: Rated current of the cables
            :param S_rtr: Rated power of the transformer
            :param react1_bi: Binary value for the first compensator
            :param react2_bi: Binary value for the second compensator
            :param react3_bi: Binary value for the third compensator
            :param react4_bi: Binary value for the fourth compensator
            :param react5_bi: Binary value for the fifth compensator
            :param Y_l1: Value of the first compensator
            :param Y_l2: Value of the second compensator
            :param Y_l3: Value of the third compensator
            :param Y_l4: Value of the fourth compensator
            :param Y_l5: Value of the fifth compensator
            :param solution_found: Boolean value indicating if a solution was found
            :return: [cost_invest, cost_tech]
            """

            """
            if not solution_found:
                print("NO SOLUTION !!!")
                cost_invest = 1e20
                cost_tech = 1e20
                cost_tech1 = 1e20 
                cost_tech2 = 1e20
                cost_tech3 = 1e20
                cost_tech4 = 1e20

            else:
            """
        #  We compute the AC power losses
            p_lossac = Sbase * (p_owf + p_wslack[5]) * 1e-6  # MW

            #  Cable cost
            Sncab = np.sqrt(3) * u_i * I_rated
            eur_sek = 0.087  # 0.087 eur = 1 sek
            c_cab = n_cables * (A + B * np.exp(C * Sncab / 1e8)) * l * eur_sek / 1e6

            #  Cost switchgears
            c_gis = (0.0017 * u_i * 1e-3 + 0.0231)  # u_i in kV

            # Cost susbstation
            c_ss = (2.534 + 0.0887 * p_owf * 100)  # p-owf in MW

            # Cost power losses
            t_owf = 25  # lie time in years
            c_ey = 100  # eu/MWh, cost of energy lost 
            c_losses = (8760 * t_owf * c_ey * p_lossac) / 1e6 # losses in MW , 8760 since 1 year is 8760 h

            # Cost transformers
            c_tr = (0.0427 * (S_rtr * 1e-6)**0.7513)  # S_rtr in MVA

            # Cost reactors
            fact = 1e3
            #fact = 1e4 #  orginal facotr JOSEP
            k_on = 0.01049 * fact
            k_mid = 0.01576 * fact
            k_off = 0.01576 * fact
            p_on = 0.8312 
            p_mid = 12.44
            p_off = 1.244

        
            if react1_bi:
                # c_r1 = k_off * (abs(Y_l1) * (V[0])**2) + p_off
                c_r1 = k_off * (abs(Y_l1) * Y_ref * (V[0])**2) + p_off
            else:
                c_r1 = 0

            if react2_bi:
                c_r2 = k_off * (abs(Y_l2) * Y_ref * (V[1])**2) + p_off
            else:
                c_r2 = 0

            if react3_bi:
                c_r3 = k_mid * (abs(Y_l3) * Y_ref * (V[2])**2) + p_mid
            else:
                c_r3 = 0

            if react4_bi:
                c_r4 = k_on * (abs(Y_l4) * Y_ref * (V[3])**2) + p_on
            else:
                c_r4 = 0

            if react5_bi:
                c_r5 = k_on * (abs(Y_l5) * Y_ref * (V[4])**2) + p_on
            else:
                c_r5 = 0
            
            # c_reac = c_r1 + c_r2 + c_r3 + c_r4 + c_r5
            c_reac = (c_r1 + c_r2 + c_r3 + c_r4 + c_r5) * 1

            # we want reactive power delivered to the grid to be as close as possible to 0
            penalty = 100
            c_react = 0
            if q_wslack[nbus-1] != 0:
                    c_react = abs(q_wslack[nbus-1]) * penalty
            
            
            # over or below voltages
            c_vol = 0
            for i in range(nbus-1):
                # c_vol += (abs(V[i] - 1) * 100)
                if V[i] > 1.1:
                    c_vol += (V[i] - 1.1) * penalty
                elif V[i] < 0.9:
                    c_vol += (0.9 - V[i]) * penalty
            """
            # overcurrents
            c_curr = 0
            for i in [1, 2]:  # check only the cable for now
                c_curr += (max(curr[i] - 1.1 * n_cables, 0)) * 100
            """
            # over current
            # g1_oc = curr[0] - 1.1 * n_cables
            i_max_tr = S_rtr / Sbase # rated current of the transformer
            # transformers
            c_curr = 0
            if abs(curr[0]) > 1.1 * i_max_tr:
                c_curr += (abs(curr[0]) - i_max_tr) * penalty
           
            
            if abs(curr[3]) > 1.1 * i_max_tr:
                c_curr += (abs(curr[3]) - i_max_tr) * penalty
            


            i_maxcb =  (Sncab / Sbase) * n_cables
            if abs(curr[1]) > 1.1 * i_maxcb:
                c_curr += (abs(curr[1]) - i_maxcb) * penalty
            
            if abs(curr[2]) > 1.1 * i_maxcb:
                c_curr += (abs(curr[2]) - i_maxcb) * penalty
            

            #g3_oc = abs(curr[1]) - i_maxcb
            #g4_oc = abs(curr[2]) - i_maxcb
            
            #c_curr = (g1_octr + g2_octr + g3_oc + g4_oc) * 100
            # we try to implement the constraints in pymoo form
            # overvoltages
            g1_vol = (1 / 1.1) * (V[0] - 1.1)
            g2_vol = (1 / 1.1) * (V[1] - 1.1)
            g3_vol = (1 / 1.1) * (V[2] - 1.1)
            g4_vol = (1 / 1.1) * (V[3] - 1.1)
            g5_vol = (1 / 1.1) * (V[4] - 1.1)
            # under voltages
            g6_vol = (1 / 0.9) * (0.9 - V[0])
            g7_vol = (1 / 0.9) * (0.9 - V[1])
            g8_vol = (1 / 0.9) * (0.9 - V[2])
            g9_vol = (1 / 0.9) * (0.9 - V[3])
            g10_vol = (1 / 0.9) * (0.9 - V[4])

            gs = [g1_vol, g2_vol, g3_vol, g4_vol, g5_vol, g6_vol, g7_vol, g8_vol, g9_vol, g10_vol]
            #gs = [g1_vol, g2_vol, g3_vol, g4_vol, g5_vol, g6_vol, g7_vol, g8_vol, g9_vol, g10_vol, g1_octr, g2_octr, g3_oc, g4_oc]
            # overcurrents
            
            
            
            
            #cost_invest = c_cab + c_gis + c_tr + c_reac + c_ss
            cost_tech = c_vol + c_curr + c_react + c_losses
            cost_invest = c_cab + c_gis + c_tr + c_reac + c_ss 
            #cost_tech = c_vol + c_curr + c_react + p_lossac*1e6
            cost_tech1 = c_vol
            cost_tech2 = c_curr
            cost_tech3 = c_react
            cost_tech4 = c_losses
            """

            cost_invest = c_cab + c_gis + c_tr + c_reac + c_ss
            cost_tech = c_curr + c_react + c_losses
            cost_tech2 = c_curr
            cost_tech3 = c_react
            cost_tech4 = c_losses
            """

            # cost_full = [c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest]
                # pprint(cost_full)

            # return np.array([cost_invest, cost_tech1, cost_tech2, cost_tech3, cost_tech4])
            
            #return np.array([cost_invest, cost_tech])
            return cost_invest, cost_tech, gs, g1_vol
            #  return cost_invest, cost_tech, gs



        # Main data
        nbus = 6
        vslack = 1.0
        dslack = 0.0
        max_iter = 20
        epss = 1e-6

        Sbase = 100e6  # VA
        f = 50  # Hz
        l = 100  #  distance to shore in km
        p_owf = 5  # p.u,  5 is equivalent to 500 MW owf
        q_owf = 0 # p.u, we assume no reactive power is generated at plant

        Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, y_trserie, y_piserie, Y_ref = build_grid_data(Sbase, f, l, p_owf, q_owf, vol, S_rtr, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1, react2, react3, react4, react5)

        V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found = run_pf(p_owf, q_owf, Y_bus, nbus, vslack, dslack, max_iter, epss, y_trserie, y_piserie)

        cost_invest, cost_tech, gs, g1_vol  = compute_costs(p_owf, p_wslack, q_wslack, V_wslack, curr, nbus, n_cables, u_i, I_rated, S_rtr, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, Y_ref, solution_found) 
        # print(cost_output)
        
        #out["G"] = anp.column_stack([gs[0],gs[1],gs[2],gs[3],gs[4],gs[5],gs[6],gs[7],gs[8],gs[9]])
        #out["G"] = anp.column_stack([gs[0]])
        out["F"] = [cost_invest, cost_tech]
        #return cost_invest, cost_tech
        
