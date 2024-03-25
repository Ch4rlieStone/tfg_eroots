
import numpy as np
import math
import cmath
epsilon = 1
k = 0
Y_ref = 1/484 # (using Sref = 100 MVA and Vref = 220 kV)
# ADD HERE ADITANCE MATRIX FORMULATION TAKING INTO ACCOUNT THE NORMALIZATION TO P.U. #
"""
#Transformer
U_rtr= 220e3
P_Cu = 60e3
P_Fe = 40e3
u_k = 0.18
i_o = 0.012
S_rtr = 500e6

G_tr=(P_Fe/U_rtr**2)/Y_ref
B_tr = -(i_o*(S_rtr/U_rtr**2)) /Y_ref
# print(B_tr)
Y_pi = 4 + 2j
G_pi = 4/Y_ref
B_pi = 2/Y_ref

Y_l = 0 -4j
G_l = 0
B_l = -4/Y_ref

Y_g = 4.95 -0.49j
G_g = 4.95/Y_ref
B_g = -0.49/Y_ref
"""
#Y_ref = 5/484
#Z_ref = 484

U_rtr= 220e3
P_Cu = 60e3
P_Fe = 40e3
u_k = 0.18
i_o = 0.012
S_rtr = 500e6
#TRAFO
# Y_transformer
G_tri = (P_Fe/U_rtr**2)
B_tri = -(i_o*(S_rtr/U_rtr**2))
Y_tr = complex(G_tri, B_tri)/Y_ref
G_tr = Y_tr.real
B_tr = Y_tr.imag
print("Y_tr =",Y_tr)
#print("B_tr =",B_tr)

# Z_transformer
"""
R_tr = P_Cu*(U_rtr/S_rtr)**2
print(R_tr)
X_tr = math.sqrt((u_k*((U_rtr)**2/S_rtr))**2 - R_tr)
print(X_tr)

"""
R_tr = P_Cu/S_rtr
#print("R_tr =",R_tr)
X_tr = math.sqrt(u_k**2 - R_tr**2)
#print("X_tr =",X_tr)
Z_tr = complex(R_tr,X_tr)
#Y_trserie = 1/Z_tr
Y_trserie = (1/Z_tr)/Y_ref
print("Y_trserie= ",Y_trserie)
G_trserie = Y_trserie.real
B_trserie = Y_trserie.imag
#print("B_trserie =",B_trserie )
# CABLES 
# we will consider l= 100000m (100km)
l= 100
R = 0.0072
Cap = 0.17e-6 # F/km
L = 0.40e-3   # H/km
Y = complex(0,2*math.pi*50* Cap/2)
Z = complex(R, 2*math.pi*50* L) #frequency of 50 Hz
theta = l/2*cmath.sqrt(Z*Y)
# print(theta)
Y_pi = (Y*l/4*cmath.tanh(theta/2)/(theta/2))/Y_ref
print("Y_pi =",Y_pi)
G_pi = Y_pi.real
B_pi = Y_pi.imag
Z_piserie = (Z*l/2*cmath.sinh(theta)/theta)
Y_piserie = (1/Z_piserie)/Y_ref
#print("Z_pi =",Z_pi)
print ("Y_piserie =", Y_piserie)
G_piserie =  Y_piserie.real
B_piserie =  Y_piserie.imag
# COMPENSATORS Y_l (has to have similar value as cable parallel part)

Y_l= complex(0,-B_pi)
G_l = Y_l.real
B_l = Y_l.imag
print("Y_l =",Y_l)
# GRID

#Y_g = (4.95 - 0.49j)/Y_ref
Y_g = (4.95 - 0.49j)
print("Y_g = ", Y_g)
#Y_gnormal = Y_g/Y_ref  WRONG
G_g = Y_g.real
B_g = Y_g.imag

Y_bus = np.array([[Y_trserie + Y_tr + Y_l,-Y_trserie,0,0,0,0],\
                [-Y_trserie,Y_piserie + Y_pi+Y_l+Y_trserie,-Y_piserie,0,0,0],\
                [0,-Y_piserie,2*Y_piserie +2*Y_pi+Y_l,-Y_piserie,0,0],\
                [0,0,-Y_piserie,Y_piserie+Y_pi+Y_l+Y_trserie,-Y_trserie,0],\
                [0,0,0,-Y_trserie,Y_trserie+Y_tr + Y_l+Y_g, -Y_g],\
                [0,0,0,0,-Y_g,Y_g]])

# print(Y_bus)

B_bus = np.array([[B_trserie + B_tr + B_l,-B_trserie,0,0,0,0],\
                [-B_trserie,B_piserie + B_pi+B_l+B_trserie,-B_piserie,0,0,0],\
                [0,-B_piserie,2*B_piserie +2*B_pi+B_l,-B_piserie,0,0],\
                [0,0,-B_piserie,B_piserie+B_pi+B_l+B_trserie,-B_trserie,0],\
                [0,0,0,-B_trserie,B_trserie+B_tr + B_l+B_g, -B_g],\
                [0,0,0,0,-B_g,B_g]])
G_bus = np.array([[G_trserie + G_tr + G_l,-G_trserie,0,0,0,0],\
                [-G_trserie,G_piserie + G_pi+G_l+G_trserie,-G_piserie,0,0,0],\
                [0,-G_piserie,2*G_piserie +2*G_pi+G_l,-G_piserie,0,0],\
                [0,0,-G_piserie,G_piserie+G_pi+G_l+G_trserie,-G_trserie,0],\
                [0,0,0,-G_trserie,G_trserie+G_tr + G_l+G_g, -G_g],\
                [0,0,0,0,-G_g,G_g]])

"""
G_bus = np.array([[G_trserie + G_tr + G_l,-G_trserie,0,0,0,0],\
                [-G_trserie,G_piserie + G_pi+G_l+G_tr,-G_piserie,0,0,0],\
                [0,-G_piserie,2*G_piserie +2*G_pi+G_l,-G_piserie,0,0],\
                [0,0,-G_piserie,G_piserie+G_pi+G_l+G_tr,-G_trserie,0],\
                [0,0,0,-G_trserie,G_trserie+G_tr + G_l+G_g, -G_g],\
                [0,0,0,0,-G_g,G_g]])
"""
#print(B_bus)
#print(G_bus)
# G_bus = np.array([[2*G_tr + G_l,-G_tr,0,0,0,0],[-G_tr,2*G_pi+G_l+G_tr,-G_pi,0,0,0],[0,-G_pi,2*G_pi+2*G_pi+G_l,-G_pi,0,0],[0,0,G_pi,2*G_pi+G_l+G_tr,-G_tr,0],[0,0,0,-G_tr,2*G_tr + G_l+G_g,-G_g],[0,0,0,0,-G_g,G_g]])




###
p_owf = 5  # (this value is P_owf/100e6  S_base=100e6) In this case, we consider 500 MW owf
q_owf = 0
V_slack = np.ones((1, 1))
V = np.ones((5,1)) 
V_wslack = np.concatenate((V,[[1]]),)
V_present = V
angles = np.zeros((5, 1))
angles_wslack = np.concatenate((angles,[[0]]),)

x0 = np.concatenate((angles,V))
x = x0

P_obj = np.array([[p_owf,0,0,0,0]]).T
Q_obj = np.array([[q_owf,0,0,0,0]]).T
PQ_obj = np.concatenate((P_obj,Q_obj)) # desired value of PQ nodes

while epsilon > 10e-6 and k<100:
    
    k = k + 1
    x = np.concatenate((angles,V))
    P = np.zeros((5,1))
    Q = np.zeros((5,1))
    angles_wslack = np.concatenate((angles,[[0]] ),)
    V_wslack = np.concatenate((V,[[1]] ),)
    V_present = V
    # print(V_slack)
    # print(V_present)

    # Compute power mismatch function
    for i in range(0,5):
        for j in range(0,6):
            P[i,0] = P[i,0] + V_wslack[i,0]* V_wslack[j,0]*(Y_bus[i,j].real*np.cos(angles_wslack[i,0]-angles_wslack[j,0]) + Y_bus[i,j].imag*np.sin(angles_wslack[i,0]-angles_wslack[j,0]))
  
        #P[i,0] = P[i,0]*V_wslack[i,0]

    for i in range(0,5):
        for j in range(0,6):
            Q[i,0] = Q[i,0] + V_wslack[i,0]* V_wslack[j,0]*(Y_bus[i,j].real*np.sin(angles_wslack[i,0]-angles_wslack[j,0]) - Y_bus[i,j].imag*np.cos(angles_wslack[i,0]-angles_wslack[j,0]))
    
        #Q[i,0] = Q[i,0]*V_wslack[i,0]
    
    # compute error in mismatch function
    PQ = np.concatenate((P, Q))
    deltaPQ = (PQ_obj - PQ)
    print("mismatch =",deltaPQ)
    # print("error =", deltaPQ)

    # Now we will build the Jacobian

    J11 = np.zeros((5,5)) # P wrt angle
    J12 = np.zeros((5,5)) # P wrt V
    J21 = np.zeros((5,5)) # Q wrt angle
    J22 = np.zeros((5,5)) # Q wrt V

    for i in range(5):
        for j in range(5):
        #n = k +1
            if j == i:
            #for j in range(5):
                J11[i,j] = -Q[i,0] - (V[i,0]**2)*Y_bus[i,i].imag
                
            
            else:
                J11[i,j] = abs(V[i,0])*abs(V[j,0])*(Y_bus[i,j].real*np.sin(angles[i,0]-angles[j,0]) - Y_bus[i,j].imag*np.cos(angles[i,0]-angles[j,0]))

    # print(J11)

    for i in range(5):
    #m = i +1
        for j in range(5):
        #n = k +1
            if j == i:
            #for j in range(5):
                J12[i,j] = P[i,0]/abs(V[i,0]) + Y_bus[i,i].real*abs(V[i,0])
            else:
                J12[i,j] = abs(V[i,0])*(Y_bus[i,j].real*np.cos(angles[i,0]-angles[j,0]) + Y_bus[i,j].imag*np.sin(angles[i,0]-angles[j,0]))

    for i in range(5):
    #m = i +1
        for j in range(5):
        #n = k +1
            if j == i:
            #for j in range(5):
                J21[i,j] = P[i,0]-(V[i,0])**2*Y_bus[i,i].real
            else:
                J21[i,j] = - abs(V[i,0])*abs(V[j,0])*(Y_bus[i,j].real*np.cos(angles[i,0]-angles[j,0]) + Y_bus[i,j].imag*np.sin(angles[i,0]-angles[j,0]))

    for i in range(5):
    #m = i +1
        for j in range(5):
        #n = k +1
            if j == i:
            #for j in range(5):
                J22[i,j] = Q[i,0]/abs(V[i,0]) - Y_bus[i,i].imag*abs(V[i,0])
            else:
                J22[i,j] = abs(V[i,0])*(Y_bus[i,j].real*np.sin(angles[i,0]-angles[j,0]) - Y_bus[i,j].imag*np.cos(angles[i,0]-angles[j,0]))

 
    J = np.concatenate((np.concatenate((J11, J12), axis=1), np.concatenate((J21, J22), axis=1)), axis=0)
    #print(J)
    #print("determinantJ =", np.linalg.det(J))
    
    # now we have to solve the system
    #print(J)
    #print("determinantJ =", np.linalg.det(J))

    # now we have to solve the system

    # WE WILL TRY JACOBIAN WITH SMALL VARIATIONS OF X VALUES#
    
   
                
     
          




    delta_x =  np.linalg.solve(J, deltaPQ)
    #print("delta_x =",delta_x)
    x_new = x + delta_x # we have updated value of angles and V [10X1] matrix 
    #print("x =",x)
    # (note this vector does not include slack!)
    #print("xnew =",x_new)
    angles = x_new[0:5]
    V = x_new[5:10]

    # we check error value
    epsilon = (max(abs(deltaPQ[:,0])))
    #print("error =", epsilon)

print(x_new)
print(epsilon)
print("iterations =",k)