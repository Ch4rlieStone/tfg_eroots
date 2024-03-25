import numpy as np
import math

p_owf = 1
q_owf = 0
Y_tr = 0.3 -0.2j
G_tr = 0.3
B_tr = 0.2
# print(B_tr)
Y_pi = 4 + 2j
G_pi = 0.4
B_pi = 0.2

Y_l = 0 -3j
G_l = 0
B_l = -0.4

Y_g = 3 -0.5j
G_g = 0.3
B_g = -0.2
# DATA
V = np.array([[1,1,1,1,1]]).T
V_ws = np.array([[1,1,1,1,1,1]]).T
V_slack = np.array([1]).T
#V = V.T
angles = np.array([[0,0,0,0,0]]).T # intial vector where angles=0 and V=1
angles_ws = np.array([[0,0,0,0,0,0]]).T
angles_slack = np.array([1]).T
x0_slack = np.concatenate((angles_slack,V_slack))
#angles = angles.T
#print(x0)
# print(len(x0))
P = np.array([[p_owf,0,0,0,0]])
P = P.T
Q = np.array([[q_owf,0,0,0,0]])
Q = Q.T
PQ = np.array([[p_owf,0,0,0,0,q_owf,0,0,0,0]])
PQ = PQ.T
#V_i = x0[0:5]
Y_bus = np.array([[2*Y_tr + Y_l,-Y_tr,0,0,0],[-Y_tr,2*Y_pi+Y_l+Y_tr,-Y_pi,0,0],[0,-Y_pi,2*Y_pi+2*Y_pi+Y_l,-Y_pi,0],[0,0,Y_pi,2*Y_pi+Y_l+Y_tr,-Y_tr],[0,0,0,-Y_tr,2*Y_tr + Y_l+Y_g]])
#B_bus = np.array([[2*B_tr + B_l,-B_tr,0,0,0],[-B_tr,2*B_pi+B_l+B_tr,-B_pi,0,0],[0,-B_pi,2*B_pi+2*B_pi+B_l,-B_pi,0],[0,0,B_pi,2*B_pi+B_l+B_tr,-B_tr],[0,0,0,-B_tr,2*B_tr + B_l+B_g]])
#G_bus = np.array([[2*G_tr + G_l,-G_tr,0,0,0],[-G_tr,2*G_pi+G_l+G_tr,-G_pi,0,0],[0,-G_pi,2*G_pi+2*G_pi+G_l,-G_pi,0],[0,0,G_pi,2*G_pi+G_l+G_tr,-G_tr],[0,0,0,-G_tr,2*G_tr + G_l+G_g]])

B_bus = np.array([[2*B_tr + B_l,-B_tr,0,0,0,0],[-B_tr,2*B_pi+B_l+B_tr,-B_pi,0,0,0],[0,-B_pi,2*B_pi+2*B_pi+B_l,-B_pi,0,0],[0,0,B_pi,2*B_pi+B_l+B_tr,-B_tr,0],[0,0,0,-B_tr,2*B_tr + B_l+B_g, -B_g],[0,0,0,0,-B_g,B_g]])
G_bus = np.array([[2*G_tr + G_l,-G_tr,0,0,0,0],[-G_tr,2*G_pi+G_l+G_tr,-G_pi,0,0,0],[0,-G_pi,2*G_pi+2*G_pi+G_l,-G_pi,0,0],[0,0,G_pi,2*G_pi+G_l+G_tr,-G_tr,0],[0,0,0,-G_tr,2*G_tr + G_l+G_g,-G_g],[0,0,0,0,-G_g,G_g]])

# P_i= V_i*np.real(Y_bus)-P
#print(V_i)
#print(P)
#print(Y_bus)
#print(P_i)
#print(B_bus)
##print(G_bus)
#print(PQ)
#print(Y_bus[0,0])
# x=x0
"""
def powermismatch (x,Y_bus):
    f = np.empty([10,1])
    for i in len(x0)/2: #5 the P's
        for j in len(x0)/2: #5
            f[i] = f[i] + x[i]*x[j]*(Y_bus[i,j].real*math.cos(x[i]-x[j]) + Y_bus[i,j].imag*math.sin(x[i]-x[j]))
        f[i] = PQ[i]-f[i]
    for i in range(len(x0)/2,len(x0)): #from 5 to 10 the Q's
        for j in len(x0)/2:
            f[i] = f[i] + x[i]*x[j]*(Y_bus[i,j].real*math.sin(x[i]-x[j]) - Y_bus[i,j].imag*math.cos(x[i]-x[j]))
        f[i]= PQ[i] - f[i]
    return f
"""
# COMPUTES POWER MISMATCH FUNCTION

Qx = np.array([[0],[0],[0],[0],[0]])
Px = np.array([[0],[0],[0],[0],[0]])
for i in range(0,5):
    for j in range(0,6): #5
        Px[i] = Px[i] + V_ws[j]*(G_bus[i,j]*math.cos(angles_ws[i]-angles_ws[j]) + B_bus[i,j]*math.sin(angles_ws[i]-angles_ws[j]))
        Qx[i] = Qx[i] + V_ws[j]*(G_bus[i,j]*math.sin(angles_ws[i]-angles_ws[j]) - B_bus[i,j]*math.cos(angles_ws[i]-angles_ws[j]))
    
    Px[i] = V[i]*Px[i]
    Qx[i] = V[i]*Qx[i]

print(Px)
PQx = np.concatenate((Px, Qx))
deltaPQ = PQ - PQx
x0 = np.concatenate((angles,V))
print(x0)        

           
    
# deltaQ[i] +=  V[j]*(G_bus[i,j]*math.sin(angles[i]-angles[j]) - B_bus[i,j]*math.cos(angles[i]-angles[j]))
    
print(deltaPQ)    
""""""     
#print(deltaP)
#print(deltaQ)

# WE CREATE NOW THE JACOBIAN MATRIX
k = 0
# J1 is dervative of P with respect to angles
"""
J1 = np.zeros((5,5))
for i in range(5-1):
    m = i +1
    for j in range(5-1):
        n = k +1
        if n == m:
            for n in range(5):
                J1[i,j] += V[i]*V[j]*(-G_bus[m,n]*np.sin(angles[m]-angles[n]) + B_bus[m,n]*np.cos(angles[m]-angles[n]))
            J1[i,j] += -V[m]**2*B_bus[m,m]
        else:
            J1[i,j] = V[m]*V[n]*(G_bus[m,n]*np.sin(angles[m]-angles[n]) - B_bus[m,n]*np.cos(angles[m]-angles[n]))
            
print(J1)
"""
J11 = np.zeros((5,5)) # P wrt angle
J12 = np.zeros((5,5)) # P wrt V
J21 = np.zeros((5,5)) # Q wrt angle
J22 = np.zeros((5,5)) # Q wrt V

# J11
for i in range(5):
    #m = i +1
    for j in range(5):
        #n = k +1
        if j == i:
            #for j in range(5):
            J11[i,j] = -Qx[i]-V[i]**2*B_bus[i,i]
                #J11[i,j] += V[i]*V[j]*(-G_bus[i,j]*np.sin(angles[i]-angles[j]) + B_bus[i,j]*np.cos(angles[i]-angles[j]))
            
        else:
            J11[i,j] = V[i]*V[j]*(G_bus[i,j]*np.sin(angles[i]-angles[j]) - B_bus[i,j]*np.cos(angles[i]-angles[j]))

print(J11)

# J12
for i in range(5):
    #m = i +1
    for j in range(5):
        #n = k +1
        if j == i:
            #for j in range(5):
            J12[i,j] = Px[i]/V[i] + G_bus[i,i]*V[i]
        else:
            J12[i,j] = V[i]*V[j]*(G_bus[i,j]*np.cos(angles[i]-angles[j]) + B_bus[i,j]*np.sin(angles[i]-angles[j]))

print(J12)

# J21
for i in range(5):
    #m = i +1
    for j in range(5):
        #n = k +1
        if j == i:
            #for j in range(5):
            J21[i,j] = Px[i]-V[i]**2*G_bus[i,i]
        else:
            J21[i,j] = -V[i]*V[j]*(G_bus[i,j]*np.cos(angles[i]-angles[j]) + B_bus[i,j]*np.sin(angles[i]-angles[j]))

print(J21)

# J22

for i in range(5):
    #m = i +1
    for j in range(5):
        #n = k +1
        if j == i:
            #for j in range(5):
            J22[i,j] = Qx[i]/V[i] - B_bus[i,i]*V[i]
        else:
            J22[i,j] = V[i]*V[j]*(G_bus[i,j]*np.sin(angles[i]-angles[j]) - B_bus[i,j]*np.cos(angles[i]-angles[j]))

print(J22)

J = np.concatenate((np.concatenate((J11, J12), axis=1), np.concatenate((J21, J22), axis=1)))
print(J)

# here we have to use the while to iterate the loop with [angles,V]_new
# J([angles,V])*deltaX = deltaPQ([angles,V]) (solve as a linear sistem)   
# [angles,V]_new = [angles,V] + deltaX

print(np.linalg.det(J)) # tenim algo malament pq em surt det=0 (matriu singular) i no pot ser: SOLUCIONAT

tol = 10e-3
iter= 0
x =x0
# while deltaPQ > tol:
delta = np.linalg.solve(J,deltaPQ)

x_new = x + delta #mirar com pillar nms PQ nodes, borrar slack !!!
 # print(delta)
iter = iter +1
print(x_new)
# print(x)
    # Now we starte the process again and we compute deltaPQ with x_new



# SEMBLA QUE POT FUNCIONAR ARA, PEE TANT INCORPOREM EL CODE DINS DEL WHILE LOOP ( LA MATRIU J I MISMATCH SHAN DEVALUAR A X_NEW)