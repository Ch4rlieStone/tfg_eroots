# AQUI CALCULEM LES ADMITANCIES DE LA GRID#
import sympy
from sympy import sqrt, symbols, Eq, solve, I
import math
import cmath

Y_ref = 1/484
Z_ref = 484

U_rtr= 220e3
P_Cu = 60e3
P_Fe = 40e3
u_k = 0.18
i_o = 0.012
S_rtr = 500e6
#TRAFO
# Y_transformer
G_tr = (P_Fe/U_rtr**2)
B_tr = -(i_o*(S_rtr/U_rtr**2)) 
Y_tr = complex(G_tr, B_tr)/Y_ref
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
Y_trserie = 1/Z_tr
Y_trserie = (1/Z_tr)/Y_ref
print("Y_trserie= ",Y_trserie)
G_trserie = Y_trserie.real
B_trserie = Y_trserie.imag
#print("B_trserie =",B_trserie )
# CABLES 
# we will consider l= 100000m (100km)
l= 20
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
Z_pi = (Z*l/2*cmath.sinh(theta)/theta)
Y_piserie = (1/Z_pi)/Y_ref
#print("Z_pi =",Z_pi)
print ("Y_piserie =", Y_piserie)
G_piserie =  Y_piserie.real
B_piserie =  Y_piserie.imag
# COMPENSATORS Y_l (has to have similar value as cable)
G_l = 0
B_l = B_piserie
Y_l= complex(G_l,B_l)
print("Y_l =",Y_l)
# GRID

Y_g = (4.95 - 0.49j)/Y_ref
print("Y_g = ", Y_g)
#Y_gnormal = Y_g/Y_ref  WRONG
G_g = Y_g.real
B_g = Y_g.imag
#print("G_g =",G_g)
#print("B_g =",B_g)














"""
importing from sympy library
from sympy import sqrt, symbols, Eq, solve, I
import math as math

# defining the symbolic variable 'z'
S = symbols('S')
U_i= 220e3
P_Cu = 60e3
P_Fe = 40e3
u_k = 0.18
i_o = 0.012

# setting up the complex equation z^2 + 1 = 0
equation = Eq(P_Cu*(U_i/S)**2 + I*(sqrt((u_k*(U_i/S))**2-(P_Cu*(U_i/S)**2)**2)), 1/(P_Fe/(U_i)**2 - I*(i_o*S/(U_i)**2)))

# solving the equation symbolically to find complex solutions
#solutions = solve(equation, S)
#solabs= abs(solutions)

# printing solutions
#print("Solutions:", solutions)

sum = 0
for i in range(5):
    sum += i
print(sum)
"""