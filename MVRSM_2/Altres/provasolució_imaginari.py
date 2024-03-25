# AQUI CALCULEM LES ADMITANCIES DE LA GRID#
import sympy
from sympy import sqrt, symbols, Eq, solve, I
import math as math
import cmath

U_i= 220e3
P_Cu = 60e3
P_Fe = 40e3
u_k = 0.18
i_o = 0.012

G_tr=(P_Fe/U_i**2)/(1/484)
print(G_tr)












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