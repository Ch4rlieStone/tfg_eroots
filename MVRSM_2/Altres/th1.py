import numpy as np

# data
# try with B = 1.0 and then with B = 0.01
P = 2.0
B = 1.0
R = 0.05
X = 0.2
Vsl = 1.0 + 1j * 0.0

# Calc th
Voc = Vsl * (1 / (-1j * B)) / (R + 1j * X + 1 / (-1j * B))
Voc_re = np.real(Voc)
Voc_im = np.imag(Voc)

Zsc = (R + 1j * X) * (1 / (-1j * B)) / (R + 1j * X + 1 / (-1j * B))
Ysc = 1 / Zsc
Gsc = np.real(Ysc)
Bsc = np.imag(Ysc)

# Solve

def f1(vre, vim):
    f = (vre**2 + vim**2) * (Gsc + 1j * Bsc) - np.conj(Voc) * (vre + 1j * vim) * (Gsc + 1j * Bsc) - P 
    return np.real(f)


def f2(vre, vim):
    f = (vre**2 + vim**2) * (Gsc + 1j * Bsc) - np.conj(Voc) * (vre + 1j * vim) * (Gsc + 1j * Bsc) - P 
    return np.imag(f)

    
delta = 1e-6
n_max = 10
vre = 1.0
vim = 0.0
for k in range(n_max):
    # f = - J Ax
    df1_dvre = (f1(vre + delta, vim) - f1(vre, vim)) / delta
    df1_dvim = (f1(vre, vim + delta) - f1(vre, vim)) / delta
    df2_dvre = (f2(vre + delta, vim) - f2(vre, vim)) / delta
    df2_dvim = (f2(vre, vim + delta) - f2(vre, vim)) / delta
    J = np.array([[df1_dvre, df1_dvim], [df2_dvre, df2_dvim]])

    ff1 = f1(vre, vim)
    ff2 = f2(vre, vim)
    f = np.array([ff1, ff2])

    Ax = -np.linalg.inv(J) @ f

    vre += Ax[0]
    vim += Ax[1]


print(f'Voltage = {vre + 1j * vim}')
ii = ((vre + 1j * vim) - Voc) / Zsc
S1 = (vre + 1j * vim) * np.conj(ii)
S2 = Voc * np.conj(ii)
Sloss = S1 - S2
print(f'P loss = {np.real(Sloss)}')
print(f'Q loss = {np.imag(Sloss)}')
