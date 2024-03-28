import numpy as np
import matplotlib.pyplot as plt


def c_cabs(n_cables):
    # ref 12 
    # Cable costs
    # :param n_cables: number of cables
    # return: cost in M€
    ll = 100  # km
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
    # return (((A + B * np.exp(C * (n_cables * np.sqrt(3) * u_i * I_rated) * 1e-6) + D) * (9 * n_cables + 1) * l) / (10 * E))
    return (((A + B * np.exp(C * (np.sqrt(3) * u_i * I_rated) * 1e-6) + D) * (9 * n_cables + 1) * ll) / (10 * E)) / 1e6

def c_sw(v_nom):
    # ref 15
    # Switchear cost
    # :param v_nom: nominal voltage in kV
    # return: cost in M€
    return 0.0117 * v_nom + 0.0231

def c_tr(s_tr):
    # ref 16
    # Transformer cost
    # :param s_tr: power in MVA
    # return: cost in M€
    return 0.0427 * s_tr**0.7513

def c_qq(q_ac):
    # ref 17
    # Reactive power compensation cost
    # :param q_ac: reactive power in Mvar
    # return: cost in M€
    K = 0.01049
    P = 0.8312
    return K * q_ac + P

def c_ss(p_ow):
    # ref 18
    # Substation cost
    # :param p_ow: wind power in MW
    # return: cost in M€
    return 2.534 + 0.0887 * p_ow

def c_loss(P_loss):
    # Loss cost
    # :param P_loss: power loss in MW
    # return: cost in M€
    return (8760 * 25 * 100 * P_loss) / 1e6

if __name__ == "__main__":
    n_cab = 2
    v_nom = 220
    s_tra = 750
    q_ac = 80
    p_ow = 500
    p_loss = 2

    c_cab = c_cabs(n_cab)
    c_swi = c_sw(v_nom)
    c_tra = c_tr(s_tra)
    c_qac = c_qq(q_ac)
    c_sub = c_ss(p_ow)
    c_los = c_loss(p_loss)

    print(f"Cost of cables: {c_cab} M€", sep="\n")
    print(f"Cost of switchgear: {c_swi} M€", sep="\n")
    print(f"Cost of transformer: {c_tra} M€", sep="\n")
    print(f"Cost of reactive power compensation: {c_qac} M€", sep="\n")
    print(f"Cost of substation: {c_sub} M€", sep="\n")
    print(f"Cost of losses: {c_los} M€", sep="\n")

