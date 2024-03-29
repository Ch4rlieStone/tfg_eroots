import numpy as np
import matplotlib.pyplot as plt

def c_cabs(n_cables, vol):
    if vol == 1:
        u_i = 66e3  # V
        R = 0.0067  # ohm/km
        Cap = 0.24e-6  # F/km
        L = 0.36e-3   # H/km
        A = 0.688e6
        B = 0.625e6
        C = 2.05
        I_rated = 470  # A

    if vol == 2:
        u_i = 132e3  # V
        R = 0.0067  # ohm/km
        Cap = 0.19e-6  # F/km
        L = 0.38e-3   # H/km
        A = 1.971e6
        B = 0.209e6
        C = 1.66
        I_rated = 500  # A

    if vol == 3:
        u_i = 220e3  # V
        R = 0.0067  # ohm/km
        Cap = 0.17e-6  # F/km
        L = 0.40e-3   # H/km
        A = 3.181e6
        B = 0.11e6
        C = 1.16
        I_rated = 540  # A

        
    l = 100  # km
    Sncab = np.sqrt(3) * u_i * I_rated
    eur_sek = 0.087  # 0.087 eur = 1 sek
    c_cab = n_cables * (A + B * np.exp(C * Sncab / 1e8)) * l * eur_sek / 1e6
    return c_cab

# def c_cabs1(n_cables):
#     # ref 12 
#     # Cable costs
#     # :param n_cables: number of cables
#     # return: cost in M€
#     ll = 100  # km
#     u_i = 110e3
#     R = 0.0067  # ohm/km
#     Cap = 0.24e-6  # F/km
#     L = 0.36e-3   # H/km
#     A = 1.3295
#     B = 0.417
#     C = 0.01855
#     D = 17e4
#     E = 8.98
#     I_rated= 470
#     return (((A + B * np.exp(C * (np.sqrt(3) * u_i * I_rated) * 1e-6) + D) * (9 * n_cables + 1) * ll) / (10 * E)) / 1e6

# def c_cabs2(n_cables):
#     # ref 12 
#     # Cable costs
#     # :param n_cables: number of cables
#     # return: cost in M€
#     ll = 100  # km
#     u_i = 150e3
#     R = 0.0067  # ohm/km
#     Cap = 0.19e-6  # F/km
#     L = 0.38e-3   # H/km
#     A = 1.971
#     B = 0.209
#     C = 0.0166
#     D = 17e4
#     E = 8.98
#     I_rated = 500
#     return (((A + B * np.exp(C * (np.sqrt(3) * u_i * I_rated) * 1e-6) + D) * (9 * n_cables + 1) * ll) / (10 * E)) / 1e6

# def c_cabs3(n_cables):
#     # ref 12 
#     # Cable costs
#     # :param n_cables: number of cables
#     # return: cost in M€
#     ll = 100  # km
#     u_i = 220e3
#     R = 0.0067  # ohm/km
#     Cap = 0.17e-6  # F/km
#     L = 0.40e-3   # H/km
#     A = 3.181
#     B = 0.11
#     C = 0.0116
#     D = 17e4
#     E = 8.98
#     I_rated = 540  # how we get this value? 
#     return (((A + B * np.exp(C * (np.sqrt(3) * u_i * I_rated) * 1e-6) + D) * (9 * n_cables + 1) * ll) / (10 * E)) / 1e6



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
    n_cab = 1
    v_nom = 220
    s_tra = 750
    q_ac = 80
    p_ow = 500
    p_loss = 2

    # c_cab1 = c_cabs1(n_cab)
    # c_cab2 = c_cabs2(n_cab)
    # c_cab3 = c_cabs3(n_cab)
    c_cab1 = c_cabs(n_cab, 1)
    c_cab2 = c_cabs(n_cab, 2)
    c_cab3 = c_cabs(n_cab, 3)
    c_swi = c_sw(v_nom)
    c_tra = c_tr(s_tra)
    c_qac = c_qq(q_ac)
    c_sub = c_ss(p_ow)
    c_los = c_loss(p_loss)

    print(f"Cost of cables: {c_cab1} M€", sep="\n")
    print(f"Cost of cables: {c_cab2} M€", sep="\n")
    print(f"Cost of cables: {c_cab3} M€", sep="\n")
    print(f"Cost of switchgear: {c_swi} M€", sep="\n")
    print(f"Cost of transformer: {c_tra} M€", sep="\n")
    print(f"Cost of reactive power compensation: {c_qac} M€", sep="\n")
    print(f"Cost of substation: {c_sub} M€", sep="\n")
    print(f"Cost of losses: {c_los} M€", sep="\n")


