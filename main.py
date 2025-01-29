import numpy as np
from math import exp
import matplotlib.pyplot as plt
import pandas as pd

N = 30


def compressibility(gas):
    p_c = gases_df[gas][0]  # critical pressure [Pa]
    p_pr = p / p_c  # pseudo-reduced pressure

    T_c = gases_df[gas][1]  # critical temperature [deg R]
    T_pr = T / T_c  # pseudo-reduced temperature

    z = 1 - 3.53 * p_pr / (10 ** (0.9813 * T_pr)) + 0.274 * p_pr ** 2 / (10 ** (0.8157 * T_pr))
    return z


def density(gas_name):
    M = gases_df[gas_name][2]
    z = compressibility(gas_name)
    R = 8.3145  # gas constant [J/mol·K]

    rho_g_range = p * M / (z * R * T * 5 / 9)  # CH4 density [kg/m3]
    return rho_g_range


def viscosity(gas_name):
    M = gases_df[gas_name][2]
    rho_g_range = density(gas_name)

    # viscosity calculation
    K = (9.4 + 20 * M) * T ** 1.5 / (209 + 1.9 * 10e4 * M + T)
    X = 3.5 + (986 / T) + 10 * M
    Y = 2.4 - 0.2 * X

    mu_g_range = []
    for n in range(N):
        rho_g = rho_g_range[n]
        # print(f"[{n}] density: {rho_g}, pressure: {p[n] / 10e6} MPa")
        # Lee-Gonzalez-Eakin equation for viscosity of natural gases
        mu_g = 10 ** (-4) * K * exp(X * (rho_g / 1000) ** Y)  # viscosity [cP]
        mu_g_range.append(mu_g)
    return mu_g_range


data = {
    "methane (CH4)": [4.604 * 10e7, 190.58 * 9 / 5, 0.016043, 'chocolate'],
    "ethane (C2H6)": [4.880 * 10e7, 305.42 * 9 / 5, 0.030069, 'gold'],
    "propane (C3H8)": [4.250 * 10e7, 369.82 * 9 / 5, 0.044097, 'darkkhaki'],
    "butane (C4H10)": [3.796 * 10e7, 425.16 * 9 / 5, 0.05812, 'olivedrab']
}

gases_df = pd.DataFrame(data)

p = np.linspace(10e5, 10e7, N)  # pressure [Pa]
T = 590  # temperature [deg R]

for (gas_name, values) in gases_df.items():
    plt.plot(p/10e6, viscosity(gas_name), label=gas_name, color=gases_df[gas_name][3])
plt.title("Viscosity of natural gases")
plt.xlabel("pressure [MPa]")
plt.ylabel("viscosity [cP]")
plt.ylim(0, 0.05)
plt.legend(loc='upper left')
plt.gca().set_facecolor('#f2f2f2')
plt.grid(color='white', linewidth=1)
plt.show()
