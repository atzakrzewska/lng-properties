import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CONSTANTS
N = 50  # number of pressure points
R = 8.3145  # gas constant [J/mol·K]
T = 510  # temperature [deg R]
T_range = np.linspace(510, 850, N)  # temperature [°R]
p = 1e7  # pressure [Pa]
p_range = np.linspace(1e6, 1e8, N)  # pressure [Pa]

# define gas properties in a dictionary
gases_data = {
    "name": [r"methane ($\mathrm{CH_4}$)", r"ethane ($\mathrm{C_2H_6}$)",
             r"propane ($\mathrm{C_3H_8}$)", r"butane ($\mathrm{C_4H_{10}}$)"],
    "color": ["chocolate", "gold", "darkkhaki", "olivedrab"],  # colors for plotting
    "p_c": [4.604e7, 4.880e7, 4.250e7, 3.796e7],  # critical pressure [Pa]
    "T_c": [190.58 * 9 / 5, 305.42 * 9 / 5, 369.82 * 9 / 5, 425.16 * 9 / 5],  # critical temperature [deg R]
    "M": [0.016043, 0.030069, 0.044097, 0.05812]  # molecular weight [kg/mol]
}

gases_df = pd.DataFrame(gases_data)


def compressibility(gas_props, p, T):
    """Calculate compressibility factor z using Lee-Gonzalez-Eakin equation."""
    p_c = gas_props["p_c"]  # critical pressure [Pa]
    p_pr = p / p_c  # pseudo-reduced pressure

    T_c = gas_props["T_c"]  # critical temperature [deg R]
    T_pr = T / T_c  # pseudo-reduced temperature

    z = 1 - 3.53 * p_pr / (10 ** (0.9813 * T_pr)) + 0.274 * p_pr ** 2 / (10 ** (0.8157 * T_pr))

    # prevent negative or zero compressibility
    z[z <= 0] = 0.001

    return z


def density(gas_props, p, T):
    """Calculate gas density."""
    M = gas_props["M"]
    z = compressibility(gas_props, p, T)

    rho_g = p * M / (z * R * T * 5 / 9)  # density [kg/m3]

    # prevent negative density
    rho_g = np.clip(rho_g, 1e-6, None)

    return rho_g


def viscosity(gas_props, p, T):
    """Calculate gas viscosity using Lee-Gonzalez-Eakin equation."""
    M = gas_props["M"]
    rho_g = density(gas_props, p, T)

    # viscosity calculation
    K = (9.4 + 20 * M) * T ** 1.5 / (209 + 1.9e4 * M + T)
    X = 3.5 + (986 / T) + 10 * M
    Y = 2.4 - 0.2 * X

    # Clamp exponent to prevent overflow
    exponent = np.clip(X * (rho_g / 1000) ** Y, -100, 100)  # Clamp within reasonable limits

    mu_g = 1e-4 * K * np.exp(exponent)
    return mu_g


# create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Viscosity of natural gases", fontsize=16, weight='bold', ha='left', x=0.083, y=0.94)

# shared handles for the legend
labels = []
handles = []

# subplot 1: viscosity vs. pressure
for (_, gas_row) in gases_df.iterrows():
    line, = axes[0].plot(p_range / 1e6, viscosity(gas_row, p_range, T), label=gas_row["name"], color=gas_row["color"])
    handles.append(line)  # Add to shared legend
    labels.append(gas_row["name"])
axes[0].set_title(r"$\mathrm{\mu(p)}$" + f" at a temperature of 10°C", loc='left')
axes[0].set_xlabel("pressure [MPa]")
axes[0].set_ylabel("viscosity [cP]")
axes[0].set_xlim(1, 40)
axes[0].set_ylim(0, 0.05)
axes[0].set_facecolor('#f2f2f2')
axes[0].grid(color='white', linewidth=1)

# subplot 2: viscosity vs. temperature
for (_, gas_row) in gases_df.iterrows():
    axes[1].plot(T_range, viscosity(gas_row, p, T_range), label=gas_row["name"], color=gas_row["color"])
axes[1].set_title(r"$\mathrm{\mu(T)}$" + f" at a pressure of 10 MPa", loc='left')
axes[1].set_xlabel("temperature [°R]")
axes[1].set_ylabel("viscosity [cP]")
axes[1].set_xlim(510, 850)
axes[1].set_ylim(0, 0.05)
axes[1].set_facecolor('#f2f2f2')
axes[1].grid(color='white', linewidth=1)

fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)

plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.show()
