""""
Nom: euler_explicit.py
Autor: Alex Guitart
Data: 26/11/2025
Programa:
"""

import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# Visualització amb LaTeX dels gràfics
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"})

# Paràmetres físics del problema
c_v = 3686  # J/(kg·K)
rho = 1081  # kg/m³
kappa = 0.56  # W/(m·K)
sigma = 0.472  # S/m
alpha = kappa / (rho * c_v)  # m²/s
V_ef = 40/np.sqrt(2)  # V
# EDP normalitzada:  T_t = T_xx + 1
# Paràmetres de normalització
l_0 = 0.02  # m
t_0 = l_0**2 / alpha  # s
T_0 = V_ef**2*sigma/kappa  # K
dx = 0.02/101
dt = 0.49*(dx)**2
#Normalització de les condicions inicials i de contorn
T_c_nonorm = 273.15 + 36.5  # K
T_c = T_c_nonorm/T_0
t_a = 0.025


# Definim la matriu amb la que treballarem
T = T_c_nonorm * np.ones(101,101)
T_len = T.shpe[1]
T_bucle = T.copy()

for i in range(1, T_len):
    for j in range(1, T_len):
        T_bucle[i][j] = T[i][j] + dt*((T[i+1][j] - 2*T[i][j] + T[i-1][j])/(dx**2) + 1)
