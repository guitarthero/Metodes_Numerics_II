""""
Nom: euler_explicit.py
Autor: Alex Guitart
Data: 26/11/2025
Programa:
"""
# pylint: disable=invalid-name
# pylint: disable=unused-import, invalid-name
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
N = 101 # punts
x = np.linspace(0, 1, N)
dx = 1 / (N - 1) # inteval dx normalitzat
r_valors = [0.51, 0.49, 0.25] # dt = r_i*(dx)**2

#Normalització de les condicions inicials i de contorn
T_c_nonorm = 273.15 + 36.5  # K
T_c = T_c_nonorm/T_0
t_a = 0.025

# Definim la matriu amb la que treballarem
# T = T_c_nonorm * np.ones(101,101)

# Diccionari per emmagatzemar els resultats finals
resultats = {}

for r in r_valors:
    dt = r*(dx)**2
    n_passos = int(t_a/dt)

    # Matriu T[i][j] -> i: esapi, j: temps
    T = np.zeros((N, n_passos+1))  # inicial: T(x,0)=0

    for j in range(n_passos):
        for i in range(1, N-1): # Nodes interns
            T[i][j+1] = T[i][j] + dt*((T[i+1][j] - 2*T[i][j] + T[i-1][j])/(dx**2) + 1)

        # Condicions de frontera
        T[0, j+1] = 0
        T[-1, j+1] = 0

    resultats[r] = T[:, -1]  # guardem l'ultima columna (temps final)

# Grafiques
plt.figure(figsize=(8,5))
for r, T_final in resultats.items():
    plt.plot(x, T_final, label=f"r = {r}")
plt.xlabel(r'$\hat{x}$')
plt.ylabel(r'$\hat{T}$')
plt.title(r'Solucions Euler explícit a $\hat{t}$ = 0.025')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
