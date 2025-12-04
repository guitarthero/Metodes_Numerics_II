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
dx = 1 / (N - 1) # inteval dx normalitzat
r_valors = [0.51, 0.49, 0.25] # dt = r_i*(dx)**2

#Normalització de les condicions inicials i de contorn
T_c_nonorm = 273.15 + 36.5  # K
T_c = T_c_nonorm/T_0
t_a = 0.025 # temps final normalitzat

# Solució analítica
def f(x, t):
    s = 0
    for n in range(100):
        k = 2*n + 1
        s += (1 - np.exp(-k*k*np.pi*np.pi*t)) * np.sin(k*np.pi*x) / k**3
    return (4/np.pi**3) * s


# Diccionari per emmagatzemar els resultats finals
resultats = {}

for r in r_valors:
    dt = r*(dx)**2
    n_passos = int(t_a/dt)

    # Matriu sol numerica: T[i][j] -> i: esapi, j: temps
    T = T_c*np.zeros((N, n_passos+1))  # inicial: T(x,0)=T_c
    # Matriu sol analitica: A[i][j] -> i: esapi, j: temps
    A = T_c*np.zeros((N, n_passos+1))

    for j in range(n_passos):
        for i in range(1, N-1): # Nodes interns
            T[i][j+1] = T[i][j] + dt*((T[i+1][j] - 2*T[i][j] + T[i-1][j])/(dx**2) + 1)

            A[i][j+1] = f(i,j)

        # Condicions de frontera
        T[0, j+1] = T_c
        T[-1, j+1] = T_c

    resultats[r] = T[:, -1]  # guardem l'ultima columna (temps final)


# Grafiques
x = np.linspace(0, 1, N)
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


from matplotlib.animation import FuncAnimation

# Elegimos un r estable para animar
r_anim = 0.25
dt_anim = r_anim * dx**2
n_passos_anim = int(t_a / dt_anim)

# Calculamos toda la matriz T_anim
T_anim = np.zeros((N, n_passos_anim+1))

for j in range(n_passos_anim):
    for i in range(1, N-1):
        lap = (T_anim[i+1, j] - 2*T_anim[i, j] + T_anim[i-1, j]) / dx**2
        T_anim[i, j+1] = T_anim[i, j] + dt_anim * (lap + 1)
    T_anim[0, j+1] = 0
    T_anim[-1, j+1] = 0

# --- Animació ---
fig, ax = plt.subplots(figsize=(8,5))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(np.min(T_anim), np.max(T_anim))
ax.set_xlabel(r'$\hat{x}$')
ax.set_ylabel(r'$\hat{T}$')
ax.set_title("Evolució temporal de la temperatura")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, T_anim[:, frame])
    return line,

frames = range(0, n_passos_anim, 3)  # salta frames
anim = FuncAnimation(fig, update, frames=n_passos_anim, init_func=init, interval=1, blit = True)
plt.show()
