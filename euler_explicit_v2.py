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
from matplotlib.animation import FuncAnimation

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
    return T_c + (4/np.pi**3) * s

# Bucle per cada r
for r in r_valors:
    dt = r * dx**2
    n_passos = int(t_a / dt)

    # Matriu sol numerica: T[i][j] -> i: esapi, j: temps
    T = T_c*np.ones((N, n_passos+1))  # inicial: T(x,0)=T_c
    # Matriu sol analitica: A[i][j] -> i: esapi, j: temps
    A = np.zeros((N, n_passos+1))

    # Omplim la solució analítica
    for j in range(n_passos+1):
        for i in range(N):
            A[i,j] = f(i*dx, j*dt)

    # Euler explícit
    for j in range(n_passos):
        for i in range(1, N-1):
            T[i,j+1] = T[i,j] + dt*((T[i+1,j] - 2*T[i,j] + T[i-1,j])/dx**2 + 1)

        # condicions de frontera adimensionals
        T[0,j+1] = T_c
        T[-1,j+1] = T_c

    # Càlcul de l'error (normalitzat)
    error = np.zeros(n_passos+1)
    for j in range(n_passos+1):
        error[j] = np.max(np.abs(T[:,j] - A[:,j]))

    # Error màxim (normalitzat)
    err_max_hat = np.max(error)

    # Convertim a Kelvin: error_K = err_hat * T_0
    err_max_K = err_max_hat * T_0

    # Convertim a ºC (les temperatures reals estan en Kelvin): ºC = K - 273.15
    err_max_degC = err_max_K  # és un error en Kelvin; per interpretació d'increment relatiu no cal restar 273.15

    print(f"Error màxim (normalitzat) per r={r}: {err_max_hat:.6e}")
    print(f"Error màxim en K per r={r}: {err_max_K:.6e} K")
    print(f"Error màxim en ºC per r={r}: {err_max_K:.6e} °C")


        # ---- PREPARAR DADES REALS PER ANIMACIÓ ----
    # Convertim la matriu T (normalitzada) a temperatura real en Kelvin:0
    T_real_K = T * T_0
    # I a graus Celsius per visualitzar:
    T_real_degC = T_real_K - 273.15

    # Límits per l'eix Y en ºC (una mica de marge)
    ymin = np.min(T_real_degC)
    ymax = np.max(T_real_degC)
    y_margin = 0.05 * (ymax - ymin) if (ymax - ymin) != 0 else 1.0
    ymin -= y_margin
    ymax += y_margin

    # ANIMACIÓ (mostrem tots els passos; ajusta interval per a la velocitat desitjada)
    x = np.linspace(0, 1, N)
    fig, ax = plt.subplots(figsize=(8,5))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0,1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$x$ (normalitzat)")
    ax.set_ylabel(r"Temperatura (°C)")
    ax.set_title(f"Evolució temporal (r = {r}) — Temperatura real")

    def init():
        line.set_data([],[])
        return line,

    def update(frame):
        # frame és l'índex temporal j
        line.set_data(x, T_real_degC[:, frame])
        ax.set_title(f"Evolució temporal (r = {r}), pas = {frame}/{n_passos}")
        return line,

    # Si vols una animació més lenta, augmenta 'interval' (ms)
    anim = FuncAnimation(fig, update, frames=range(0, n_passos+1),
                        init_func=init, interval=80, blit=True)

    plt.show()

