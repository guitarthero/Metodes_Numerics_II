import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
#Visualització amb LaTeX dels gràfics
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"})
#Solució analítica fins al i-èsim terme
def T(z,t,i,T0):
    for n in range(0,i+1):
        T0+=(4/(np.pi*(2*n+1)))*((1-np.exp(-((2*n+1)**2)*(np.pi**2)*t))/(((2*n+1)**2)*(np.pi**2)))*np.sin(np.pi*z*(2*n+1))
    return T0
#Funció que implementa el mètode de Gauss-Seidel: AT_i = B
def gauss_seidel(A, B, T_i, max_iter=1000, tolerancia=1e-10):
    n = len(B)
    for iteration in range(max_iter):
        T_i1 = np.zeros(n)
        T_i1[0] = (1/A[0,0])*(B[0] - np.dot(A[0,1:],T_i[1:]))
        for index in range(1,n):
            T_i1[index] = (1/A[index,index])*(B[index] - np.dot(A[index,:index],T_i1[:index])-
                                            np.dot(A[index,index+1:],T_i[index+1:]))
        error= la.norm(T_i-T_i1)
        if error<tolerancia:
            break
        T_i = T_i1.copy()
    return T_i

#Paràmetres físics del problema
c_v = 3686  # J/(kg·K)
rho = 1081  # kg/m³
k = 0.56  # W/(m·K)
sigma = 0.472  # S/m
alpha = k / (rho * c_v)  # m²/s
Vef = 40/np.sqrt(2)  # V
l0 = 0.02  # m
t0 = l0**2 / alpha  # s
Temp0 = Vef**2*sigma/k  # K

#Normalització de les condicions inicials i de contorn
#EDP normalitzada:  T_t = T_xx + 1
T_c_nonorm = 273.15 +36.5  # K
T_c = T_c_nonorm/Temp0
print('--------------------')
print('PARÀMETRES FÍSICS DEL PROBLEMA:')
print('Difusivitat termica alpha:', alpha, 'm2/s')
print('Temps caracteristic t0:', t0, 's')
print('Temperatura caracteristica Temp0:', Temp0 - 273.15, 'ºC')
print('Distància característica l0:', l0, 'm')
print('Condició de contorn T_c:', T_c_nonorm, 'K')
print('--------------------')

#Paràmetres de discretització fixats en tot el programa
n=101   # Nombre de punts mallat espacial
t_a = 0.025 #normalitzat

# Definició de matrius usades en els càlculs
I = np.identity(n)
M = 2*I + (-1)*np.diag(np.ones(n-1),1) + (-1)*np.diag(np.ones(n-1),-1)
# Modifiquem la 1a i última fila per incorporar les condicions de contorn
M[0,0:3] = [0,0,0]
M[-1,-3:] = [0,0,0]
Q = np.concatenate([np.array([0]),np.ones(n-2),np.array([0])])

# Inicialitzem la temperatura t=0 a T=T_in a tota posicio
T_inicial = T_c * np.ones(n)

def analitica(gamma, n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    #gamma = Dt/(Dx^2)
    gamma =0.49
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals

    sol_T_an = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            sol_T_an[i,j] = T(j*Dx, i*Dt, 100, T_c)
    return sol_T_an
def explicit(gamma, T_i = T_inicial, n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    #gamma = Dt/(Dx^2)
    gamma =0.49
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals

    E2 = I - gamma * M
    # Esquema d'iteració Euler explícit:
    # T^{temps + Dt} = E2 * T^{temps} + Dt * (0,1,...,1,0) = B^{temps}
    sol_T_exp = np.zeros((m,n))
    sol_T_exp[0,:] = T_i.copy()
    for temps in range(1,m):
        T_i = E2 @ T_i + Dt * Q
        sol_T_exp[temps,:] = T_i.copy()
    return sol_T_exp
def implicit(gamma, T_i = T_inicial,  n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    #gamma = Dt/(Dx^2)
    gamma =0.49
    Dt = gamma * Dx**2
    m = int(t_a/Dt)  # Nombre d'iteracions temporals

    E1 = I + gamma * M
    # Esquem d'iteració Euler implícit:
    # E1*T^{temps + Dt} = T^{temps - Dt} + Dt * (0,1,...,1,0) = B^{temps}
    sol_T_imp = np.zeros((m,n))
    sol_T_imp[0,:] = T_i.copy()
    for temps in range(1,m):
        # Sistema lineal E1*T^{temps+Dt} = B^{temps} --> E1*T' = B_imp
        B_imp = T_i + Dt * Q
        T_i = gauss_seidel(E1, B_imp, T_i)
        sol_T_imp[temps,:] = T_i.copy()
    return sol_T_imp
def cranc(gamma, T_i = T_inicial, n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    #gamma = Dt/(Dx^2)
    gamma =0.49
    Dt = gamma * Dx**2
    m = int(t_a/Dt)  # Nombre d'iteracions temporals

    N1 = I + gamma/2 * M
    N2 = I - gamma/2 * M
    # Esquema d'iteració Crank-Nicolson:
    # N1 * T^{temps + Dt} = N2 * T^{temps} + Dt * (0,1,...,1,0) = B^{temps}
    sol_T_cn = np.zeros((m,n))
    sol_T_cn[0,:] = T_i.copy()
    for temps in range(1,m):
        # Sistema lineal N1*T^{temps+Dt} = B^{temps} --> N1*T' = B_exp
        B_cn = N2 @ T_i + Dt * Q
        T_i = gauss_seidel(N1, B_cn, T_i)
        sol_T_cn[temps,:] = T_i.copy()
    return sol_T_cn
sol_T_an = analitica(0.49)
sol_T_cn = cranc(0.49)
sol_T_exp = explicit(0.49)
sol_T_imp = implicit(0.49)

plt.plot(2*np.linspace(0,1,n), sol_T_an[-1,:]*Temp0-273.15, label='Analítica', linewidth = 1,color='r')
plt.plot(2*np.linspace(0,1,n), sol_T_cn[-1,:]*Temp0-273.15, label='Cranc-Nicolson', linewidth = 1,color='g')
plt.plot(2*np.linspace(0,1,n), sol_T_exp[-1,:]*Temp0-273.15, label='Euler explícit', linewidth = 1,color='blue')
plt.plot(2*np.linspace(0,1,n), sol_T_imp[-1,:]*Temp0-273.15, label='Euler implícit', linewidth = 1,color='orange')
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.vlines(1-0.25, ymin=T_c*Temp0-273.15, ymax=55, colors='black', linestyles='-.')
plt.vlines(1+0.25, ymin=T_c*Temp0-273.15, ymax=55, colors='black', linestyles='-.')
plt.hlines(50, xmin=0, xmax=2, colors='red', linestyles='dashed')
#Detalls estètics del gràfic
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig("temperature_distribution.png",dpi=300)
plt.show()