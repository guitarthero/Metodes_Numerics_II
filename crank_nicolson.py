import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

#Visualització amb LaTeX dels gràfics
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"})

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


#Paràmetres de discretització
n=101   # Nombre de punts mallat espacial
t_final = 0.025 #normalitzat
Dx = 1/(n-1) #normalitzat
#gamma = Dt/(Dx^2)
gamma =0.49
Dt = gamma * Dx**2
m = int(t_final/Dt)  # Nombre d'iteracions temporals

print('Paràmetres de discretització:')
print('Nombre de punts mallat espacial n:', n)
print('Pas temporal Dt:', Dt*t0, 's')
print('Pas espacial Dx:', Dx)
print('Nombre d\'iteracions temporals m:', m)
print('--------------------')
# Definició de matrius usades en els càlculs
I = np.identity(n)
M = 2*I + (-1)*np.diag(np.ones(n-1),1) + (-1)*np.diag(np.ones(n-1),-1)
#Modifiquem la 1a i última fila per incorporar les condicions de contorn
M[0,0:3] = [-1,2,-1]
M[-1,-3:] = [-1,2,-1]
A1 = I + gamma/2 * M
A2 = I - gamma/2 * M

#Inicialitzem la temperatura t=0 a T=T_in a tota posicio
T_i = T_c * np.ones(n)

#Esquema d'iteració Crank-Nicolson:
# A1*T^{temps+Dt} = A2*T^{temps} + Dt * (1,...,1) = B^{temps}
sol_T = np.zeros((m,n))
#Inicialitzem la temperatura t=0 a T=T_in a tota posicio
sol_T[0,:] = T_i.copy()
for temps in range(1,m):
    B = A2 @ T_i + Dt * np.ones(n)
    # Implementem el mètode de Gauss-Seidel. Sistema lineal A1*T^{temps+Dt} = B^{temps}
    # Fixem temps constant: A1*T' = B
    for iteration in range(100):
        T_iterat = np.zeros(n)
        T_iterat[0] = (1/A1[0,0])*(B[0] - np.dot(A1[0,1:],T_i[1:]))
        for index in range(1,n):
            T_iterat[index] = (1/A1[index,index])*(B[index] - np.dot(A1[index,:index],T_iterat[:index])-
                                            np.dot(A1[index,index+1:],T_i[index+1:]))
        error = (np.abs(T_i-T_iterat)).max()
        T_i = T_iterat.copy()
        # Aproximem l'error a partir de l'estabilitat de la solució
        if error<1e-6: #Precisió desitjada
            break
    T_i[0] = T_c  #Condició de contorn
    T_i[-1] = T_c  #Condició de contorn
    sol_T[temps,:] = T_i.copy()

plt.plot(2*np.linspace(0,1,n), sol_T[0,:]*Temp0-273.15, label='0')
plt.plot(2*np.linspace(0,1,n), sol_T[m//4,:]*Temp0-273.15, label='$t_a/4$')
plt.plot(2*np.linspace(0,1,n), sol_T[m//2,:]*Temp0-273.15, label='$t_a/2$')
plt.plot(2*np.linspace(0,1,n), sol_T[3*m//4,:]*Temp0-273.15, label='$3t_a/4$')
plt.plot(2*np.linspace(0,1,n), sol_T[-1,:]*Temp0-273.15, label='$t_a$')
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.vlines(1-0.25, ymin=T_c*Temp0-273.15, ymax=55, colors='black', linestyles='-.')
plt.vlines(1+0.25, ymin=T_c*Temp0-273.15, ymax=55, colors='black', linestyles='-.')
plt.hlines(50, xmin=0, xmax=2, colors='red', linestyles='dashed')
#Detalls estètics del gràfic
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10)
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig("temperature_distribution.png",dpi=300)
