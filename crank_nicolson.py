import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
#Visualització amb LaTeX dels gràfics
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"})
#Solució analítica fins al i-èsim terme
def T(z,t,T0,i=100):
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

def analitica(gamma, n = 101, t_final = t_a, complet = False):
    Dx = 1/(n-1) #normalitzat
    #gamma = Dt/(Dx^2)
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals

    sol_T_an = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            sol_T_an[i,j] = T(j*Dx, i*Dt, T_c)
    if complet == True:
        return sol_T_an
    else:
        return sol_T_an[-1,:]
def explicit(gamma, T_i = T_inicial, n = 101, t_final = t_a,complet=False):
    Dx = 1/(n-1) #normalitzat
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals

    E2 = I - gamma * M
    # Esquema d'iteració Euler explícit:
    # T^{temps + Dt} = E2 * T^{temps} + Dt * (0,1,...,1,0) = B^{temps}
    sol_T_exp = np.zeros((m,n))
    sol_T_exp[0,:] = T_i.copy()
    for temps in range(1,m):
        T_i = E2 @ T_i + Dt * Q
        if complet ==True:
            sol_T_exp[temps,:] = T_i.copy()
    if complet == True:
        return sol_T_exp
    else:
        return T_i
def implicit(gamma, T_i = T_inicial,  n = 101, t_final = t_a,complet=False):
    Dx = 1/(n-1) #normalitzat
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
        if complet == True:
            sol_T_imp[temps,:] = T_i.copy()
    if complet == True:
        return sol_T_imp
    else:
        return T_i
def cranc(gamma, T_i = T_inicial, n = 101, t_final = t_a,complet = False):
    Dx = 1/(n-1) #normalitzat
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
        if complet == True:
            sol_T_cn[temps,:] = T_i.copy()
    if complet == True:
        return sol_T_cn
    else:
        return T_i
exacta_1 = analitica(0.51)
exacta_2 = analitica(0.49)
exacta_3 = analitica(0.25)
exacta_4 = analitica(1)
exacta_5 = analitica(0.5)
#PART 1: EULER EXPLICIT
def part_1():
    explicit_1 = explicit(0.51)
    plt.plot(2*np.linspace(0,1,n), explicit_1*Temp0-273.15, label = '0.51', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('explicit_51.png', dpi = 300)
    plt.cla()

    explicit_2 = explicit(0.49)
    plt.plot(2*np.linspace(0,1,n), explicit_2*Temp0-273.15, label = '0.49', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('explicit_49.png', dpi = 300)
    plt.cla()

    explicit_3 = explicit(0.25)
    plt.plot(2*np.linspace(0,1,n), explicit_3*Temp0-273.15, label = '0.25', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('explicit_25.png', dpi = 300)
    plt.cla()

    e_explicit_1 = np.abs(explicit_1-exacta_1)
    e_explicit_2 = np.abs(explicit_2-exacta_2)
    e_explicit_3 = np.abs(explicit_3-exacta_3)

    plt.plot(2*np.linspace(0,1,n), e_explicit_1*Temp0, label = '$\gamma=0.51$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_explicit_51.png', dpi = 300)
    plt.cla()

    plt.plot(2*np.linspace(0,1,n), e_explicit_2*Temp0, label = '$\gamma=0.49$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_explicit_49.png', dpi = 300)
    plt.cla()

    plt.plot(2*np.linspace(0,1,n), e_explicit_3*Temp0, label = '$\gamma=0.25$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_explicit_25.png', dpi = 300)
    plt.cla()

#PART 2: EULER IMPLICIT
def part_2():
    implicit_1 = implicit(1)
    plt.plot(2*np.linspace(0,1,n), implicit_1*Temp0-273.15, label = '$\gamma=1$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('implicit_100.png', dpi = 300)
    plt.cla()

    implicit_2 = implicit(0.5)
    plt.plot(2*np.linspace(0,1,n), implicit_2*Temp0-273.15, label = '$\gamma=0.5$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('implicit_50.png', dpi = 300)
    plt.cla()

    e_implicit_1 = np.abs(implicit_1-exacta_4)
    e_implicit_2 = np.abs(implicit_2-exacta_5)

    plt.plot(2*np.linspace(0,1,n), e_implicit_1*Temp0, label = '$\gamma=1$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_implicit_100.png', dpi = 300)
    plt.cla()

    plt.plot(2*np.linspace(0,1,n), e_implicit_1*Temp0, label = '$\gamma=0.5$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_implicit_50.png', dpi = 300)
    plt.cla()

#PART 3a: CRANC-NICOLSON
def part_3a():
    cranc_1 = cranc(1)
    plt.plot(2*np.linspace(0,1,n), cranc_1*Temp0-273.15, label = '$\gamma=1$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('cranc_100.png', dpi = 300)
    plt.cla()

    cranc_2 = cranc(0.5)
    plt.plot(2*np.linspace(0,1,n), cranc_2*Temp0-273.15, label = '$\gamma=0.5$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('cranc_50.png', dpi = 300)
    plt.cla()

    e_cranc_1 = np.abs(cranc_1-exacta_4)
    e_cranc_2 = np.abs(cranc_2-exacta_5)

    plt.plot(2*np.linspace(0,1,n), e_cranc_1*Temp0, label = '$\gamma=1$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_cranc_100.png', dpi = 300)
    plt.cla()

    plt.plot(2*np.linspace(0,1,n), e_cranc_2*Temp0, label = '$\gamma=0.5$', color = 'blue')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('error_cranc_50.png', dpi = 300)
    plt.cla()

#PART 3b: Comparativa
def part_3b():
    err_T_cn = np.abs(cranc(0.49)-exacta_2)
    err_T_exp = np.abs(explicit(0.49)-exacta_2)
    err_T_imp = np.abs(implicit(0.49)-exacta_2)

    plt.plot(2*np.linspace(0,1,n), err_T_cn*Temp0, label='Cranc-Nicolson', linewidth = 1,color='g')
    plt.plot(2*np.linspace(0,1,n), err_T_exp*Temp0, label='Euler explícit', linewidth = 1,color='blue')
    plt.plot(2*np.linspace(0,1,n), err_T_imp*Temp0, label='Euler implícit', linewidth = 1,color='orange')
    plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
    plt.ylabel('Error numèric [$^\circ$C]',fontsize=13)
    plt.xlim(0,2)
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(fontsize=10, loc = 'upper left')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig("comparacio_error_49.png",dpi=300)
    plt.show()
#EXECUTEU AQUI LA PART DE PROGRAMA QUE US INTERESSI (sinó tarda massa)
part_1()
part_2()
part_3a()
part_3b()