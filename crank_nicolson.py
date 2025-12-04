import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
#Visualització amb LaTeX dels gràfics
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"})
#Solució analítica fins al i-èsim terme de la sèrie de Fourier
def T(z,t,T0,i=200):
    for n in range(0,i+1):
        T0+=(4/(np.pi*(2*n+1)))*((1-np.exp(-((2*n+1)**2)*(np.pi**2)*t))/(((2*n+1)**2)*(np.pi**2)))*np.sin(np.pi*z*(2*n+1))
    return T0

#Funció que implementa el mètode de Gauss-Seidel: AT_i = B
#Definim una tolerància de 1e-10 per defecte
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
Dx = 1/(n-1)

#Zona malalta
x_1 = 0.5 - 0.25/2
x_2 = 0.5 + 0.25/2
n_mal_1 = int(x_1/Dx)+1
n_mal_2 = int(x_2/Dx)-1
T_crit = (50+273.15)/Temp0
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
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals
    sol_T_an = np.zeros(n)
    t_final_aprox=m*Dt
    for j in range(0,n):
        sol_T_an[j] = T(j*Dx, t_final_aprox, T_c)
    return sol_T_an
def explicit(gamma, T_i = T_inicial, n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals
    E2 = I - gamma * M
    # Esquema d'iteració Euler explícit:
    # T^{temps + Dt} = E2 * T^{temps} + Dt * (0,1,...,1,0) = B^{temps}
    a,b=0,0
    for temps in range(1,m+1):
        T_i = E2 @ T_i + Dt * Q
        if a == 0:        
            if np.max(T_i[n_mal_1:n_mal_2+1]) > T_crit:
                print('La regió malalta supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'a E.E')
                a+=1
                c=temps
        if b == 0:        
            if np.max(T_i[0:n_mal_1]) > T_crit:
                print('La regió sana supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'a E.E')
                b+=1
                d = temps
    print('Duració', (temps-c)*Dt*t0,'s')
    return T_i
#Definirem ara r = Dt/Dx ----> gamma = r/Dx
def implicit(gamma, T_i = T_inicial,  n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals
    E1 = I + gamma * M
    # Esquem d'iteració Euler implícit:
    # E1*T^{temps + Dt} = T^{temps - Dt} + Dt * (0,1,...,1,0) = B^{temps}
    a,b = 0,0
    for temps in range(1,m+1):
        # Sistema lineal E1*T^{temps+Dt} = B^{temps} --> E1*T' = B_imp
        B_imp = T_i + Dt * Q
        T_i = gauss_seidel(E1, B_imp, T_i)
        if a == 0:        
            if np.max(T_i[n_mal_1:n_mal_2+1]) > T_crit:
                print('La regió malalta supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'a E.I')
                a+=1
                c=temps
        if b == 0:        
            if np.max(T_i[0:n_mal_1]) > T_crit:
                print('La regió sana supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'a E.I')
                b+=1
    print('Duració', (temps-c)*Dt*t0,'s')
    return T_i

def cranc(gamma, T_i = T_inicial, n = 101, t_final = t_a):
    Dx = 1/(n-1) #normalitzat
    Dt = gamma * Dx**2
    m = int(t_final/Dt)  # Nombre d'iteracions temporals
    N1 = I + gamma/2 * M
    N2 = I - gamma/2 * M
    # Esquema d'iteració Crank-Nicolson:
    # N1 * T^{temps + Dt} = N2 * T^{temps} + Dt * (0,1,...,1,0) = B^{temps}
    a,b=0,0
    for temps in range(1,m+1):
        # Sistema lineal N1*T^{temps+Dt} = B^{temps} --> N1*T' = B_exp
        B_cn = N2 @ T_i + Dt * Q
        T_i = gauss_seidel(N1, B_cn, T_i)
        if a == 0:        
            if np.max(T_i[n_mal_1:n_mal_2+1]) > T_crit:
                print('La regió malalta supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'a C-N')
                a+=1
                c=temps
        if b == 0:        
            if np.max(T_i[0:n_mal_1]) > T_crit:
                print('La regió sana supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'C-N')
                b+=1
    print('Duració', (temps-c)*Dt*t0,'s')
    return T_i
exacta_0 = np.array([ T(j*Dx,t_a,T_c) for j in range(0,n)])
exacta_1 = analitica(0.51)
exacta_2 = analitica(0.49)
exacta_3 = analitica(0.25)
exacta_4 = analitica(1)
exacta_5 = analitica(0.5)

#PART 1: EULER EXPLICIT
explicit_1 = explicit(0.51)
plt.plot(2*np.linspace(0,1,n), explicit_1*Temp0-273.15, label = '$\gamma=0.51$', color = 'blue',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), exacta_1*Temp0-273.15, label = 'Analítica', color = 'red',linewidth = 1)
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
explicit_3 = explicit(0.25)
plt.plot(2*np.linspace(0,1,n), explicit_2*Temp0-273.15, label = '$\gamma = 0.49$', color = 'green',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), explicit_3*Temp0-273.15, label = '$\gamma = 0.25$', color = 'orange',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), exacta_0*Temp0-273.15, label = 'Analítica', color = 'red')
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig('explicit_49_25.png', dpi = 300)
plt.cla()

e_explicit_1 = np.abs(explicit_1-exacta_1)
e_explicit_2 = np.abs(explicit_2-exacta_2)
e_explicit_3 = np.abs(explicit_3-exacta_3)

plt.plot(2*np.linspace(0,1,n), e_explicit_2*Temp0, label = '$\gamma=0.49$', color = 'green',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), e_explicit_3*Temp0, label = '$\gamma=0.25$', color = 'orange',linewidth = 1)
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Error numèric [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.ylim(bottom=0)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig('error_explicit_49_25.png', dpi = 300)
plt.cla()

#PART 2: EULER IMPLICIT
implicit_1 = implicit(1)
implicit_2 = implicit(0.5)
plt.plot(2*np.linspace(0,1,n), implicit_1*Temp0-273.15, label = '$\gamma = 1$', color = 'green',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), implicit_2*Temp0-273.15, label = '$\gamma = 0.5$', color = 'orange',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), exacta_0*Temp0-273.15, label = 'Analítica', color = 'red',linewidth = 1)
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig('implicit_100_500.png', dpi = 300)
plt.cla()

e_implicit_1 = np.abs(implicit_1-exacta_4)
e_implicit_2 = np.abs(implicit_2-exacta_5)

plt.plot(2*np.linspace(0,1,n), e_implicit_1*Temp0, label = '$\gamma = 1$', color = 'green',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), e_implicit_2*Temp0, label = '$\gamma = 0.5$', color = 'orange',linewidth = 1)
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Error numèric [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.ylim(bottom=0)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig('error_implicit_100_50.png', dpi = 300)
plt.cla()

#PART 3a: CRANC-NICOLSON
cranc_1 = cranc(1)
cranc_2 = cranc(0.5)
plt.plot(2*np.linspace(0,1,n), cranc_1*Temp0-273.15, label = '$\gamma = 1$', color = 'green',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), cranc_2*Temp0-273.15, label = '$\gamma = 0.5$', color = 'orange',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), exacta_0*Temp0-273.15, label = 'Analítica', color = 'red',linewidth = 1)
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Temperatura [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig('cranc_100_50.png', dpi = 300)
plt.cla()

e_cranc_1 = np.abs(cranc_1-exacta_4)
e_cranc_2 = np.abs(cranc_2-exacta_5)

plt.plot(2*np.linspace(0,1,n), e_cranc_1*Temp0, label = '$\gamma=1$', color = 'green',linewidth = 1)
plt.plot(2*np.linspace(0,1,n), e_cranc_2*Temp0, label = '$\gamma = 0.5$', color = 'orange',linewidth = 1)
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Error numèric [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.ylim(bottom=0)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig('error_cranc_100_50.png', dpi = 300)
plt.cla()

#PART 3b: Comparativa
err_T_exp = np.abs(explicit(0.49)-exacta_2)
err_T_imp = np.abs(implicit(0.49)-exacta_2)
err_T_cn = np.abs(cranc(0.49)-exacta_2)
plt.plot(2*np.linspace(0,1,n), err_T_cn*Temp0, label='Cranc-Nicolson', linewidth = 1,color='orange')
plt.plot(2*np.linspace(0,1,n), err_T_exp*Temp0, label='Euler explícit', linewidth = 1,color='blue')
plt.plot(2*np.linspace(0,1,n), err_T_imp*Temp0, label='Euler implícit', linewidth = 1,color='green')
plt.xlabel('Distància del primer electròde, $x$ [m]',fontsize=13)
plt.ylabel('Error numèric [$^\circ$C]',fontsize=13)
plt.xlim(0,2)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(fontsize=10, loc = 'upper left')
plt.grid(True, linestyle='dotted', color='gray')
plt.tight_layout()
plt.savefig("comparacio_error_49.png",dpi=300)


#Calcul del temps en que el mètode és eficient amb la solució analítica
Dx = 1/(n-1) #normalitzat
gamma = 0.25
Dt = gamma * Dx**2
m = int(t_a/Dt)  # Nombre d'iteracions temporals
sol_T_completa = np.zeros(n)
a,b=0,0
for temps in range(1,m+1):
    for j in range(0,n):
        sol_T_completa[j] = T(j*Dx, temps*Dt, T_c)
    if a == 0:        
        if np.max(sol_T_completa[n_mal_1:n_mal_2+1]) > T_crit:
            print('La regió malalta supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'a ANALÍTICA')
            a+=1
            c=temps
    if b == 0:        
        if np.max(sol_T_completa[0:n_mal_1]) > T_crit:
            print('La regió sana supera els 50ºC a', temps*Dt*t0, 's amb gamma=',gamma, 'ANALÍTICA')
            print('Duració', (temps-c)*Dt*t0,'s')
            break