import numpy as np
import matplotlib.pyplot as plt

#implementem Euler implícit, hem d'anar amb cuidado, doncs hem de resoldre per tots el temps

#paràmetres del problema
c_v = 3686
dens = 1081
kappa = 0.56
sigma = 0.472
l_0 = 0.02
V_ef = 40/np.sqrt(2)


#M i N depenen de la discretització, cada vector pot ser un temps donat
#fem un vector de vectors, on cada vector serà un temps donat, i en aquest vector tinderem totes les posicions

t_a = 0.025 #temps final adimensional
t_0 = c_v *dens * l_0**2 / kappa
T_norm = V_ef**2 * sigma /kappa
#implementem euler 


element = 2
N = 101
DeltaX = 1/(N-1)
DeltaT_1 = DeltaX**2
DeltaT_2 = 0.5*(DeltaX**2)
N_temps_1 = int(t_a / DeltaT_1) + 1  # +1 per incloure t=0
N_temps_2 = int(t_a / DeltaT_2) + 1

T_c = 36.5
T_0 = T_c / T_norm
#
temperature_field = np.zeros((N, N))



gamma_1 = DeltaT_1 / (DeltaX**2)
gamma_2 = DeltaT_2 / (DeltaX**2)
matriu_identitat = np.eye(N)
matriu_diagonal = np.eye(N) * element
matriu_sup = -np.eye(N, k=1)
matriu_inf = -np.eye(N, k=-1)
M = matriu_diagonal + matriu_inf + matriu_sup


A_1 = gamma_1 * M + matriu_identitat
A_2 = gamma_2 * M + matriu_identitat

#definim la funció per fre gauss seidel
def gauss_seidel(A, b, x0, tol=1e-8, max_iter = 1000):
    #resolem un sistema Ax = b amb Gauss-Seidel
    x = x0.copy()
    n = len(b)
    #posem copy, perquè si no modifiquem x0 al modificar x
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            if i==0 or i==n-1:
                x[i] = T_0
            else:
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, i+1:], x_old[i+1:])
                #amb A[i, :i], li diem que agafi la fila i, i les columnes fins a i-1
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

                #mirem si ha convergit amb el criteri de tol
        if np.linalg.norm(x - x_old) < tol:
            break
    return x

#ara, definm una funció per fer euler implícit, on necessitarem aplicar gauss seidel
def euler_implicit(N_temps, gamma, DeltaT, C):
    #condicions inicials
    T_actual = np.ones(N) * T_0
    #creem un vector de vectors per emmagatzemar els valors
    T_history = np.zeros((N_temps, N))
    T_history[0, :] = T_actual

    #creem el vector que suma al final de b i és sempre el mateix
    Q = np.ones(N) * DeltaT
    
    
    
    #ara fem per a passos temporals posteriors
    for n in range(1, N_temps):
        b = (matriu_identitat) @ T_actual +  Q
        T_nou = gauss_seidel(C, b, T_actual)
        #condicionsd de contorn
        T_nou[0] = T_0
        T_nou[-1] = T_0

        #guardem el valor al historial
        T_history[n, :] = T_nou

        #actualitzem els valors per al següent pas
        
        T_actual = T_nou.copy()
    return T_history

T_history_1 = euler_implicit(N_temps_1, gamma_1, DeltaT_1, A_1)
T_history_2 = euler_implicit(N_temps_2, gamma_2, DeltaT_2, A_2)



# Convertir a temperatura física (°C)
T_history_fisica_1 = T_history_1 * T_norm 
T_history_fisica_2 = T_history_2 * T_norm


# Crear eixos espacials (en cm)
x_cm = np.linspace(0, l_0 * 100, N)  # Posició en cm

# Seleccionar 5 temps uniformement distribuïts per a cada cas
def seleccionar_temps_uniformes(T_history, num_temps=5):
    n_total = T_history.shape[0]
    indices = np.linspace(0, n_total-1, num_temps, dtype=int)
    return indices, T_history[indices]

# Seleccionar temps
temps_indices_1, temperatura_data_1 = seleccionar_temps_uniformes(T_history_fisica_1)
temps_indices_2, temperatura_data_2 = seleccionar_temps_uniformes(T_history_fisica_2)

# Calcular temps reals en segons
temps_reals_1 = temps_indices_1 * DeltaT_1 * t_0
temps_reals_2 = temps_indices_2 * DeltaT_2 * t_0

#fem els plots per visualitzar els resultats
# GRÀFICA PER AL CAS 1 (Δt = Δx²)



plt.plot(x_cm, temperatura_data_1[0], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_1[0]:.4f} s')
plt.plot(x_cm, temperatura_data_1[1], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_1[1]:.4f} s')
plt.plot(x_cm, temperatura_data_1[2], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_1[2]:.4f} s')
plt.plot(x_cm, temperatura_data_1[3], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_1[3]:.4f} s')
plt.plot(x_cm, temperatura_data_1[4], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_1[4]:.4f} s')
plt.xlabel('Posició (cm)')
plt.ylabel('Temperatura (°C)')
plt.title('Distribució de T per a diferents temps')
plt.grid(True, alpha=0.3)
plt.ylim(T_c, np.max(temperatura_data_1) + 5)

plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().tick_params(axis='x', which='both', top=True, bottom=True, direction='in', labelsize=12)
plt.gca().tick_params(axis='y', which='both', right=True, left=True, direction='in', labelsize=12)

plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.show()


plt.figure()

plt.plot(x_cm, temperatura_data_2[0], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_2[0]:.4f} s')
plt.plot(x_cm, temperatura_data_2[1], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_2[1]:.4f} s')
plt.plot(x_cm, temperatura_data_2[2], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_2[2]:.4f} s')
plt.plot(x_cm, temperatura_data_2[3], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_2[3]:.4f} s')
plt.plot(x_cm, temperatura_data_2[4], linewidth=2, marker='o', markersize=3, label=f't = {temps_reals_2[4]:.4f} s')
plt.xlabel('Posició (cm)')
plt.ylabel('Temperatura (°C)')
plt.title('Distribució de T per a diferents temps')
plt.grid(True, alpha=0.3)
plt.ylim(T_c, np.max(temperatura_data_2) + 5)

plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().tick_params(axis='x', which='both', top=True, bottom=True, direction='in', labelsize=12)
plt.gca().tick_params(axis='y', which='both', right=True, left=True, direction='in', labelsize=12)

plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.show()