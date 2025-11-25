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
N = 101
t_a = 0.025
t_0 = c_v *dens * l_0**2 / kappa
T_norm = V_ef**2 * sigma /kappa
#implementem euler 

#hi ha diferents formes de crear matrius, fem amb python
#n es la dimensió de la matriu, element es lo que posem a la diagonal
#poso valors de prova, cal posar tots els paràmetres i normalitzar dsprés
element = 1
n = 5
DeltaX = 1/(N-1)
DeltaT_1 = DeltaX
DeltaT_2 = 0.5*DeltaX
T_c = 36
T_0 = T_c / T_norm
#
temperature_field = np.zeros((N, N))



gamma_1 = DeltaX / (DeltaT_1**2)
gamma_2 = DeltaX / (DeltaT_2**2)
matriu_identitat = np.eye(n)
matriu_diagonal = np.eye(n) * element
matriu_sup = -np.eye(n, k=1)
matriu_inf = -np.eye(n, k=-1)
print("Usando np.eye:")
print(matriu_inf)
M = matriu_diagonal + matriu_inf + matriu_sup

A_1 = -gamma_1 * M
A_2 = -gamma_2 * M

#ara definim el vector b per euler impl
b_vec = (A + matriu_identitat)* T_0
temp_discr_1 = np.zeros(N,N)
temp_discr_2 = np.zeros(N,N)
for i in range(1, N-1):
    #imposem condicions inicials + contorn
    temperature_field[:,0] = T_0
    temperature_field[0,:] = T_0
    temperature_field[N, :] = T_0
    for j in range(1,5000):
        if i == 1:
            temperature_field[:,i] = (gamma*M + matriu_identitat)* temperature_field[:,0] + DeltaT * np.ones(N)
        else:
            for k in range(1, N-1):
                b_vec[i] = (-gamma * M + matriu_identitat)* temperature_field[:,k-1] + 2 * DeltaT * np.ones(N)
                temperature_field[i,k] = (1/A[i,i])*(b_vec[i]-np.sum(A[i,:]*temperature_field))
