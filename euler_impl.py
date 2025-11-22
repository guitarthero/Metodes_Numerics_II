import numpy as np
import matplotlib.pyplot as plt

#implementem Euler implícit, hem d'anar amb cuidado, doncs hem de resoldre per tots el temps
#cada iteració del mètode resoldrem per T^{n+1}+T^{n-1}
#les condicions inicials es que T_0 = T_ambient
#les condicions de contorn son T(0,t) = T_ambient i T(L,t) = T_ambien

#M i N depenen de la discretització, cada vector pot ser un temps donat
#fem un vector de vectors, on cada vector serà un temps donat, i en aquest vector tinderem totes les posicions
M = 3
N = 2

#implementem euler 

#hi ha diferents formes de crear matrius, fem amb python
#n es la dimensió de la matriu, element es lo que posem a la diagonal
#poso valors de prova, cal posar tots els paràmetres i normalitzar dsprés
element = 1
n = 5
DeltaX = 1
DeltaT = 1
T_0 = 36

vector_de_vectors = np.zeros((N, M))
# Asignar el valor constante a toda la primera fila
vector_de_vectors[0, :] = T_0
#N es num de vecs i M dimensió
#sembla que haurem de fer una matriu 3D, per guardar cadascuna de les iteracions


gamma = DeltaX / (DeltaT**2)
matriu_identitat = np.eye(n)
matriu_diagonal = np.eye(n) * element
matriu_sup = -np.eye(n, k=1)
matriu_inf = -np.eye(n, k=-1)
print("Usando np.eye:")
print(matriu_inf)
M = matriu_diagonal + matriu_inf + matriu_sup

A = -gamma * M

#ara definim el vector b per euler impl
b_vec = (A + matriu_identitat)* T_0
