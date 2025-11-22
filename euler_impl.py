import numpy as np
import matplotlib.pyplot as plt

#implementem Euler implícit, hem d'anar amb cuidado, doncs hem de resoldre per tots el temps
#cada iteració del mètode resoldrem per T^{n+1}+T^{n-1}
#les condicions inicials es que T_0 = T_ambient
#les condicions de contorn son T(0,t) = T_ambient i T(L,t) = T_ambien

#fem un vector de vectors, on cada vector serà un temps donat, i en aquest vector tinderem totes les posicions
M = 3
N = 2

# Método 3: Array 2D con valores específicos
vectr_de_vectors = np.zeros((N, M))
#N es num de vecs i M dimensió

print(vectr_de_vectors)
#implementem euler 

