import numpy as np
import matplotlib.pyplot as plt
def T(z,t,i,T0):
    for n in range(0,i+1):
        T0+=(4/(np.pi*(2*n+1)))*((1-np.exp(-((2*n+1)**2)*(np.pi**2)*t))/(((2*n+1)**2)*(np.pi**2)))*np.sin(np.pi*z*(2*n+1))
    return T0
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})
t=np.linspace(0,10,200)
plt.plot(t,T(t,1,5,0))
plt.tick_params(axis='x', direction='in', top=True, bottom=True)
plt.tick_params(axis='y', direction='in', left=True, right=True)
plt.legend(loc='upper left', fontsize="small")
plt.grid(True, linestyle='dotted', color='gray',alpha=0.3)
plt.show()