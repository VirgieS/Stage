#librairies
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

s0 = np.linspace(1.1, 9.9, 100) #parameter s0 for 1<s0<10

#beta function
def beta_func(s):
    return np.sqrt(1-1/s)

#phi function
def integrand(s):
    beta = beta_func(s)
    return s*(1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

n=len(s0)
phis = np.zeros_like(s0)
for i in range (0,n):
    phis[i] = quad(integrand, 1, s0[i])[0]

#Graph
plt.plot(s0, phis/(s0-1))
plt.xlabel('parameter s0')
plt.ylabel('phi/(s0-1)')
plt.show()
