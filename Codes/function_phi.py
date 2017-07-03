#librairies
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, fixed_quad, romberg

s0 = np.linspace(1.1, 9.9, 100) #parameter s0 for 1<s0<10
print(s0)
beta0 = np.sqrt(1-1/s0) #parameter beta0 (formula 11 of the Gould's article)
w0 = (1+beta0)/(1-beta0) #parameter w0 (formula 11 of the Gould's article)

n = len(s0) #size of s0

#integrande
def integrand(w):
    return (np.log(w+1))/w

#general integration (quad)
#Lq = np.zeros_like(s0)
#for i in range(0,n) :
#    Lq[i] = quad(integrand, 1, w0[i])[0]

#Gaussian quadrature
#Lg = fixed_quad(integrand, 1, w0[1])[0]

#Romberg's method
Lr = np.zeros_like(s0)
for i in range(0,n) :
    Lr[i] = romberg(integrand, 1, w0[i], tol=1.48e-08, rtol=1.48e-08)


#function phi (formula 10 of the Gould's article)
def phi(beta0, w0, Lr):
    a1 = (1+beta0**2)/(1-beta0**2)*np.log(w0)
    a2 = -np.log(w0)*beta0**2
    a3 = -(np.log(w0))**2
    a4 = -(4*beta0)/(1-beta0**2)
    a5 = 2*beta0
    a6 = 4*np.log(w0)*np.log(w0+1)
    a7 = -Lr
    return a1 + a2 + a3 + a4 + a5 + a6 + a7

phis = phi(beta0, w0, Lr)

#Graph
plt.plot(s0, phis)
plt.xlabel('parameter s0')
plt.ylabel('phi/(s0-1)')
plt.show()
