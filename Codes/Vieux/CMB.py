import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate as itg

from principal_functions import integration, sigma_func_CMB

T = 2.7 #Temperature of the CMB (K)
k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV
hbar = 6.6260755e-34/1.602e-19/(2 * np.pi) #Planck's constant in eV
c = 2.99792458e+10 #light speed in cm/s
R = 0 #distance to the centre of the object in kpc
Rs = 8.5 #distance to the centre of the Sun in kpc
alpha = 0 #right ascension in radian

mc2 = 510.9989461E+3 #electron mass (eV)
epsc = np.linspace(mc2+1, 1e17, 10000)
rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))
L = np.sqrt(R**2 + rho**2)
x = np.linspace(0, L, 100)
E = 1e15

def func_x(epsc):
    [sigma, ind] = sigma_func_CMB(epsc, E)
    print(ind[0])
    sigma = sigma[ind[0]]
    epsc = epsc[ind[0]]
    ind = np.where(1 - np.exp(-epsc**2/(E * k * T)) > 0)
    sigma = sigma[ind[0]]
    epsc = epsc[ind[0]]
    print(1-np.exp(-epsc**2/(E * k * T)))
    integrand_eps = epsc**3 * sigma * np.log(1 - np.exp(-epsc**2/(E * k * T)))
    return integration(epsc, integrand_eps)

integral_eps = np.zeros_like(x)
integral_eps = func_x(epsc)
print(integral_eps)
#for i in range (0, len(x)):
    #integral_eps[i] = func_x(epsc)

#integral_x = integration(x, integral_eps)

#final = -4 * k *T/((hbar * c)**3 * np.pi**2 * E**2) * integral_x

#print(final)
