#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *

def planck1(eps, T):
    c = 2.99792458e+10 #light speed in cm/s
    k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s

    nu = eps/h

    return 2*h*nu**3/c**2 * 1/(np.exp(h*nu/(k*T)) - 1), nu

def planck2(eps, T):
    c = 2.99792458e+10 #light speed in cm/s
    k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s

    return 2*eps**3/(h**3 * c**2) * 1/(np.exp(eps/(k*T)) - 1)

#Density of a Black Body
def density_n(eps, T):

    #Global constants
    k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s

    return 2 * eps**2/(h * c)**3 * 1/(np.exp(eps/(k*T)) - 1) #how does the density vqry with the energy ?

T = 2.7
mc2 = 510.9989461 #electron mass (keV)
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K

#energy of the gamma-photon
Emin = log10(1e8)
Emax = log10(1e14)
E = np.logspace(Emin, Emax, 80) # keV
Ef = 10**E #keV

for i in range (len(E)):

    epsmin = mc2**2/E[i]
    epsmax = 15*k*T

    if epsmin > epsmax:
        continue
    else:
        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*40.0))

    #[Bnu, nu] = planck1(eps, T)
    #Beps = planck2(eps, T)
    dn = density_n(eps, T)
    plt.plot(eps, dn)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend('%.2e' %Ef[i])
    plt.show()
