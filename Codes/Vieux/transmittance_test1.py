#! /Applications/Anaconda/bin/python

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad, dblquad

def f1(psi, eps, E, T):

    #Density of a Black Body
    def density_n(eps, T):

        #Global constants
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
        h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
        c = 2.99792458e+10 #light speed in cm/s

        return 2 * eps**2/(h * c)**3 * 1/(np.exp(eps/(k*T)) - 1)

    #Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.
    def cos_theta(psi):

        return np.cos(np.pi - psi)

    costheta = cos_theta(psi)

    #sigma_int  = 1/2.0 * pi * r0**2 * sigma (equation 1 of Gould's article), we calculate sigma and not sigma_int
    def cross_section(eps, E, costheta, T): #eps and E must be given in keV

        #Parameter s as define at the equation 3 of Gould's article
        def parameter_s(eps, E, costheta):

            mc2 = 510.9989461 #electron mass (keV)

            return eps * E/(2 * mc2**2) * (1 - costheta)

        #Parameter beta as define at the equation 4 of Gould's article
        def parameter_beta(eps, E, costheta):

            s = parameter_s(eps, E, costheta)

            return np.sqrt(1 - 1/s)

        beta = parameter_beta(eps, E, costheta)
        beta = np.nan_to_num(beta)

        return (1 - beta**2)*((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 *beta * (2 - beta**2))

    sigma = cross_section(eps, E, costheta, T)
    dn = density_n(eps, T)

    return sigma * dn * (1 - costheta) * np.sin(psi)

def f2(eps, E, T):
    return quad(f1, epsmin, b, args=())

#The limits of integration in psi: from gfun(x) to hfun(x)
def gfun(x):
    return 0

def hfun(x):
    return np.pi


#Global constants
r0 =  2.818E-13 #classical elctron radius (cm)
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)
T = 2.7 #temperature of the CMB (K)

#For the vector eps, E
number_bin_E = 160
number_bin_eps = 20

#Distance to the source
R = 0 #distance to the Galactic center of the source (kpc)
Rs = 8.5 #distance to the Galactic center of Sun (kpc)
alpha_r = 0 #right ascension of the source (radian)
z = 0 #height above the Galactic plane (kpc)
L = np.sqrt(z**2 + R**2 + Rs**2 - 2*R*Rs*np.cos(alpha_r)) #distance to the source (kpc)
L = L * 3.085678e21 #in cm

#energy of the gamma-photon
Emin = log10(1e8)
Emax = log10(1e14)
E = np.logspace(Emin, Emax, number_bin_E) # keV
E_tev = E*1e-9 #TeV

integral_eps = np.zeros_like(E)

for i in range (len(E)):

    epsmin = mc2**2/E[i]
    epsmax = 2.0*k*T

    if epsmin > epsmax:
        continue

    else:
        integral_eps[i] = dblquad(f1, epsmin, epsmax, gfun, hfun, args=(E[i], T), epsabs=1.49e-08, epsrel=1.49e-08)[0]

tau = 1/2.0 * np.pi * r0**2 * L * 2 * np.pi * integral_eps
print(len(tau), len(E))
print(np.exp(-tau))

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
plt.plot(E_tev, np.exp(-tau))
plt.show()
