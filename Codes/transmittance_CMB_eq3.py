#librairies and functions
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import integration_log
from optical_depth import *

##=========##
# Functions #
##=========##

def f(epsc, E, T): # eps must be given in keV/mc2 and e in keV

    """
    Return the function that we integration to obtain the transmittance

    Parameters:
	epsc 	: center-of-momentum system energy of a photon (keV/mc2)
	E	: energy of the gamma photon (keV)
	T	: temperature of the source (K)
    """

    k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
    mc2 = 510.9989461 #electron mass (keV)

    def cross_section(epsc): #sigma_int  = 1/2.0 * pi * r0**2 * sigma (equation 1 of Gould's article), we calculate sigma and not sigma_int

        def parameter_s(epsc):

            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    log = np.log(1 - np.exp(- (epsc * mc2)**2/(E * k * T)))
    sigma = cross_section(epsc)
    return -epsc**3 * sigma * log

#For the vector eps, E
number_bin_E = 80
number_bin_eps = 40.0

#Constants
r0 =  2.818E-13 #classical elctron radius (cm)
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)
hbar = 6.6260755e-34/1.602e-16*1/(2*np.pi) #Planck's constant in keV*s
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
R = 0 #distance to the centre of the object in kpc
Rs = 8.5 #distance to the centre of the Sun in kpc
alpha = 0 #right ascension in radian
rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha)) #in kpc
L = np.sqrt(R**2 + rho**2) #in kpc
L = L * 3.085678e21 #in cm

#Parameters
T = 2.7 #temperature of the CMB in K
E = np.linspace(1e8, 1e14, 100) #energy of the second photon in keV
lxmin = np.log10(E.min())
lxmax = np.log10(E.max())
E = 10**np.linspace(lxmin, lxmax, 100) #energy-vector with a number of points that increase like a power of 10
#Emin = log10(1e8)
#Emax = log10(1e14)
#E = np.logspace(Emin, Emax, number_bin_E) # keV
Ef = E*1e-12*1e3
integral_x = np.zeros_like(E)

for i in range(len(E)):
    epsc = np.linspace(1+0.01, 1 + 0.01 + np.sqrt(10*k*E[i]*T/mc2**2), 1000) #energy of the target-photon in keV/mc2
    integrand = f(epsc, E[i], T)

    integral_x[i] = integration_log(epsc, integrand)

tau = integral_x * L * 4 * k * T/((hbar * c)**3 * np.pi**2 * E**2) * 1/2.0 * np.pi * r0**2 * mc2**3 * mc2

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
plt.plot(Ef, np.exp(-tau))
plt.show()
