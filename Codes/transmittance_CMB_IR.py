#! /Applications/Anaconda/bin/python

#Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

#librairies
#librairies and functions
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import integration_log
from optical_depth import *
from transmittance_CMB_eq1 import f_eq1
from transmittance_IR import f_IR

# Global constants
TCMB = 2.7 #temperature of the CMB (K)
TIR = 25 #temperature of the IR (K)

# For the vector eps, E
number_bin_E = 80
number_bin_eps = 40.0

# Distance to the source
R = 0                                   # distance to the Galactic center of the source (kpc)
Rs = 8.5                                # distance to the Galactic center of Sun (kpc)
alpha_r = 0                             # right ascension of the source (radian)
z = 0                                   # height above the Galactic plane (kpc)
L = np.sqrt(z**2 + R**2 + Rs**2 - 2*R*Rs*np.cos(alpha_r))   # distance to the source (kpc)
L = L * kpc2cm                                              # cm

# Energy of the gamma-photon
Emin = 1e-1*TeV2keV 			                # we choose Emin = 10^-1 TeV (keV)
Emax = 1e5*TeV2keV 			                    # we choose Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E)         # keV
E_tev = E/TeV2keV 					                            # TeV

integral_eps = np.zeros_like(E)

for i in range (len(E)):

    epsmin = mc2**2/E[i]
    epsmax = 10*(kb*erg2kev)*TCMB

    #Because epsmin must be lower than epsmax
    if epsmin > epsmax:
        continue
    else:
        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

    integral_psi = np.zeros_like(eps)
    psi = np.linspace(0, np.pi, 100) #azimutal angle if z is the direction of the gamma-photon

    for j in range (len(eps)):

        integrand = f_eq1(psi, eps[j], E[i], TCMB)
        integrand = np.nan_to_num(integrand)

       	integral_psi[j] = integration_log(psi, integrand)

    integral_eps[i] = integration_log(eps, integral_psi)

tau_CMB = 1/2.0 * np.pi * r0**2 * L * 2 * np.pi * integral_eps #the intergation in phi gives 2*pi and the integration in x gives L

integral_eps = np.zeros_like(E)

for i in range (len(E)):

    epsmin = mc2**2/E[i]
    epsmax = 10*(kb*erg2kev)*TIR

    #Because epsmin must be lower than epsmax
    if epsmin > epsmax:
        continue
    else:
        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

    integral_psi = np.zeros_like(eps)
    psi = np.linspace(0, np.pi, 100) #azimutal angle if z is the direction of the gamma-photon

    for j in range (len(eps)):

        integrand = f_IR(psi, eps[j], E[i], TIR)
        integrand = np.nan_to_num(integrand)

        integral_psi[j] = integration_log(psi, integrand)

    integral_eps[i] = integration_log(eps, integral_psi)

tau_IR = 1/2.0 * np.pi * r0**2 * L * 2 * np.pi * integral_eps #the intergation in phi gives 2*pi and the integration in x gives L

CMB, = plt.plot(E_tev, np.exp(-tau_CMB), label="CMB")
IR, = plt.plot(E_tev, np.exp(-tau_IR), label="IR")

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB and IR photons')
plt.legend(loc='best')
plt.show()
