#! /Applications/Anaconda/bin/python

#Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

#librairies and functions
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import integration_log
from optical_depth import *
from transmittance_CMB_eq3 import f_eq3
from transmittance_CMB_eq1 import f_eq1

#For the vector eps, E
number_bin_E = 80
number_bin_eps = 20.0

#Constants
R = 0                                           # distance to the centre of the object (kpc)
Rs = 8.5                                        # distance to the centre of the Sun (kpc)
alpha = 0                                       # right ascension (rad)
rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))  # kpc
L = np.sqrt(R**2 + rho**2)                          # kpc
L = L * kpc2cm                                      # cm
T = 2.7                                         # temperature of the CMB (K)

# Energy of the gamma-photon
Emin = 1e-1*TeV2keV 			                # we choose Emin = 10^-1 TeV (keV)
Emax = 1e5*TeV2keV 			                    # we choose Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E)         # keV
E_tev = E/TeV2keV 					                            # TeV

# Computes the optical depth
integral_eps = np.zeros_like(E)
integral_x = np.zeros_like(E)
hbar = hp/(2*np.pi)

for i in range (len(E)):

    # Equation 3
    epsc_min = mc2/mc2                                                          # keV/mc2
    epsc_max = epsc_min + np.sqrt(10*(kb*erg2kev)*E[i]*T/mc2**2)                 # keV/mc2
    epsc = np.logspace(log10(epsc_min), log10(epsc_max), int(log10(epsc_max/epsc_min)*number_bin_eps))   # keV/mc2
    integrand = f_eq3(epsc, E[i], T)
    integral_x[i] = integration_log(epsc, integrand)

    # Equation 1
    epsmin = mc2**2/E[i]
    epsmax = 10*(kb*erg2kev)*T

    #Because epsmin must be lower than epsmax
    if epsmin > epsmax:
        continue
    else:
        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

    integral_psi = np.zeros_like(eps)
    psi = np.linspace(0, np.pi, 100) #azimutal angle if z is the direction of the gamma-photon

    for j in range (len(eps)):

        integrand = f_eq1(psi, eps[j], E[i], T)
        integrand = np.nan_to_num(integrand)

       	integral_psi[j] = integration_log(psi, integrand)

    integral_eps[i] = integration_log(eps, integral_psi)

tau_eq1 = 1/2.0 * np.pi * r0**2 * L * 2 * np.pi * integral_eps
tau_eq3 = integral_x * L * 4 * (kb*erg2kev) * T/(((hbar*erg2kev) * cl)**3 * np.pi**2 * E**2) * 1/2.0 * np.pi * r0**2 * mc2**3 * mc2

fig = plt.figure()
ax = fig.add_subplot(111)

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
plt.plot(E_tev, np.exp(-tau_eq1), label = 'eq1')
plt.plot(E_tev, np.exp(-tau_eq3), label = 'eq3')
plt.text(0.0, 0.05,'number_bin_eps(c) = %.2f' %number_bin_eps, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
plt.legend()
plt.show()
