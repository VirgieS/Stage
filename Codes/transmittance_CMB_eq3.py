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

def f_eq3(epsc, E, T):

    """
    Return the function that we integration to obtain the transmittance

    Parameters:
	epsc 	 : center-of-momentum system energy of a photon (keV/mc2)
	E	     : energy of the gamma photon (keV)
	T	     : temperature of the source (K)
    """

    def cross_section(epsc):

        def parameter_s(epsc):

            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    log = np.log(1 - np.exp(- (epsc * mc2)**2/(E * (kb*erg2kev) * T)))
    sigma = cross_section(epsc)
    return -epsc**3 * sigma * log

if __name__ == '__main__':

    #For the vector eps, E
    number_bin_E = 80
    number_bin_epsc = 20.0

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
    integral_x = np.zeros_like(E)

    for i in range(len(E)):
        epsc_min = mc2/mc2                                                          # keV/mc2
        epsc_max = epsc_min + np.sqrt(10*(kb*erg2kev)*E[i]*T/mc2**2)                 # keV/mc2
        epsc = np.logspace(log10(epsc_min), log10(epsc_max), int(log10(epsc_max/epsc_min)*number_bin_epsc))   # keV/mc2
        integrand = f_eq3(epsc, E[i], T)

        integral_x[i] = integration_log(epsc, integrand)

    hpbar = hp/(2*np.pi)
    tau = integral_x * L * 4 * (kb*erg2kev) * T/((hpbar*erg2kev * cl)**3 * np.pi**2 * E**2) * 1/2.0 * np.pi * r0**2 * mc2**3 * mc2


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xscale('log')
    plt.xlabel(r'$E_\gamma$' '(TeV)')
    plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
    plt.text(0.75, 0.95,'equation 3 of Moskalenko', horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
    plt.plot(E_tev, np.exp(-tau))
    plt.show()
