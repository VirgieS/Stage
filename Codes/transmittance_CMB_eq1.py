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

##=========##
# Functions #
##=========##

def f_eq1(psi, eps, E, T):

    """
    Return the function that we integrate to compute the optical depth

    Parameters:
	psi	: azimutal angle of the target photon (rad)
	eps	: energy of the target photon (keV)
	E	: energy of the gamma photon (keV)
	T	: temperature of the isotropic source (K)
    """

    def density_n(eps, T):

	"""
	Return the density of a Black Body

	Parameters:
	    eps		: energy of the target photon (keV)
	    T		: temperature of the isotropic source (K)
	"""

        nu = eps/(hp*erg2kev)

        def planck(nu, T):

            return (2*(hp*erg2kev)*(nu**3))/(cl**2) * 1/(np.exp(((hp*erg2kev)*nu)/((kb*erg2kev)*T)) - 1)

        Bnu = planck(nu, T)

        return Bnu/(hp*erg2kev * eps * cl)

    def cos_theta(psi):

	"""
	Return the cosinus of the angle between the two momenta of the two photons

	Parameter:
	    psi		: azimutal angle of the target photon (rad)
	"""

        return np.cos(np.pi-psi)

    costheta = cos_theta(psi)
    epsc = np.sqrt(eps * E/2 * (1 - costheta))/mc2 #it gives epsc in keV/mc2

    def cross_section(epsc):

	"""
	Return the dimensionless cross section of the interaction gamma-gamma

	Parameter:
	    epsc	: center-of-momentum system energy of a photon
	"""

        def parameter_s(epsc):

            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

        beta = parameter_beta(epsc)
        beta = np.nan_to_num(beta)

        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    sigma = cross_section(epsc)
    dn = density_n(eps, T)

    return sigma * dn * (1 - costheta) * np.sin(psi)

if __name__ == '__main__':

    # Global constants
    T = 2.7 				# temperature of the CMB (K)

    # For the vector eps, E
    number_bin_E = 80
    number_bin_eps = 40.0

    # Distance to the source
    R = 0 					# distance to the Galactic center of the source (kpc)
    Rs = 8.5 				# distance to the Galactic center of Sun (kpc)
    alpha_r = 0 				# right ascension of the source (radian)
    z = 0 					# height above the Galactic plane (kpc)
    L = np.sqrt(z**2 + R**2 + Rs**2 - 2*R*Rs*np.cos(alpha_r)) # distance to the source (kpc)
    L = L * kpc2cm 						  # cm

    # Energy of the gamma-photon
    Emin = 1e-1*TeV2keV 			# we choose Emin = 10^-1 TeV (keV)
    Emax = 1e5*TeV2keV 			# we choose Emax = 10^5 TeV (keV)
    E = np.logspace(log10(Emin), log10(Emax), number_bin_E) # keV
    E_tev = E/TeV2keV 					#TeV

    integral_eps = np.zeros_like(E)

    for i in range (len(E)):

        epsmin = mc2**2/E[i]
        epsmax = 10*(kb*erg2kev)*T

        # Because epsmin must be lower than epsmax
        if epsmin > epsmax:
            continue
        else:
            eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

        integral_psi = np.zeros_like(eps)
        psi = np.linspace(0, np.pi, 100) # azimutal angle if z is the direction of the gamma-photon

        for j in range (len(eps)):

            integrand = f(psi, eps[j], E[i], T)
            integrand = np.nan_to_num(integrand)
            integral_psi[j] = integration_log(psi, integrand)

        integral_eps[i] = integration_log(eps, integral_psi)

    tau = 1/2.0 * np.pi * r0**2 * L * 2 * np.pi * integral_eps # the intergation in phi gives 2*pi and the integration in x gives L

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xscale('log')
    plt.xlabel(r'$E_\gamma$' '(TeV)')
    plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
    plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
    plt.plot(E_tev, np.exp(-tau))
    plt.text(0.0, 0.05,'equation 1 of Moskalenko', horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.show()
