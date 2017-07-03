import matplotlib.pyplot as plt
import numpy as np

def integrand(epsc, mc2, E, r0, k, hbar, c):

    def cross_section(epsc, mc2, r0):

        def parameter_beta(s):
            return np.sqrt(1 - 1/s) #equation 4 from Gould's article

        s = (epsc/mc2)**2 #epsc = (1/2 * eps * E * (1 - cos(theta)))^(1/2), the center-of-momentum system energy of a photon-photon
        beta = parameter_beta(s)

        return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))#, ind

    sigma = cross_section(epsc, mc2, r0)

    def density_n(eps, T):
        h = 6.6260755e-34/1.602e-19 #Planck's constant in eV

        #spectral radiance of a black body (in wavelength)
        def B_eps(eps, T, h):
            k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV
            c = 2.99792458e+10 #light speed in cm/s
            return 2 * eps**3 / (h**4 * c**3) * 1/(np.exp(eps/(k * T))-1)

        Beps = B_eps(eps, T, h)

        return  Beps * h/eps
