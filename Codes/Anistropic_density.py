#Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

def distance(beta, L, R, z): #distance between the gamma photon and the source IR

    return np.sqrt((L - z)**2 + Rs**2 - 2 * (L - z) * Rs * np.cos(beta))

"""
density of the photons is not isotropic and is : dn = u/(a*T^4) * Bnu/(c*h*nu)

Parameters:

u_in is the choosen energy density of the (ponctual) star in a position along the line of sight (keV/cm^3)
b is the impact parameter (cm)
z is the position along the line of sight (cm)
eps is the energy of the target-photon (keV)
T is the temperature of the source (K)
alpha is the right ascension (rad)
L is the distance between us and the source of gamma (cm)
R is the galactic radius of the source of gamma (cm)
Rs is the galactic radius of the sun (cm)
"""

def density_n(u_in, b, d, eps, T):

    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s

    def energy_density(u_in, b, d, T): #energy density of the source IR

        #Global energy
        c = 2.99792458e+10 #light speed in cm/s
        sb = 5.670367e-12/1.602e-16 #Stefan-Boltzmann constant (keV s^-1 cm^-2 K^-4)
        a = 4 * sb/c #radiation constant (keV cm^-3 K^-4)
        h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s

        def luminosity(u_in, b): #luminosity of the source IR

            c = 2.99792458e+10 #light speed in cm/s

            return u_in * c * 4 * np.pi * b**2

        Lum = luminosity(u_in, b)

        return Lum/(4 * np.pi * d**2 * c) * 1.0/(a * T**4)

    nu = eps/h

    def planck(nu, T):

        #Global constants
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
        c = 2.99792458e+10 #light speed in cm/s

        return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(k*T)) - 1)

    Bnu = planck(nu, T)

    u = energy_density(u_in, b, d, T)

    return u * 4 * np.pi * Bnu/(h * c * eps)

eps = 1.24e-3 #keV
#Distance to the source
R = 8.5 *  3.085678e21#distance between the gamma-source and the IR source (cm)
Rs = 5 * 3.085678e21 #distance between us and the IR source (cm)
L = 10 * 3.085678e21 #distance between us and the gamma-source (cm)

beta = np.arccos((L**2 + Rs**2 - R**2)/(2 * Rs * L))
b = Rs * np.sin(beta) #impact parameter (cm)
z = np.linspace(0, L, 100) #position along the line of sight (cm)
d = distance(beta, L, Rs, z)

T = 25 #temperature of the CMB (K)
u_in = 1e-3

d = d/3.085678e21


dn = density_n(u_in, b, d, eps, T)

plt.plot(d, dn)
plt.show()
