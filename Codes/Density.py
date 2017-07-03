#! /Applications/Anaconda/bin/python

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad, dblquad

def f1(eps, T):

    #Density of a Black Body
    def density_n(eps, T):

        #Global constants
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
        h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
        c = 2.99792458e+10 #light speed in cm/s

        nu = eps/h

        def planck_nu(nu, T):

            return 2*h*nu**3/c**2 * 1/(np.exp((h*nu)/(k*T)) - 1)

        def planck_lambda(l, T):

            return 2 * h * c**2/l**5 * 1/(np.exp((h*c)/(l*k*T)) - 1)

        Bnu = planck_nu(nu, T)

        return Bnu/(h * eps * c)

    dn = density_n(eps, T)

    return dn * eps

#Global constants
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
c = 2.99792458e+10 #light speed in cm/s
T = 2.7 #temperature of the CMB (K)
sb = 5.670367e-12/1.602e-16 #Stefan-Boltzmann constant (keV s^-1 cm^-2 K^-4)
a = 4 * sb/c #radiation constant (keV cm^-3 K^-4)

energy = quad(f1, 0, 10000*k*T, args = (T))[0]
energy_v = 4*np.pi*energy/(a*T**4)
print(a)
print(a*T**4)
print(energy_v)
