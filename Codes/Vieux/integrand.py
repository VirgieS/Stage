import matplotlib.pyplot as plt
import numpy as np

#parametre s (formula 3 from Gould's article)
def parametre_s(epsc, mc2):
    s = (epsc/mc2)**2
    ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition
    return s, ind

#Parametre beta (formula 4 from Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)

#Cross section of the interaction photon-photon (formula 1 from Gould's article) (cm^2)
def cross_section(s, r0):
    beta = beta_func(s)
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

def f(epsc, E, k, T, mc2, r0):
    [s, ind] = parametre_s(epsc, mc2)
    s = s[ind[0]]
    epsc = epsc[ind[0]]
    sigma = cross_section(s, r0)
    print(sigma)
    return epsc**3 * sigma * np.log(1 - np.exp(-epsc**2/(E * k * T)))

T = 2.7 #temperature of the CMB in K
r0 =  2.818E-13 #classical elctron radius (cm)
c = 2.99792458e+10 #light speed in cm/s
h = 6.6260755e-34/1.602e-19 #Planck's constant in eV
mc2 = 510.9989461e+3 #electron mass (eV)
k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV
l = np.array([1e-4, 10e-4, 100e-4, 1000e-4]) #wavelength of the second photon-photon
eps = np.linspace(510.9989461e+3, 1e12, 10000)
E = 1e15
for i in range (0, len(eps)):
    integrand = f(eps, E, k, T, mc2, r0)
    print(integrand)
