#librairies
import matplotlib.pyplot as plt
import numpy as np

#Wavelength (cm) to energy (eV)
def energy(l):
    h = 6.6260755e-34/1.602e-19 #Planck's constant (eV/K)
    c = 2.99792458E+10 #speed of light (cm/s)
    return h*c/l

#parametre s (formula 3 from Gould's article)
def parametre_s(epsc, mc2):
    s = (epsc/mc2)**2
    ind = np.where(s>=1)
    return s, ind

#Parametre beta (formula 4 from Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)

#Cross section of the interaction photon-photon (formula 1 from Gould's article) (cm^2)
def cross_section(s, r0):
    beta = beta_func(s)
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#Constants
c = 2.99792458E+10 #speed of light (cm/s)
mc2 = 510.9989461E+3 #electron mass (eV)
r0 =  2.818E-13 #classical elctron radius (cm)
k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV

E = 1e17 #energy of the first photon
eps = 1 #epsilon pour eviter la divergence de l'integrale
epsc = np.linspace(mc2+eps, 1e12, 10000) #energy of the second photon

T = 2.7 #temperature of the CMB in K
[s, ind] = parametre_s(epsc, mc2)
s = s[ind[0]]
sigma = cross_section(s, r0)
epsc = epsc[ind[0]]
print(len(sigma))

integrand = np.zeros(len(epsc))

for i in range (0, len(epsc)):
    integrand = epsc**3 * sigma * np.log(1 - np.exp(-epsc**2/(E*k*T)))
    print(integrand)
