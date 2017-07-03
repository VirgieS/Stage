import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

#Function to calculate rho
def rho_func(R, Rs, alpha):
    return np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))

#Calculation of L
def L_func(R, Rs, alpha, z):
    rho = rho_func(R, Rs, alpha)
    return np.sqrt(rho**2 + z**2)

#parametre s (formula 3 from Gould's article)
def parametre_s(Ee, mc2):
    s = (Ee/mc2)**2
    ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition
    return s, ind

#Parametre beta (formula 4 from Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)

#Cross section of the interaction photon-photon (formula 1 from Gould's article) (cm^2)
def cross_section(s):
    r0 =  2.818E-13 #classical elctron radius (cm)
    beta = beta_func(s)
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#Integrand
def f(epsc, x):
    s = (epsc/mc2)**2
    sigma = cross_section(s)

    L = L_func(R, Rs, alpha, z)
    E = np.linspace(1e11, 1e17, 10000)

    tau = np.zeros_like(E)
    for i in range (0, len(E)):
        tau[i] = integrate.dblquad(epsc**3 * sigma * np.log(1-np.exp(-epsc**2/(E[i]*k*T)))*x, 0, L, gfun, hfun, epsabs=1.49e-08, epsrel=1.49e-08)[0]
    return tau

#Function to calculate the optical depth
#def optical_depth(R, Rs, alpha, z, mc2):
#    return integrate.nquad(integrand, [[0, L_func(R, Rs, alpha, z)], [mc2, inf]])

def gfun(x):
    return 510.9989461e+3

def hfun(x):
    return np.inf

#Parametres
mc2 = 510.9989461e+3 #electron mass (eV)
T = 2.7 #cosmic microwave background (K)
c = 2.99792458 #light speed (cm/s)
hbar = 6.6260755e-34/1.602e-19*1/(2*np.pi) #Planck's constant (eV/s)
k = 1.380658e-23/1.602e-19 #Boltzmann's constant (eV/K)
z = 0 #Redshift
R = 0 #Distance from the Galactic center (kpc)
Rs = 8.5 #Distance of the Sun from the Galactic center (kpc)
alpha = 0

tau = f(epsc, x)

plt.plot(E, np.exp(-tau))
plt.show()
