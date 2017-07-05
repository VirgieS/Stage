#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *

#Parametre s (formula 3 from Gould's article)
def parametre_s(eps, E, mc2, theta):
    s = eps*E/(2*mc2**2)*(1-np.cos(theta))
    ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition.
    return s, ind

#Parametre beta (formula 4 from Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)
#voir si les deux fonctions fonctionnent

#Cross section of the interaction photon-photon (formula 1 from Gould's article)
def cross_section(s, r0):
    beta = beta_func(s)
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#Vector for the plot of the total cross section
def vec_plot(theta, eps, E, mc2):
    s = np.zeros_like(theta) #initialization
    [s,ind] = parametre_s(eps, E, mc2, theta)
    s = s[ind[0]]
    return cross_section(s, r0), ind

#Constants
c = 2.99792458e+10 #speed of light (cm/s)
mc2 = 510.9989461e+3 #electron mass (eV)
r0 =  2.818e-13 #classical electron radius (cm)
conv_en = 1.602e-19 # Conversion factor from J to eV
kb = 1.380658e-23/conv_en # Boltzmann's constant in eV/K
T = 10000 #temperature of the source (K)

#Parametres for the photons (target and gamma)

theta = np.linspace(0.000001,np.pi,100) #direction of the second photon (rad)

# Energy of the gamma-photon
E = 1e12  # eV
E_tev = E*1e-12   # TeV

eps = kb * T * 2.821439372

[sigma, ind] = vec_plot(theta, eps, E, mc2)
#eps_tev = eps
plt.plot(theta[ind[0]], sigma, label="$\epsilon$ = %.2f eV" %eps)

plt.title('Cross section for a gamma-photon at %.2f TeV' %E_tev)
plt.xlabel(r'$\theta$''(rad)')
plt.ylabel(r'$\sigma_{int}$''(cm' r'$^2$'')')
plt.xlim(0,np.pi)
plt.legend()
plt.show()
