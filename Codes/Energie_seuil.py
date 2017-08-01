#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import *

#Wavelength (cm) to energy (eV)
def energy(l):
    return (hp/eV2erg)*cl/l

#Threshold energy
def energy_th(theta, eps):
    return 2*(mc2*keV2eV)**2/(eps*(1-np.cos(theta)))

#Parametres for the photons (target and gamma)

theta = np.linspace(0,np.pi,100)                # direction of the second photon (rad)
ind = np.where(1-np.cos(theta)>0)               # because of the equation 3 from Gould's article
theta = theta[ind[0]]                           # we selected only cos(theta)=!1
l = np.array([1e-4, 10e-4, 100e-4, 1000e-4])    # wavelength of the target (cm)
eps = energy(l)                                 # energy of the second photon (eV)
l = l*1e4                                       # in mm

Eth = energy_th(theta, eps[0])
first, = plt.plot(theta, Eth*1e-9, label = "%d $\mu$m" %l[0])

Eth = energy_th(theta, eps[1])
second, = plt.plot(theta, Eth*1e-9, label = "%d $\mu$m" %l[1])

Eth = energy_th(theta, eps[2])
third, = plt.plot(theta, Eth*1e-9, label = "%d $\mu$m" %l[2])

Eth = energy_th(theta, eps[3])
fourth, = plt.plot(theta, Eth*1e-9, label = "%d $\mu$m" %l[3])

plt.yscale('log')
plt.xlim(0,np.pi)
plt.xlabel(r'$\theta$' '(rad)')
plt.ylabel(r'$E_{th}$''(GeV)')
plt.title('Threshold energy of the first photon')
plt.legend(handles=[first, second, third, fourth])
plt.show()
