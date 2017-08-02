#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *

##=========##
# Functions #
##=========##

def energy(l):

    """
    Return the energy of a photon at a wavelenght l

    Parameter:
        l           : wavelenght (cm)
    """

    return (hp/eV2erg)*cl/l

def energy_th(theta, eps):

    """
    Return the threshold energy of the gamma photon for the interaction with a target photon at energy eps

    Parameters :
        theta       : angle between the two momenta of the two photons (rad)
        eps         : energy of the target photon (eV)
    """

    return 2*(mc2*keV2eV)**2/(eps*(1-np.cos(theta)))

##===================================##
# Computation of the threshold energy #
##===================================##

if __name__ == '__main__':

    # Parametres for the photons (target and gamma)

        # direction of the second photon (rad)

    theta = np.linspace(0,np.pi,100)

            # because of the equation 3 from Gould's article, we select only cos(theta)=!1

    ind = np.where(1-np.cos(theta)>0)
    theta = theta[ind[0]]

        # energy of the target photon (eV)
    l = np.array([1e-4, 10e-4, 100e-4, 1000e-4])        # wavelenght (cm)
    eps = energy(l)                                     # Energy (eV)
    print(eps)
    l = l*1e4                                           # micrometer

    # Computation of the threshold energy for each energy of the target photon

    for i in range (len(eps)):

        Eth = energy_th(theta, eps[i])
        plt.plot(theta, Eth/GeV2eV, label="%d $\mu$m" %l[i])

    plt.yscale('log')
    plt.xlim(0,np.pi)
    plt.xlabel(r'$\theta$' '(rad)')
    plt.ylabel(r'$E_{th}$''(GeV)')
    plt.title('Threshold energy of the first photon')
    plt.legend(loc='best')
    plt.show()
