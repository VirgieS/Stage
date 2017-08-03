#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *

##=========##
# Functions #
##=========##

def cross_section(Ee):

    """
    Return the cross section of the interaction gamma-gamma from the forula 1 of Gould's article

    Parameter :
        Ee      : electron (positron) total energy in the CM frame (eV)
    """

    def beta_func(Ee):

        """
        Return the parameter beta from the formula 4 of Gould's article

        Parameter:
            Ee      : electron (positron) total energy in the CM frame (eV)
        """

        def parameter_s(Ee):

            """
            Return the parameter s from the formula 3 of Gould's article

            Parameter:
                Ee      : electron (positron) total energy in the CM frame (eV)
            """

            s = (Ee/mc2/keV2eV)**2
            ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition

            return s, ind

        s, ind = parameter_s(Ee)
        s = s[ind[0]]

        return np.sqrt(1-1/s), ind

    beta, ind = beta_func(Ee)

    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2)), ind

##================================================##
# Computation of the cross section in the CM frame #
##================================================##

if __name__ == '__main__':

    # Electron (positron) total energy in the CM frame (eV)
    E_min = mc2*keV2eV
    E_max = 100*E_min
    Ee = np.linspace(E_min, E_max, 10000)

    # Computation of the cross section (cm)
    sigma, ind = cross_section(Ee)

    # Plot of the cross section follow the total energy in the CM frame
    Ee = Ee[ind[0]]/MeV2eV                  # MeV

    plt.xscale('log')

    plt.plot(Ee, sigma)
    plt.title('Cross section for different energy in the CM frame')
    plt.xlabel(r'$E_{cm,e}$''(MeV)')
    plt.ylabel(r'$\sigma_{\gamma, \gamma}$''(cm' r'$^2$'')')
    plt.show()
