#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Energie_seuil import energy

##===========##
#  Functions  #
##===========##

def cross_section(eps, E, theta):

    """
    Return the cross section of the interaction gamma-gamma (formula 1 from Gould's article) (cm^2)

    Parameters:
        eps         : energy of the target photon (eV)
        E           : energy of the gamma photon (eV)
        theta       : angle between the two momenta of the two photons (rad)
    """

    def beta_func(eps, E, theta):

        """
        Return the parameter beta (formula 4 from Gould's article)

        Parameters:
            eps         : energy of the target photon (eV)
            E           : energy of the gamma photon (eV)
            theta       : angle between the two momenta of the two photons (rad)
        """

        def parameter_s(eps, E, theta):

            """
            Return the parameter s (formula 3 from Gould's article)

            Parameters:
                eps         : energy of the target photon (eV)
                E           : energy of the gamma photon (eV)
                theta       : angle between the two momenta of the two photons (rad)
            """

            s = eps*E/(2*(mc2*keV2eV)**2)*(1-np.cos(theta))
            ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition.

            return s, ind

        s, ind = parameter_s(eps, E, theta)
        s = s[ind[0]]

        return np.sqrt(1-1/s), ind

    beta, ind = beta_func(eps, E, theta)

    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2)), ind

def vec_plot(eps, E, theta):
    """
    Return the vector cross section for the plot (cm^2)

    Parameters:
        eps         : energy of the target photon (eV)
        E           : energy of the gamma photon (eV)
        theta       : angle between the two momenta of the two photons (rad)

    """

    return cross_section(eps, E, theta)

if __name__ == '__main__':


    # Parameters for the photons (target and gamma)

        # direction of the second photon (rad)
    theta = np.linspace(0.000001,np.pi,100)

        # Energy of the gamma photon (eV)
    E = np.array([1e9, 100e9, 1e12, 10e12])

        # Energy of the target photon (eV)
    l = np.array([1e-4, 10e-4, 100e-4, 1000e-4])            # Wavelenght (cm)
    eps = energy(l)                                         # Energy (eV)
    l = l*1e4                                               # in micrometer

    # Computation of the cross section for each energy of the gamma photon in interaction of each target photon

    for i in range (len(E)):

        plt.figure()

        for j in range (len(eps)):

            sigma, ind = vec_plot(eps[j], E[i], theta)
            plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[j])

        E_gev = E[i]/GeV2eV             # in GeV
        plt.title('Cross section for the first photon at %d GeV' %E_gev)
        plt.xlabel(r'$\alpha$''(rad)')
        plt.ylabel(r'$\sigma_{\gamma, \gamma}$''(cm' r'$^2$'')')
        plt.xlim(0,np.pi)
        plt.legend(loc='best')

    plt.show()
