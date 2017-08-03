#Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

# Librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import integration_log, calculate_tau

##=========##
# Functions #
##=========##

def distance(beta, L, D_s, z):
    """
    Return the distance betwee the gamma photon and the IR source

    Parameters:
        beta        : angle formed by the direction to the gamma source and the direction to the IR source from the observer (rad)
        L           : lenght of the line of sight (cm)
        D_s         : distance to the IR source from the observer (cm)
        z           : position along the line of sight (cm)
    """

    return np.sqrt((L - z)**2 + D_s**2 - 2 * (L - z) * D_s * np.cos(beta))

def density_n(u_in, b, D, eps, T):

    """
    Return the density of the photons is not isotropic and is : dn = u/(a*T^4) * Bnu/(c*h*nu)

    Parameters:

        u_in            : choosen energy density of the (ponctual) star in a position along the line of sight (erg/cm^3)
        b               : impact parameter (cm)
        D               : distance to the IR source from the gamma photon (cm)
        eps             : energy of the target-photon (erg)
        T               : temperature of the source (K)
    """

    def energy_density(u_in, b, D, T):

        """
        Return the dimesnionless energy density of the source IR (u = u/(asb*T^4))

        Parameters:
            u_in            : choosen energy density of the (ponctual) star in a position along the line of sight (erg/cm^3)
            b               : impact parameter (cm)
            D               : distance to the IR source from the gamma photon (cm)
            T               : temperature of the source (K)
        """

        def luminosity(u_in, b):

            """
            Return the luminosity of the source IR for a choosen energy density at a particular distance

            Parameters:
                u_in            : choosen energy density of the (ponctual) star in a position along the line of sight (erg/cm^3)
                b               : impact parameter (cm)
            """

            return u_in * cl * 4 * np.pi * b**2

        Lum = luminosity(u_in, b)                               # erg/s

        return Lum/(4 * np.pi * D**2 * cl) * 1.0/(asb * T**4)

    nu = eps/hp

    def planck(nu, T):

        """
        Return the density of a Black body

        Parameters:
            nu          : frequency of the target photon (Hz)
            T           : temperature of the source (K)
        """

        return (2*hp*(nu**3))/(cl**2) * 1/(np.exp((hp*nu)/(kb*T)) - 1)

    Bnu = planck(nu, T)

    u = energy_density(u_in, b, D, T)

    return 4 * np.pi * u * Bnu/(hp * cl * eps)                  # cm^-3/sr/erg

def angle(D, b, z, zb):

    """
    Return the angle between the two momenta of the two photons in the observer's frame

    Parameters:
        D               : distance to the IR source from the gamma photon (cm)
        b               : impact parameter (cm)
        z               : position along the line of sight (cm)
        zb              : position along the line of sight closely the source (cm)
    """

    if z <zb:
        theta = np.pi - np.arcsin(b*1.0/D)

    else:
        theta = np.arcsin(b*1.0/D)

    return theta

def f(u_in, b, D, eps, T, z, zb, E):

    """
    Return the function for integration to compute the optical depth

    Parameters:
        u_in            : choosen energy density of the (ponctual) star in a position along the line of sight (erg/cm^3)
        b               : impact parameter (cm)
        D               : distance between the gamma-photon and the source IR (cm)
        eps             : energy of the target-photon (keV)
        T               : temperature of the source (K)
        z               : position along the line of sight (cm)
        zb              : position along the line of sight closely the source (cm)
        E               : energy of the gamma-photon (erg
    """

    theta = angle(D, b, z, zb)

    epsc = np.sqrt(eps * E/2 * (1 - np.cos(theta)))/(mc2/erg2kev) #it gives epsc in erg/(mc2 in erg)

    def cross_section(epsc):
        """
    	Return the dimensionless cross section of the interaction gamma-gamma (formula 1 of Gould's article)

    	Parameter:
    	    epsc	: center-of-momentum system energy of a photon
    	"""

        def parameter_s(epsc):

            """
    	    Return the parameter s (formula 3 of Gould's article)

    	    Parameter:
    	        epsc	: center-of-momentum system energy of a photon
    	    """

            return epsc**2

        def parameter_beta(epsc):

            """
    	    Return the parameter beta (formula 4 of Gould's article)

    	    Parameter:
    	        epsc	: center-of-momentum system energy of a photon
    	    """

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        beta = np.nan_to_num(beta)

        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    dn = density_n(u_in, b, D, eps, T)          # differential density of target photons (cm^-3/sr/erg)
    sigma = cross_section(epsc)                 # dimensionless cross section of the interaction gamma-gamma

    return dn * sigma * (1 - np.cos(theta))

def calculate_tau(E, u_in, b, D, T, z, zb):

    """
    Return the optical depth for a dimensionless source

    Parameters:
        E               : energy of the gamma photon (erg)
        u_in            : choosen energy density of the (ponctual) star in a position along the line of sight (erg/cm^3)
        b               : impact parameter (cm)
        D               : distance between the gamma-photon and the source IR (cm)
        T               : temperature of the source (K)
        z               : position along the line of sight (cm)
        zb              : position along the line of sight closely the source (cm)
    """

    integral = np.zeros_like(E)

    for i in range(len(E)):

        integral_eps = np.zeros_like(z)

        # for the vector eps
        number_bin_eps = 20.0

        #energy of the target-photon
        epsmin = (mc2/erg2kev)**2/E[i]
        epsmax = 10*kb*T

        #Because epsmin must be lower than epsmax
        if epsmin > epsmax:
            continue
        else:
            eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

        for j in range(len(z)):

            integrand = f(u_in, b, D[j], eps, T, z[j], zb, E[i])
            integrand = np.nan_to_num(integrand)

            integral_eps[j] = integration_log(eps, integrand)

        integral[i] = integration_log(z, integral_eps)

    return 1/2.0 * np.pi * r0**2 * integral

if __name__ == '__main__':

    # For the vector eps, E
    number_bin_E = 80

    # Distance to the source
    D_gamma = np.array([2, 10, 20]) *  kpc2cm               # distance between the gamma-source and the IR source (cm)
    D_s = 12 * kpc2cm                                       # distance between us and the IR source (cm)
    L = 10 * kpc2cm                                         # distance between us and the gamma-source (cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for k in range (len(D_gamma)):

        beta = np.arccos((L**2 + D_s**2 - D_gamma[k]**2)/(2 * D_s * L))

        b = D_s * np.sin(beta)                              # impact parameter (cm)
        z = np.linspace(0, L, 100)                          # position along the line of sight (cm)
        zb = np.sqrt(D_gamma[k]**2 - b**2)                  # position along the line of sight closely the source IR (cm)
        D = distance(beta, L, D_s, z)

        # Parameters
        T = 20                                              # temperature of the source (K)
        u_in = 1e-3/erg2kev                                 # energy density at the impact parameter (keV/cm^3)

        # Energy of the gamma-photon
        Emin = 1e-1 * TeV2keV/erg2kev                       # we choose Emin = 10^-1 TeV (erg)
        Emax = 1e5 * TeV2keV/erg2kev                        # we choose Emax = 10^5 TeV (erg)
        E = np.logspace(log10(Emin), log10(Emax), number_bin_E) # erg
        E_tev = E/(TeV2keV/erg2kev)                           # TeV

        tau = calculate_tau(E, u_in, b, D, T, z, zb)
        Dgamma_kpc = D_gamma[k]/kpc2cm                          # kpc

        plt.plot(E_tev, np.exp(-tau), label = "D$_\gamma$ = %.2f kpc" %Dgamma_kpc)

    Ds_kpc = D_s/kpc2cm                                         # kpc
    L_kpc = L/kpc2cm                                            # kpc

    plt.xscale('log')
    plt.xlabel(r'$E_\gamma$' '(TeV)')
    plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
    plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with IR photons of a ponctual source at %.2f K' %T)
    plt.text(0.80, 0.5,'D$_s$ = %.2f kpc,\nL = %.2f kpc' %(Ds_kpc, L_kpc), horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.legend(loc='best')
    plt.show()
