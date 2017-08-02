"""
Now we consider that the star, not along the line of sight, is a sphere of uniform brightness (black body at T) with a radius R.

Our coordinate are theta, angle formed by the ray from the star and the direction of the centre from a position on the line of sight and phi, polar angle around this axis.

Parameters that we can give for this code are:
    L      : distance to the gamma-source (au)
    zb     : position along the line of sight nearly the star (au)
    b      : impact parameter (au)
    R      : radius of the star (au)
    T      : temperature of the star (K)
"""

# Librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import integration_log
from optical_depth import *

if __name__ == '__main__':

        # Parameters for the code
    L = 20 * AU2cm                              # the distance to the gamma-source (cm)
    zb = np.array([5, 10, 15]) * AU2cm          # position along the line of sight nearly the star (cm)
    b = 5 * AU2cm                               # impact parameter (cm)
    D_star =  np.sqrt(b**2 + (L - zb)**2)       # distance to the star (from us) (cm)
    D_gamma = np.sqrt(b**2 + zb**2)             # distance between the star and the gamma-source (cm)
    R = 0.5 * AU2cm                             # radius of the star (express in Rsun)
    T = 3000                                    # temperature of the star (K)
    z = np.linspace(0, L, 100)                  # position along the line of sight (cm)
    phi = np.linspace(0, 2*np.pi, 10)           # polar angle (rad)

        # Energy of the gamma-photon
    E = 1e9                                     # keV
    E_tev = E*keV2eV/TeV2eV                     # TeV

    b_au = b/AU2cm                              # au
    z_au = z/AU2cm                              # au

    # Computation and plot of the transmittance

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range (len(zb)):
        tau = calculate_tau(E, z, phi, zb[i], b, R, T)
        zb_au = zb[i]/AU2cm                     # au
        plt.plot(z_au, tau, label = "zb = %.2f au" %(zb_au))

    D_star_au = D_star/AU2cm # in au
    L_au = L/AU2cm                              # au
    R_au = R/AU2cm                              # au

    plt.xlabel(r'z (au)')
    plt.ylabel(r'$\frac{d \tau_{\gamma \gamma}}{d z}$' ' ' r'$(cm^{-1})$' )
    #plt.title(u'Optical depth for the interaction between 'r'$\gamma$' '-rays at %.2f GeV \n and photons of a star at %.2f K and a radius %.2f au' %(E_gev, T, R_au))
    plt.text(0.65, 0.5, u'L = %.2f au, T = %.2f K \nb = %.2f au, E$_\gamma$ = %.2f TeV' %(L_au, T, b_au, E_tev), horizontalalignment='left',
     verticalalignment='center', transform = ax.transAxes)
    plt.legend(loc='best')
    plt.show()
