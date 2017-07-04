"""
Now we consider that the star, not along the line of sight, is a sphere of uniform brightness (black body at T) with a radius R.

Our coordinate are theta, angle formed by the ray from the star and the direction of the centre from a position on the line of sight and phi, polar angle around this axis.

Parameters that we can give for this code are:
    L      : distance to the gamma-source (kpc)
    D_star : distance to the star (kpc)
    z_b    : position along the line of sight nearly the star (kpc)
    b      : impact parameter (kpc)
    R      : radius of the star (m)
    T      : temperature of the star (K)
"""

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

#Important functions for the calculation

def density_n(eps, T, theta):

    """
    Density of the photons is not isotropic : dn = Bnu * cos(theta)/(c * h**2 *nu)

    Parameters:
        eps   : energy of the target-photon (keV)
        theta : angle formed by the ray (from the star) and the straight line connecting the centre of the star and a position z on the line of sight (rad)
        T     : temperature of the star (K)
    """

    #Global constants
    h = 6.6260755e-34/conv_en #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s

    nu = eps/h

    def planck(nu, T):

        #Global constants
        k = 1.380658e-23/conv_en #Boltzmann's constant in keV/K
        c = 2.99792458e+10 #light speed in cm/s

        return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(k*T)) - 1)

    Bnu = planck(nu, T)

    return Bnu/(c * h**2 * nu) * np.cos(theta) #return dn in cm^3/sr/keV

def distance(L, D_star, z):

    """
    Return for each position z along the line of sight the distance between this position and the centre of the star (cm)

    Parameters:
        L       : distance to the gamma-source (cm)
        D_star  : distance to the star (cm)
        z       : position along the line of sight (cm)
    """

    #Angular opening between the gamma source and the star (rad)
    def angle_dzeta(b, D_star):

        return np.arcsin(b/D_star)

    dzeta = angle_dzeta(b, D_star)

    return np.sqrt((L - z)**2 + D_star**2 - 2 * (L - z) * D_star * np.cos(dzeta))

def angle_alpha(b, D, z, zb, theta, phi):

    """
    Return the cosinus of the angle (alpha) between the two momenta of the two photons in the observer's frame

    Parameters :
        b     : impact parameter (cm)
        D     : distance to the star from each position along the line of sight (cm)
        z     : position along the line of sight (cm)
        zb    : position along the line of sight nearly the star (cm)
        theta : angle formed by the ray (from the star) and the straight line connecting the centre of the star and a position z on the line of sight (rad)
            careful theta = 0 when the ray 'comes' from the centre of the star and theta = theta_max when the ray is tangent to the sphere
        phi   : angle around the direction between the centre of the star and the position along the line of sight (rad)
    """

    #Angle formed by the direction between the centre of the star and the position along the line of sight, and the line of sight (rad)
    def angle_beta(b, D, z, zb):

        if z <= zb:
            beta = np.pi - np.arcsin(b/D)

        else:
            beta = np.arcsin(b/D)

        return beta

    beta = angle_beta(b, D, z, zb)

    return  np.sin(beta) * np.sin(theta) * np.sin(phi) - np.cos(beta) * np.cos(theta)

def f_ang_theta(theta, phi, eps, z, L, D_star, b, R, E, T, zb):

    """
    Return the function for the integration in theta : f = dn * sigma * (1 - cos(alpha))
        where dn is the density of photons, sigma is the dimensionless cross section of the interaction,
              alpha is the between the two momenta of the two photons in the observer's frame

    Parameters:
        theta   : angle formed by the ray (from the star) and the straight line connecting the centre of the star and a position z on the line of sight (rad)
        phi     : angle around the direction between the centre of the star and the position along the line of sight (rad)
        eps     : energy of the target-photon (keV)
        z       : position along the line of sight (cm)
        L       : the distance to the gamma-source (cm)
        D_star  : distance to the star (cm)
        b       : impact parameter (cm)
        E       : energy of the gamma-photon (keV)
        T       : temperature of the star (K)
        zb      : position along the line of sight nearly the star (cm)
    """

    #Global constants
    mc2 = 510.9989461 #electron mass (keV)

    D = distance(L, D_star, z)

    cos_alpha = angle_alpha(b, D, z, zb, theta, phi)
    epsc = np.sqrt(eps * E/2 * (1 - cos_alpha))/mc2 # epsc in keV/mc2

    #First : sigma (cm^2)
    def cross_section(epsc):

        def parameter_s(epsc):

            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        beta = np.nan_to_num(beta)

        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    dn = density_n(eps, T, theta)
    sigma = cross_section(epsc)

    return dn * sigma * (1 - cos_alpha)

def f_ang_phi(phi, eps, z, L, D_star, b, R, E, T, zb):

    """
    Return the function for the integration in phi : integral from theta = 0 to theta of f_ang_theta

    Parameters:
        phi   : angle around the direction between the centre of the star and the position along the line of sight (rad)
        eps   : energy of the target-photon (keV)
        z     : position along the line of sight (cm)
        L     : the distance to the gamma-source (cm)
        D_star  : distance to the star (cm)
        b     : impact parameter (cm)
        E     : energy of the gamma-photon (keV)
        T     : temperature of the star (K)
        zb    : position along the line of sight nearly the star (cm)
    """

    D = distance(L, D_star, z)
    theta_max = np.arcsin(R/D)

    return quad(f_ang_theta, 0, theta_max, args = (phi, eps, z, L, D_star, b, R, E, T, zb))[0]

def f_eps(eps, z, L, D_star, b, R, E, T, zb):

    """
    Return the function for the integration on epsmin = mc2/E to 10*k*T of f_eps = quad(f_ang_phi, 0, 2*np.pi, args = (eps, z, L, D_star, b, R, E, D))

    Parameters:
        eps   : energy of the target-photon (keV)
        z     : position along the line of sight (cm)
        L     : the distance to the gamma-source (cm)
        D_star  : distance to the star (cm)
        b     : impact parameter (cm)
        E     : energy of the gamma-photon (keV)
        T     : temperature of the star (K)
        zb    : position along the line of sight nearly the star (cm)
    """

    #print(f_ang_phi)

    return quad(f_ang_phi, 0, 2 * np.pi, args = (eps, z, L, D_star, b, R, E, T, zb))[0]

def f_z(z, L, D_star, b, R, E, T, zb):

    """
    Return the function for the integration on z from 0 to L: f_z = int_epsmin^epsmax f_eps

    Parameters:
        z     : position along the line of sight (cm)
        L     : the distance to the gamma-source (cm)
        D_star  : distance to the star (cm)
        b     : impact parameter (cm)
        E     : energy of the gamma-photon (keV)
        T     : temperature of the star (K)
        zb    : position along the line of sight nearly the star (cm)
    """

    #energy of the target-photon
    epsmin = mc2**2/E
    epsmax = 10*k*T

    return quad(f_eps, epsmin, epsmax, args = (z, L, D_star, b, R, E, T, zb))[0]

def calculate_tau(L, D_star, b, R, E, T, zb):
    """
    Return the optical depth : tau = int_L dx int_eps d eps int_0^{2*pi} d phi int_0^{theta_max} dn * sigma_int * (1 - cos(alpha))

    Parameters:
        L     : the distance to the gamma-source (cm)
        D_star  : distance to the star (cm)
        b     : impact parameter (cm)
        E     : energy of the gamma-photon (keV)
        T     : temperature of the star (K)
        zb    : position along the line of sight nearly the star (cm)
    """

    #Global constants
    r0 =  2.818e-13 #classical electron radius (cm)

    return  1/2.0 * np.pi * r0**2 * quad(f_z, 0, L, args = (L, D_star, b, R, E, T, zb))[0]

#Global constants
conv_l = 3.085678e21     #conversion factor from kpc to cm
conv_en = 1.602e-16      #conversion factor from J to keV
c = 2.99792458e+10       #light speed in cm/s
k = 1.380658e-23/conv_en # Boltzmann's constant in keV/K
mc2 = 510.9989461        #electron mass (keV)

#For the vector eps, E
number_bin_E = 80
number_bin_eps = 40.0

#Parameters for the code
L = 20 * conv_l         # the distance to the gamma-source (cm)
D_star =  8.5 * conv_l  # distance to the star (from us) (cm)
zb = 10 * conv_l        # position along the line of sight nearly the star (cm)
b = 5 * conv_l          # impact parameter (cm)
Rsun = 6.957e10         # radius of the Sun (cm)
R = 1000 * Rsun         # radius of the star (express in Rsun)
T = 25                  # temperature of the star (K)

#energy of the gamma-photon
Emin = 1e8       #Emin = 10^-1 TeV (keV)
Emax = 1e14      #Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E)  # keV
E_tev = E*1e-9   # TeV
tau = np.zeros_like(E)

#for i in range (len(E)):

epsmin = (mc2)**2/E[len(E)-1]
epsmax = 10*k*T

#if epsmin > epsmax:
#    continue

tau = calculate_tau(L, D_star, b, R, E[len(E)-1], T, zb) # optical depth

D_star_kpc = D_star/conv_l     #in kpc
plt.plot(E_tev[len(E_tev)-1], np.exp(-tau), label = "D$_{star}$ = %.2f kpc" %D_star_kpc)

D_star_kpc = D_star_kpc/conv_l #in kpc
L_kpc = L/conv_l #in kpc

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with IR photons of a ponctual source at %.2f K' %T)
plt.text(100, 1,'D' r'$_{star}$' ' = %.2f kpc, L = %.2f kpc' %(D_star_kpc, L_kpc))
plt.legend()
plt.show()
