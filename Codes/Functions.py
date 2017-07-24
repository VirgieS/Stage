#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factor import *

#function to integrate a function in log-log scale
def integration_log(x, y):

    #Looking for a and b for y = a*x^b
    def calculate_ab(xi, xf, yi, yf):
        logxi = np.log(xi)
        logxf = np.log(xf)
        logyi = np.log(yi)
        logyf = np.log(yf)
        b = (logyf - logyi)/(logxf - logxi)
        loga = logyi - b*logxi
        a = np.exp(loga)
        a = np.nan_to_num(a)
        return a, b

    #Calculate deltaS from deltaS = int from xi to xf a*x^b
    def delta_S(xi, xf, yi, yf):
        [a, b] = calculate_ab(xi, xf, yi, yf)
        return a/(b+1)*(xf**(b+1) - xi**(b+1))

    integral = 0

    # Because the function integral_log works only if there is more than two elements not zero
    idx=(y > 0.0)
    #idy=(y < 0.0)
    idt = idx #+ idy
    if sum(idt) > 2:

        x = x[idt]
        y = y[idt]

        #Calculate total integral from init to final a*x^b
        deltaS = 0

        for i in range (1, len(x)):
            deltaS = delta_S(x[i-1], x[i], y[i-1], y[i])
            integral = integral + deltaS

            integral = np.nan_to_num(integral)

    return integral

# for WD : compute z_WD, b_WD
def compute_WD(psi_gamma, psi_o, phi_gamma, phi_o, r_gamma, theta_max_WD):

    """
    Return z_RG and b_RG

    Parameters :
        psi_gamma       : colatitude of the gamma-source (rad)
        psi_o           : colatitude of the observator (rad)
        phi_gamma       : polar angle of the gamma-source (rad)
        phi_o           : polar angle of the observator (rad)
        r_gamma         : distance to the gamma-source (cm)
    """

    condition_WD = False
    gamma_WD = np.arccos(-(np.sin(psi_gamma)*np.sin(psi_o)*np.cos(phi_gamma)*np.cos(phi_o) + np.sin(psi_gamma)*np.sin(psi_o)*np.sin(phi_gamma)*np.sin(phi_o) + np.cos(psi_gamma)*np.cos(psi_o)))
    b_WD = r_gamma * np.sin(gamma_WD)

    if gamma_WD <= np.pi/2:

        z_WD = np.sqrt(r_gamma**2 - b_WD**2)

    else:

        z_WD = - np.sqrt(r_gamma**2 - b_WD**2)

    if gamma_WD <= theta_max_WD: # if there is an eclispe

            condition_WD = True

    return b_WD, z_WD, condition_WD

# for WD : compute z_RG, b_RG
def compute_RG(psi_gamma, psi_o, phi_gamma, phi_o, r_gamma, d_orb, theta_max_RG):

    """
    Return z_RG and b_RG

    Parameters :
        psi_gamma       : colatitude of the gamma-source (rad)
        psi_o           : colatitude of the observator (rad)
        phi_gamma       : polar angle of the gamma-source (rad)
        phi_o           : polar angle of the observator (rad)
        r_gamma         : distance to the gamma-source (cm)
        d_orb           : orbital separation (cm)
    """

    condition_RG = False

    gamma_RG = np.arccos((d_orb*np.sin(psi_o)*np.cos(phi_o) - r_gamma * (np.sin(psi_gamma)*np.sin(psi_o)*np.cos(phi_gamma)*np.cos(phi_o) + np.sin(psi_gamma)*np.sin(psi_o)*np.sin(phi_gamma)*np.sin(phi_o) + np.cos(psi_gamma)*np.cos(psi_o)))/(np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(psi_gamma)*np.cos(phi_gamma) + r_gamma**2)))

    b_RG = sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(psi_gamma)*np.sin(phi_gamma) + r_gamma**2) * np.sin(gamma_RG)

    if gamma_RG <= np.pi/2:

        z_RG = np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(psi_gamma)*np.sin(phi_gamma) + r_gamma**2- b_RG**2)

    else:

        z_RG = - np.sqrt(r_gamma**2 - b_RG**2)

    if gamma_RG <= theta_max_RG: # if there is an eclispe

            condition_RG = True

    return b_RG, z_RG, condition_RG

def distance(zb, z, b):

    """
    Return for each position z along the line of sight the distance between this position and the centre of the star (cm)

    Parameters:
        zb       : position on the line of sight closely the star (cm)
        z        : position along the line of sight (cm)
        b        : impact parameter
    """

    return np.sqrt((zb - z)**2 + b**2)

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

    #Return the cosinus of the angle formed by the direction between the centre of the star and the position along the line of sight, and the line of sight (rad)
    def angle_beta(b, D, z, zb):

        if z <= zb:
            beta = np.arcsin(b/D)

        else:
            beta = np.pi - np.arcsin(b/D)

        return beta

    beta = angle_beta(b, D, z, zb)

    return  - np.sin(beta) * np.sin(theta) * np.sin(phi) - np.cos(beta) * np.cos(theta)

def density_n(eps, T, theta):

    """
    Density of the photons is not isotropic : dn = Bnu * cos(theta)/(c * h**2 *nu)

    Parameters:
        eps   : energy of the target-photon (keV)
        theta : angle formed by the ray (from the star) and the line connecting the centre of the star and a position z on the line of sight (rad)
        T     : temperature of the star (K)
    """

    nu = eps/hp

    def planck(nu, T):

        return (2*hp*(nu**3))/(cl**2) * 1/(np.exp((hp*nu)/(kb*T)) - 1)

    Bnu = planck(nu, T)

    return Bnu/(cl * hp**2 * nu) * np.cos(theta) #return dn in cm^3/sr/erg


def f(theta, phi, eps, z, D, b, R, E, T, zb):

    """
    Return the function for the integration in phi : f = dn * sigma * (1 - cos(alpha)) * sin(theta)
        where dn is the density of photons, sigma is the dimensionless cross section of the interaction,
              alpha is the between the two momenta of the two photons in the observer's frame

    Parameters:
        theta   : angle formed by the ray (from the star) and the straight line connecting the centre of the star and a position z on the line of sight (rad)
        phi     : angle around the direction between the centre of the star and the position along the line of sight (rad)
        eps     : energy of the target-photon (keV)
        z       : position along the line of sight (cm)
        L       : the distance to the gamma-source (cm)
        b       : impact parameter (cm)
        E       : energy of the gamma-photon (keV)
        T       : temperature of the star (K)
        zb      : position along the line of sight nearly the star (cm)
    """

    cos_alpha = angle_alpha(b, D, z, zb, theta, phi)
    epsc = np.sqrt(eps * E/2 * (1 - cos_alpha))/(mc2/ergkev) # epsc/mc2 in erg

    #First : sigma (dimensionless)
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

    return dn * sigma * (1 - cos_alpha) * np.sin(theta)

def calculate_tau(E, z, phi, b, R, T, zb):

    """
    Return tau(E) for all E
    Parameters :
        E           : energy of the gamma-photon (erg)
        z           : position along the line of sight (cm)
        phi         : angle around the direction between the centre of the star and the position along the line of sight (rad)
        zb          : position along the line of sight nearly the star (cm)
        L           : maximum distance for the integration about z (cm)
        b           : impact parameter (cm)
        R           : radius of the secundary source (cm)
        T           : temperature of the secundary source (cm)
    """

    integral = np.zeros_like(E)
    number_bin_eps = 40.0

    for i in range(len(E)): # integration over z

        integral_eps = np.zeros_like(z)

        # Energy of the target-photon (erg)
        epsmin = (mc2/ergkev)**2/E[i]
        epsmax = 10*kb*T

        # Because epsmin must be lower than epsmax
        if epsmin > epsmax:
            continue
        else:
            eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))
            print(i)

        for j in range (len(z)): # integration over eps

            integral_theta = np.zeros_like(eps)
            D = distance(zb, z[j], b)
            theta_max = np.arcsin(R/D)
            theta = np.linspace(0, theta_max, 10)

            for l in range (len(eps)): # integration over theta

                integral_phi = np.zeros_like(theta)

                for m in range (len(theta)): # integration over phi

                    integrand = f(theta[m], phi, eps[l], z[j], D, b, R, E[i], T, zb)
                    integrand = np.nan_to_num(integrand)

                    integral_phi[m] = integration_log(phi, integrand)

                integral_theta[l] = integration_log(theta, integral_phi)

            integral_eps[j] = integration_log(eps, integral_theta) #you get d(tau)/dx

        integral[i] = integration_log(z, integral_eps)

    return  1/2.0 * np.pi * r0**2 * integral

def calculate_tau_L(E, z, phi, b, R, T, zb):

    """
    Return tau(E) for all E
    Parameters :
        E           : energy of the gamma-photon (erg)
        z           : position along the line of sight (cm)
        phi         : angle around the direction between the centre of the star and the position along the line of sight (rad)
        zb          : position along the line of sight nearly the star (cm)
        L           : maximum distance for the integration about z (cm)
        b           : impact parameter (cm)
        R           : radius of the secundary source (cm)
        T           : temperature of the secundary source (cm)
    """

    integral = np.zeros_like(E)
    number_bin_eps = 20.0
    #step_theta = 0.01

    # integration over z

    integral_eps = np.zeros_like(z)

    # Energy of the target-photon (erg)
    epsmin = (mc2/ergkev)**2/E
    epsmax = 10*kb*T

    eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

    for j in range (len(z)): # integration over eps

        integral_theta = np.zeros_like(eps)
        D = distance(zb, z[j], b)
        theta_max = np.arcsin(R/D)
        theta = np.linspace(0, theta_max, 10)

        for l in range (len(eps)): # integration over theta

            integral_phi = np.zeros_like(theta)

            for m in range (len(theta)): # integration over phi

                integrand = f(theta[m], phi, eps[l], z[j], D, b, R, E, T, zb)
                integrand = np.nan_to_num(integrand)

                integral_phi[m] = integration_log(phi, integrand)

            integral_theta[l] = integration_log(theta, integral_phi)

        integral_eps[j] = integration_log(eps, integral_theta) #you get d(tau)/dx

    integral = integration_log(z, integral_eps)

    return  1/2.0 * np.pi * r0**2 * integral
