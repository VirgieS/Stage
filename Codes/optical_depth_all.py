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

##=========##
# Functions #
##=========##

def tau_anistropic(E, z, phi, zb, L, D_star, b, R, T):

    """
    Return the optical depth for a anistropic density_n_cst

    Parameters :
        E       : Energy of the gamma photon (keV)
        z       : position along the line of sight (cm)
        phi     : angle around the ray (rad)
        zb      : position along the line of sight closely the secundary source (cm)
        L       : lenght of the line of sight (cm)
        D_star  : distance between the star and the observator (cm)
        b       : impact parameter (cm)
        R       : radius of the star (cm)
        T       : temperature of the star (K)
    """

    def density_n_cst(eps, T, theta):

        """
        Density of the photons is not isotropic : dn = Bnu * cos(theta)/(c * h**2 *nu)

        Parameters:
            eps   : energy of the target-photon (keV)
            theta : angle formed by the ray (from the star) and the line connecting the centre of the star and a position z on the line of sight (rad)
            T     : temperature of the star (K)
        """

        nu = eps/(hp*erg2kev)

        def planck(nu, T):

            return (2*(hp*erg2kev)*(nu**3))/(cl**2) * 1/(np.exp(((hp*erg2kev)*nu)/((kb*erg2kev)*T)) - 1)

        Bnu = planck(nu, T)

        return Bnu/(cl * (hp*erg2kev)**2 * nu) # return dn in cm^3/sr/keV

    def density_n(eps, T, theta):

        """
        Density of the photons is not isotropic : dn = Bnu * cos(theta)/(c * h**2 *nu)

        Parameters:
            eps   : energy of the target-photon (keV)
            theta : angle formed by the ray (from the star) and the line connecting the centre of the star and a position z on the line of sight (rad)
            T     : temperature of the star (K)
        """

        nu = eps/(hp*erg2kev)

        def planck(nu, T):

            return (2*(hp*erg2kev)*(nu**3))/(cl**2) * 1/(np.exp(((hp*erg2kev)*nu)/((kb*erg2kev)*T)) - 1)

        Bnu = planck(nu, T)

        return Bnu/(cl * (hp*erg2kev)**2 * nu) * np.cos(theta) #return dn in cm^3/sr/keV

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

        #Angle formed by the direction between the centre of the star and the position along the line of sight, and the line of sight (rad)
        def angle_beta(b, D, z, zb):

            if z <= zb:
                beta = np.arcsin(b/D)

            else:
                beta = np.pi - np.arcsin(b/D)

            return beta

        beta = angle_beta(b, D, z, zb)

        return  - np.sin(beta) * np.sin(theta) * np.sin(phi) - np.cos(beta) * np.cos(theta)

    def f(theta, phi, eps, z, L, b, R, E, T, zb):

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
            D_star  : distance to the star (cm)
            b       : impact parameter (cm)
            E       : energy of the gamma-photon (keV)
            T       : temperature of the star (K)
            zb      : position along the line of sight nearly the star (cm)
        """

        D = distance(zb, z, b)

        cos_alpha = angle_alpha(b, D, z, zb, theta, phi)
        epsc = np.sqrt(eps * E/2 * (1 - cos_alpha))/mc2 # epsc in keV/mc2

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

    def f_cst(theta, phi, eps, z, L, b, R, E, T, zb):

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
            D_star  : distance to the star (cm)
            b       : impact parameter (cm)
            E       : energy of the gamma-photon (keV)
            T       : temperature of the star (K)
            zb      : position along the line of sight nearly the star (cm)
        """

        D = distance(zb, z, b)

        cos_alpha = angle_alpha(b, D, z, zb, theta, phi)
        epsc = np.sqrt(eps * E/2 * (1 - cos_alpha))/mc2 # epsc in keV/mc2

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

        dn = density_n_cst(eps, T, theta)
        sigma = cross_section(epsc)

        return dn * sigma * (1 - cos_alpha) * np.sin(theta)

    def calculate_tau(E, z, phi, zb, L, D_star, b, R, T):

        integral_eps = np.zeros_like(z)

        # Energy of the target-photon
        epsmin = mc2**2/E
        epsmax = 10*(kb*erg2kev)*T

        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

        for j in range (len(z)): # integration over eps

            integral_theta = np.zeros_like(eps)
            D = distance(zb, z[j], b)
            theta_max = np.arcsin(R/D)
            theta = np.linspace(0, theta_max, 10)

            for l in range (len(eps)): # integration over theta
                integral_phi = np.zeros_like(theta)

                for m in range (len(theta)): # integration over phi

                    integrand = f(theta[m], phi, eps[l], z[j], L, b, R, E, T, zb)
                    integrand = np.nan_to_num(integrand)

                    idx=(integrand > 0.0)

                    integral_phi[m] = integration_log(phi, integrand)

                integral_theta[l] = integration_log(theta, integral_phi)

            integral_eps[j] = integration_log(eps, integral_theta) #you get d(tau)/dx

        return  1/2.0 * np.pi * r0**2 * integral_eps

    tau = calculate_tau(E, z, phi, zb, L, D_star, b, R, T)

    return tau

def tau_cste(E, T, z, b, zb, R):

    def density_n(eps, T, R, b):

        """
        Density of the photons at zb : dn = 2 * pi * int_0^theta_max Bnu * cos(theta)/(c * h**2 *nu) * sin(theta)

        Parameters:
            eps     : energy of the target-photon (keV)
            T       : temperature of the star (K)
            R       : radius of the star (cm)
            b       : impact parameter (cm)
        """

        nu = eps/(hp*erg2kev)

        def planck(nu, T):

            return (2*(hp*erg2kev)*(nu**3))/(cl**2) * 1/(np.exp(((hp*erg2kev)*nu)/((kb*erg2kev)*T)) - 1)

        Bnu = planck(nu, T)

        theta_max = np.arcsin(R/b)
        theta = np.linspace(0, theta_max, 10)

        dn_phi = Bnu/(cl * (hp*erg2kev)**2 * nu)  * np.cos(theta) * np.sin(theta)

        integral = integration_log(theta, dn_phi)

        return 2 * np.pi * integral #return dn in cm^3/keV

    def distance(zb, z, b):

        """
        Return for each position z along the line of sight the distance between this position and the centre of the star (cm)

        Parameters:
            zb       : position on the line of sight closely the star (cm)
            z        : position along the line of sight (cm)
            b        : impact parameter
        """

        return np.sqrt((zb - z)**2 + b**2)

    def angle_alpha(b, D, z, zb):

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
                beta = np.arcsin(b/D)

            else:
                beta = np.pi - np.arcsin(b/D)

            return beta

        beta = angle_beta(b, D, z, zb)

        return  np.pi - beta

    def f(b, z, zb, eps, E, T, R):

        """
        Return the function for the integration over eps : f = dn * sigma * (1 - cos(alpha))
            where dn is the density of photons, sigma is the dimensionless cross section of the interaction,
                  alpha is the between the two momenta of the two photons in the observer's frame

        Parameters:
            b       : impact parameter (cm)
            z       : position along the line of sight (cm)
            zb      : position along the line of sight nearly the star (cm)
            eps     : energy of the target-photon (keV)
            E       : energy of the gamma-photon (keV)
            T       : temperature of the star (K)
            R       : radius of the star (cm)
        """

        D = distance(zb, z, b)

        alpha = angle_alpha(b, D, z, zb)
        cos_alpha = np.cos(alpha)
        epsc = np.sqrt(eps * E/2 * (1 - cos_alpha))/mc2 # epsc in keV/mc2

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

        dn = density_n(eps, T, R, b)
        sigma = cross_section(epsc)

        return dn * sigma * (1 - cos_alpha)

    def calculate_tau(E, T, z, b, zb, R):

        """
        Return the value of (d tau)/dz : int_{(mc^2)^2/E}^{infty} f

        Parameters:
            E       : energy of the gamma-photon (keV)
            T       : temperature of the star (K)
            z       : position along the line of sight (cm)
            b       : impact parameter (cm)
            zb      : position along the line of sight nearly the star (cm)
            R       : radius of the star (cm)
        """

        integral_eps = np.zeros_like(z)

        # Energy of the target-photon
        epsmin = mc2**2/E
        epsmax = 10*(kb*erg2kev)*T

        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

        for i in range (len(z)):

            integrand = np.zeros_like(eps)

            for j in range (len(eps)):

                integrand[j] = f(b, z[i], zb, eps[j], E, T, R)

            integral_eps[i] = integration_log(eps, integrand) #you get d(tau)/dx

        return  1/2.0 * np.pi * r0**2 * integral_eps

    tau = calculate_tau(E, T, z, b, zb, R)

    return tau

##================================##
# Computation of the optical depth #
##================================##

if __name__ == '__main__':

    # For the vector eps
    number_bin_eps = 40.0

    # Parameters for the code
    L = 20 * AU2cm                          # the distance to the gamma-source (cm)
    zb = 10 * AU2cm                         # position along the line of sight nearly the star (cm)
    b = 5 * AU2cm                           # impact parameter (cm)
    D_star =  np.sqrt(b**2 + (L - zb)**2)   # distance to the star (from us) (cm)
    D_gamma = np.sqrt(b**2 + L**2)          # distance between the star and the gamma-source (cm)
    R = 0.5 * AU2cm                         # radius of the star (express in Rsun)
    T = 10000                               # temperature of the star (K)
    z = np.linspace(0, L, 100)              # position along the line of sight (cm)
    phi = np.linspace(0, 2*np.pi, 10)       # polar angle (rad)

    # Energy of the gamma-photon
    E = 1e9                                 # keV
    E_tev = E*keV2eV/TeV2eV                 # TeV


    # Calculation of the transmittance
    tau = tau_anistropic(E, z, phi, zb, L, D_star, b, R, T)
    tau_cst = tau_cste(E, T, z, b, zb, R)
    z_au = z/AU2cm                          # au
    zb_au = zb/AU2cm                        # au
    b_au = b/AU2cm                          # au

    f = plt.figure()
    ax = f.add_subplot(111)

    plt.plot(z_au, tau, label = "anistropic density")
    plt.plot(z_au, tau_cst, label = "uniform density")

    D_star_au = D_star/AU2cm                # au
    L_au = L/AU2cm                          # au
    R_au = R/AU2cm                          # au

    plt.xlabel(r'z (au)')
    plt.ylabel(r'$\frac{d \tau_{\gamma \gamma}}{d z}$' ' ' r'$(cm^{-1})$' )
    #plt.title(u'Optical depth for the interaction between 'r'$\gamma$' '-rays at %.2f GeV \n and photons of a star at %.2f K and a radius %.2f au' %(E_gev, T, R_au))
    plt.text(0.65, 0.5, u'D$_{star}$ = %.2f au, L = %.2f au \nz$_b$ = %.2f au, b = %.2f au \n E$_\gamma$=%.2f TeV' %(D_star_au, L_au, zb_au, b_au, E_tev), horizontalalignment='left',
     verticalalignment='center', transform = ax.transAxes)

    plt.legend(loc='best')
    plt.show()
