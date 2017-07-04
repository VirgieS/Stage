"""
We consider that the star, not along the line of sight, has no dimension and has a uniform brightness (black body at T).

Our coordinate are z, the position along the line of sight and psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

Parameters that we can give for this code are:
    L      : distance to the gamma-source (au)
    zb    : position along the line of sight nearly the star (au)
    b      : impact parameter (au)
    T      : temperature of the star (K)
"""

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

def distance(zb, z, b):

    """
    Return for each position z along the line of sight the distance between this position and the centre of the star (cm)

    Parameters:
        zb       : position on the line of sight closely the star (cm)
        z        : position along the line of sight (cm)
        b        : impact parameter
    """

    return np.sqrt((zb - z)**2 + b**2)

def density_n(u_in, b, d, eps, T):

    """
    Return the density of the photons is not isotropic and is : dn = u/(a*T^4) * Bnu/(c*h*nu)

    Parameters:

    u_in    : choosen energy density of the (ponctual) star in a position along the line of sight (keV/cm^3)
    b       : impact parameter (cm)
    d       : distance between the gamma-photon and the source IR (cm)
    eps     : energy of the target-photon (keV)
    T       : temperature of the source (K)
    """

    #Global constant
    h = 6.6260755e-34/conv_en   # Planck's constant in keV*s

    def energy_density(u_in, b, d, T): #energy density of the source IR

        #Global constants
        c = 2.99792458e+10          # light speed in cm/s
        sb = 5.670367e-12/conv_en   # Stefan-Boltzmann constant (keV s^-1 cm^-2 K^-4)
        a = 4 * sb/c                # radiation constant (keV cm^-3 K^-4)
        h = 6.6260755e-34/conv_en   # Planck's constant in keV*s

        def luminosity(u_in, b): # luminosity of the source IR

            c = 2.99792458e+10  # light speed in cm/s

            return u_in * c * 4 * np.pi * b**2

        Lum = luminosity(u_in, b)

        return Lum/(4 * np.pi * d**2 * c) * 1.0/(a * T**4)

    nu = eps/h

    def planck(nu, T):

        #Global constants
        k = 1.380658e-23/conv_en #Boltzmann's constant in keV/K
        c = 2.99792458e+10 #light speed in cm/s

        return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(k*T)) - 1)

    Bnu = planck(nu, T)

    u = energy_density(u_in, b, d, T)

    return Bnu/(h**2 * c * nu) * u

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

    #Calculate total integral from init to final a*x^b
    integral = 0
    deltaS = 0

    for i in range (1, len(x)):
        deltaS = delta_S(x[i-1], x[i], y[i-1], y[i])
        integral = integral + deltaS

    integral = np.nan_to_num(integral)

    return integral

#angle (rad) between the momenta of the two photons (gamma and IR) in the observer's frame
def angle(d, b, z, zb): # d, b, z and zb must be given in cm

    if z <= zb:
        theta = np.pi - np.arcsin(b*1.0/d)

    else:
        theta = np.arcsin(b*1.0/d)

    return theta

def f(u_in, b, d, eps, T, z, zb, E):

    """
    Return the function for the integration on epsilon : dn * sigma * (1 - cos(theta))
        where dn is the density of photons, sigma is the dimensionless cross section of the interaction,
              theta is the between the two momenta of the two photons in the observer's frame

    Parameters:
    u_in    : choosen energy density of the (dimensionless) star at a position along the line of sight (keV/cm^3)
    b       : impact parameter (cm)
    d       : distance between the gamma-photon and the star (cm)
    eps     : energy of the target-photon (keV)
    T       : temperature of the star (K)
    z       : position along the line of sight (cm)
    zb      : position on the line of sight closely the star (cm)
    E       : energy of the gamma-photon (keV)
    """

    mc2 = 510.9989461 #electron mass (keV)

    theta = angle(d, b, z, zb)

    epsc = np.sqrt(eps * E/2 * (1 - np.cos(theta)))/mc2 # it gives epsc in keV/mc2

    def cross_section(epsc): # sigma_int  = 1/2.0 * pi * r0**2 * sigma (equation 1 of Gould's article), we calculate sigma and not sigma_int

        def parameter_s(epsc):

            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        beta = np.nan_to_num(beta)

        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    dn = density_n(u_in, b, d, eps, T)
    sigma = cross_section(epsc)
    return dn * sigma * (1 - np.cos(theta))

def calculate_tau(E, u_in, b, T, z, zb):

    # Because the source is dimensionless the integration over the solid angle gives 4*pi.

    # Global constants
    k = 1.380658e-23/conv_en #Boltzmann's constant in keV/K
    r0 =  2.818e-13 #classical electron radius (cm)

    integral = np.zeros_like(E)

    for i in range(len(E)): # integration over z

        integral_eps = np.zeros_like(z)

        #energy of the target-photon
        epsmin = mc2**2/E[i]
        epsmax = 10*k*T

        # epsmin must be lower than epsmax
        if epsmin > epsmax:
            continue
        else:
            eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))
            print(i)
            print(int(log10(epsmax/epsmin)*number_bin_eps))
            print(len(eps))

        for j in range(len(z)): # integration over eps

            d = distance(zb, z[j], b)

            integrand = f(u_in, b, d, eps, T, z[j], zb, E[i])
            integrand = np.nan_to_num(integrand)

            idx=(integrand > 0.0)

            #Because the function integral_log works only if there is more than two elements not zero
            if sum(idx) > 2:
                integral_eps[j] = integration_log(eps[idx], integrand[idx])

        #Because the function integral_log works only if there is more than two elements not zero
        idx=(integral_eps > 0.0)
        if sum(idx) > 2:
            integral[i] = integration_log(z[idx], integral_eps[idx])

    return  1/2.0 * np.pi * r0**2 * 4 * np.pi * integral

#Global constants
conv_l = 3e21      # conversion factor from au to cm
conv_en = 1.602e-16      # conversion factor from J to keV
c = 2.99792458e+10       # light speed in cm/s
k = 1.380658e-23/conv_en # Boltzmann's constant in keV/K
mc2 = 510.9989461        # electron mass (keV)


#For the vector eps, E
number_bin_E = 80
number_bin_eps = 40.0

#Parameters for the code
L = 20 * conv_l                         # the distance to the gamma-source (cm)
zb = 10 * conv_l                        # position along the line of sight nearly the star (cm)
b = 5 * conv_l                          # impact parameter (cm)
D_star =  np.sqrt(b**2 + (L - zb)**2)   # distance to the star (from us) (cm)
D_gamma = np.sqrt(b**2 + L**2)          # distance between the star and the gamma-source (cm)
T = 10000                                # temperature of the star (K)
z = np.linspace(0, L, 100)              # position along the line of sight (cm)
u_in = 1e-3                             # energy density at the impact parameter (keV/cm^3)


# Energy of the gamma-photon
Emin = 1e7          # Emin = 10^-1 TeV (keV)
Emax = 1e14         # Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E) # keV
E_tev = E*1e-9 #TeV

tau = calculate_tau(E, u_in, b, T, z, zb)
b_au = b/conv_l # in au
D_star_au = D_star/conv_l # in au

plt.plot(E_tev, np.exp(-tau), label = "b = %.2f kpc" %b_au)

D_star_au = D_star/conv_l # in au
L_au = L/conv_l # in au

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with IR photons of a ponctual source at %.2f K' %T)
plt.text(100, 1,'D$_{star}$ = %.2f kpc, L = %.2f kpc' %(D_star_au, L_au))
plt.legend()
plt.show()
