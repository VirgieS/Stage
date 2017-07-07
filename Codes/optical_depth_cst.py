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

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

#Important functions for the calculation

def density_n(eps, T, R, b):

    """
    Density of the photons at zb : dn = 2 * pi * int_0^theta_max Bnu * cos(theta)/(c * h**2 *nu) * sin(theta)

    Parameters:
        eps     : energy of the target-photon (keV)
        T       : temperature of the star (K)
        R       : radius of the star (cm)
        b       : impact parameter (cm)
    """

    #Global constants
    h = 6.6260755e-34/conv_en #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s

    nu = eps/h

    def planck(nu, T):

        #Global constants
        kb = 1.380658e-23/conv_en #Boltzmann's constant in keV/K
        c = 2.99792458e+10 #light speed in cm/s

        return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(kb*T)) - 1)

    Bnu = planck(nu, T)

    theta_max = np.arcsin(R/b)
    theta = np.linspace(0, theta_max, 10)

    dn_phi = Bnu/(c * h**2 * nu) * np.cos(theta) * np.sin(theta)

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

    # Because the function integral_log works only if there is more than two elements not zero
    idx=(y > 0.0)
    if sum(idx) > 2:

        x = x[idx]
        y = y[idx]

        #Calculate total integral from init to final a*x^b
        integral = 0
        deltaS = 0

        for i in range (1, len(x)):
            deltaS = delta_S(x[i-1], x[i], y[i-1], y[i])
            integral = integral + deltaS

            integral = np.nan_to_num(integral)

    return integral

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

    #Global constants
    mc2 = 510.9989461 #electron mass (keV)

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

    #Global constants
    r0 =  2.818e-13 #classical electron radius (cm)
    kb = 1.380658e-23/conv_en # Boltzmann's constant in keV/K

    integral_eps = np.zeros_like(z)

    # Energy of the target-photon
    epsmin = mc2**2/E
    epsmax = 10*kb*T

    eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

    for i in range (len(z)):

        integrand = np.zeros_like(eps)

        for j in range (len(eps)):

            integrand[j] = f(b, z[i], zb, eps[j], E, T, R)

        integral_eps[i] = integration_log(eps, integrand) #you get d(tau)/dx

    return  1/2.0 * np.pi * r0**2 * integral_eps

# Global constants
conv_l = 1.45979e13      # Conversion factor from au to cm
conv_en = 1.602e-16      # Conversion factor from J to keV
c = 2.99792458e+10       # Light speed in cm/s
kb = 1.380658e-23/conv_en # Boltzmann's constant in keV/K
mc2 = 510.9989461        # Electron mass (keV)

# For the vector eps
number_bin_eps = 40.0

# Parameters for the code
L = 20 * conv_l                         # the distance to the gamma-source (cm)
zb = 10 * conv_l                        # position along the line of sight nearly the star (cm)
b = 5 * conv_l                          # impact parameter (cm)
D_star =  np.sqrt(b**2 + (L - zb)**2)   # distance to the star (from us) (cm)
D_gamma = np.sqrt(b**2 + L**2)          # distance between the star and the gamma-source (cm)
R = 0.5 * conv_l                        # radius of the star (express in Rsun)
T = 10000                               # temperature of the star (K)
z = np.linspace(0, L, 100)              # position along the line of sight (cm)

# Energy of the gamma-photon
E = 1e9  # keV
E_tev = E*1e-9   # TeV


# Calculation of the transmittance
tau = calculate_tau(E, T, z, b, zb, R)
z_au = z/conv_l # in au
zb_au = zb/conv_l #in au

b_au = b/conv_l # in au
plt.plot(z_au, tau, label = "b = %.2f, zb = %.2f au" %(b_au, zb_au))

D_star_au = D_star/conv_l # in au
L_au = L/conv_l # in au
R_au = R/conv_l #in au

plt.xlabel(r'z (au)')
plt.ylabel(r'$\frac{d \tau_{\gamma \gamma}}{d z}$' ' ' r'$(cm^{-1})$' )
#plt.title(u'Optical depth for the interaction between 'r'$\gamma$' '-rays at %.2f GeV \n and photons of a star at %.2f K and a radius %.2f au' %(E_gev, T, R_au))
plt.text(0, 0, u'D$_{star}$ = %.2f au, L = %.2f au \n E$_\gamma$=%.2f TeV' %(D_star_au, L_au, E_tev))
plt.legend()
plt.show()
