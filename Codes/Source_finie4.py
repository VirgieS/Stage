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

def f(theta, phi, eps, z, L, D_star, b, R, E, T, zb):

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

def calculate_tau(E, z, phi, zb, L, D_star, b, R, T):

    #Global constants
    k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
    r0 =  2.818e-13 #classical electron radius (cm)

    integral = np.zeros_like(E)

    for i in range(len(E)):

        integral_eps = np.zeros_like(z)

        #energy of the target-photon
        epsmin = mc2**2/E[i]
        epsmax = 10*k*T

        #Because epsmin must be lower than epsmax
        if epsmin > epsmax:
            continue
        else:
            eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))
            print(i)
            print(len(eps))
            #nu = h/eps

        for j in range (len(z)):

            integral_theta = np.zeros_like(eps)
            D = distance(L, D_star, z[j])
            theta_max = np.arcsin(R/D)
            theta = np.linspace(0, theta_max, 10)

            for k in range (len(eps)):

                integral_phi = np.zeros_like(theta)

                for l in range (len(theta)):

                    integrand = f(theta[l], phi, eps[k], z[j], L, D_star, b, R, E[i], T, zb)
                    integrand = np.nan_to_num(integrand)

                    idx=(integrand > 0.0)

                    #Because the function integral_log works only if there is more than two elements not zero
                    if sum(idx) > 2:
                        integral_phi[l] = integration_log(phi[idx], integrand[idx])

                #Because the function integral_log works only if there is more than two elements not zero
                idx=(integral_phi > 0.0)
                if sum(idx) > 2:
                    integral_theta[k] = integration_log(theta[idx], integral_phi[idx])

            #Because the function integral_log works only if there is more than two elements not zero
            idx=(integral_phi > 0.0)
            if sum(idx) > 2:
                integral_eps[j] = integration_log(eps[idx], integral_theta[idx])

        #Because the function integral_log works only if there is more than two elements not zero
        idx=(integral_eps > 0.0)
        if sum(idx) > 2:
            integral[i] = integration_log(z[idx], integral_eps[idx])

    return  1/2.0 * np.pi * r0**2 * integral

#Global constants
conv_l = 1.45979e13      #conversion factor from kpc to cm
conv_en = 1.602e-16      #conversion factor from J to keV
c = 2.99792458e+10       #light speed in cm/s
k = 1.380658e-23/conv_en # Boltzmann's constant in keV/K
mc2 = 510.9989461        #electron mass (keV)

#For the vector eps, E
number_bin_E = 10
number_bin_eps = 40.0

#Parameters for the code
L = 20 * conv_l                         # the distance to the gamma-source (cm)
zb = 10 * conv_l                        # position along the line of sight nearly the star (cm)
b = 5 * conv_l                          # impact parameter (cm)
D_star =  np.sqrt(b**2 + (L - zb)**2)   # distance to the star (from us) (cm)
D_gamma = np.sqrt(b**2 + L**2)          # distance between the star and the gamma-source (cm)
R = 0.5 * conv_l                        # radius of the star (express in Rsun)
T = 10000                               # temperature of the star (K)
z = np.linspace(0, L, 100)              # position along the line of sight (cm)
phi = np.linspace(0, 2*np.pi, 10)       # angle polar

#energy of the gamma-photon
Emin = 1e8       #Emin = 10^-1 TeV (keV)
Emax = 1e10      #Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E)  # keV
E_tev = E*1e-9   # TeV

tau = calculate_tau(E, z, phi, zb, L, D_star, b, R, T)
D_star_kpc = D_star/conv_l #in kpc
b_au = b/conv_l
plt.plot(E_tev, np.exp(-tau), label = "b = %.2f kpc" %b_au)

D_star_au = D_star/conv_l #in kpc
L_au = L/conv_l #in kpc

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with IR photons of a ponctual source at %.2f K' %T)
plt.text(100, 1,'D$_{star}$ = %.2f kpc, L = %.2f kpc' %(D_star_au, L_au))
plt.legend()
plt.show()
