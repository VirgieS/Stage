#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

def density_n(eps, T, theta):

    """
    Density of the photons is not isotropic : dn = Bnu * cos(theta)/(c * h**2 *nu)

    Parameters:
        eps   : energy of the target-photon (keV)
        theta : angle formed by the ray (from the star) and the line connecting the centre of the star and a position z on the line of sight (rad)
        T     : temperature of the star (K)
    """

    #Global constants
    h = 6.6260755e-34/conv_en #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s

    nu = eps/h

    def planck(nu, T):

        #Global constants
        kb =1.380658e-23/conv_en #Boltzmann's constant in keV/K
        c = 2.99792458e+10 #light speed in cm/s

        return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(kb*T)) - 1)

    Bnu = planck(nu, T)

    return Bnu/(c * h**2 * nu) * np.cos(theta) #return dn in cm^3/sr/keV

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

# Global constants
conv_l = 1.45979e13      # Conversion factor from au to cm
conv_en = 1.602e-16      # Conversion factor from J to keV
c = 2.99792458e+10       # Light speed in cm/s
kb =1.380658e-23/conv_en # Boltzmann's constant in keV/K
mc2 = 510.9989461        # Electron mass (keV)

# For the vector eps and E
number_bin_E = 40
number_bin_eps = 40.0

# Parameters for the code
L = 20 * conv_l                         # the distance to the gamma-source (cm)
zb = 10 * conv_l                        # position along the line of sight nearly the star (cm)
b = 5 * conv_l                          # impact parameter (cm)
D_star =  np.sqrt(b**2 + (L - zb)**2)   # distance to the star (from us) (cm)
D_gamma = np.sqrt(b**2 + L**2)          # distance between the star and the gamma-source (cm)
R = 0.5 * conv_l                        # radius of the star (express in Rsun)
T = np.array([3000, 6000, 10000])             # temperature of the star (K)
z = np.linspace(0, L, 100)              # position along the line of sight (cm)
phi = np.linspace(0, 2*np.pi, 10)       # angle polar

# Energy of the gamma-photon
E = 1e9  # keV
E_tev = E*1e-9   # TeV

integral = np.zeros_like(E)
theta = np.pi/2

integral_eps = np.zeros_like(T)

for j in range (len(T)):

    # Energy of the target-photon
    epsmin = mc2**2/E
    epsmax = 10*kb*T[j]
    eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))
    integrand = density_n(eps, T[j], theta)
    integrand = np.nan_to_num(integrand)

    # Because the function integral_log works only if there is more than two elements not zero
    idx=(integrand > 0.0)
    if sum(idx) > 2:
        integral_eps[j] = integration_log(eps[idx], integrand[idx])
        plt.plot(T[j], integral_eps[j], label = "For $\theta$ = %.2f and T = %.2f$" %(theta, T[j]))

plt.xlabel(r'$T$' '(K)')
plt.ylabel(r'$c$' ' ' r'$(sr^{-1} cm^{-1})$')
#plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with IR photons of a ponctual source at %.2f K' %T)
#plt.text(100, 1,'D$_s$ = %.2f kpc, L = %.2f kpc' %(Ds_kpc, L_kpc))
plt.legend()
plt.show()
