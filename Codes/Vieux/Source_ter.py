#Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

def distance(zb, z, b): #distance between the gamma photon and the source IR

    return np.sqrt((zb - z)**2 + b**2)

"""
density of the photons is not isotropic and is : dn = u/(a*T^4) * Bnu/(c*h*nu)

Parameters:

u_in is the choosen energy density of the (ponctual) star in a position along the line of sight (keV/cm^3)
b is the impact parameter (cm)
z is the position along the line of sight (cm)
eps is the energy of the target-photon (keV)
T is the temperature of the source (K)
alpha is the right ascension (rad)
L is the distance between us and the source of gamma (cm)
R is the galactic radius of the source of gamma (cm)
Rs is the galactic radius of the sun (cm)
"""

def density_n(u_in, b, d, eps, T):

    def energy_density(u_in, b, d, T): #energy density of the source IR

        #Global energy
        c = 2.99792458e+10 #light speed in cm/s
        sb = 5.670367e-12/1.602e-16 #Stefan-Boltzmann constant (keV s^-1 cm^-2 K^-4)
        a = 4 * sb/c #radiation constant (keV cm^-3 K^-4)
        h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s

        def luminosity(u_in, b): #luminosity of the source IR

            c = 2.99792458e+10 #light speed in cm/s

            return u_in * c * 4 * np.pi * b**2

        Lum = luminosity(u_in, b)

        return Lum/(4 * np.pi * d**2 * c) * 1.0/(a * T**4)

    nu = eps/h

    def planck(nu, T):

        #Global constants
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
        c = 2.99792458e+10 #light speed in cm/s

        return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(k*T)) - 1)

    Bnu = planck(nu, T)

    u = energy_density(u_in, b, d, T)

    return u * 4 * np.pi * Bnu/(h * c * eps)

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
def angle(d, b, z, zb): #d, b, z and zb must be given in cm

    if z <zb:
        theta = np.pi - np.arcsin(b*1.0/d)

    else:
        theta = np.arcsin(b*1.0/d)

    return theta

"""
function for integration

Parameters:
u_in is the choosen energy density of the (ponctual) star in a position along the line of sight (keV/cm^3)
b is the impact parameter (cm)
d is the distance between the gamma-photon and the source IR (cm)
eps is the energy of the target-photon (keV)
T is the temperature of the source (K)
z is the position along the line of sight (cm)
E is the energy of the gamma-photon (keV)
"""

def f(u_in, b, d, eps, T, z, zb, E):

    mc2 = 510.9989461 #electron mass (keV)

    theta = angle(d, b, z, zb)

    epsc = np.sqrt(eps * E/2 * (1 - np.cos(theta)))/mc2 #it gives epsc in keV/mc2

    def cross_section(epsc): #sigma_int  = 1/2.0 * pi * r0**2 * sigma (equation 1 of Gould's article), we calculate sigma and not sigma_int

        def parameter_s(epsc):

            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

        beta = parameter_beta(epsc)
        beta = np.nan_to_num(beta)

        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    dn = density_n(u_in, b, d, eps, T)
    sigma = cross_section(epsc)
    return dn * sigma * (1 - np.cos(theta))

#Global constants
r0 =  2.818e-13 #classical electron radius (cm)
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)


#For the vector eps, E
number_bin_E = 80
number_bin_eps = 40.0

#Distance to the source
R = 5 *  3.085678e21#distance between the gamma-source and the IR source (cm)
L = 30 *  3.085678e21  #distance to the source (cm)

b = 2 * 3.085678e21 #impact parameter (cm)
z = np.linspace(0, L, 100) #position along the line of sight (cm)
zb = np.sqrt(R**2 - b**2) #position along the line of sight closely the source IR (cm)
d = distance(zb, z, b)

#Parameters
T = 25 #temperature of the CMB (K)
u_in = 1e-3 #energy density at the impact parameter (keV/cm^3)


#energy of the gamma-photon
Emin = 1e8 #we choose Emin = 10^-1 TeV (keV)
Emax = 1e14 #we choose Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E) # keV
E_tev = E*1e-9 #TeV

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
        #nu = h/eps

    for j in range(len(z)):

        integrand = f(u_in, b, d[j], eps, T, z[j], zb, E[i])
        integrand = np.nan_to_num(integrand)

        idx=(integrand > 0.0)

        #Because the function integral_log works only if there is more than two elements not zero
        if sum(idx) > 2:
            integral_eps[j] = integration_log(eps[idx], integrand[idx])

    #Because the function integral_log works only if there is more than two elements not zero
    idy=(integral_eps > 0.0)
    if sum(idy) > 2:
        integral[i] = integration_log(z[idy], integral_eps[idy])

tau =  1/2.0 * np.pi * r0**2 * integral

R_kpc = R/3.085678e21 #in kpc
L_kpc = L/3.085678e21 #in kpc
b_kpc = b/3.085678e21 #in kpc

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with IR photons of a ponctual source at %.2f K' %T)
plt.text(100, 0.98, 'R = %.2f kpc, L = %.2f kpc b = %.2f kpc' %(R_kpc, L_kpc, b_kpc))
plt.plot(E_tev, np.exp(-tau))
plt.show()
