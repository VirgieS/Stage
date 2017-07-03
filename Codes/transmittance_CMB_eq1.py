#! /Applications/Anaconda/bin/python

#Our coordinate are psi, azimutal angle from the direction of the gamma-photon and phi, polar angle around this axis.

#librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.integrate import quad

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

def f(psi, eps, E, T):

    #Density of a Black Body
    def density_n(eps, T):

        #Global constants
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
        h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
        c = 2.99792458e+10 #light speed in cm/s

        nu = eps/h

        def planck(nu, T):

            return (2*h*(nu**3))/(c**2) * 1/(np.exp((h*nu)/(k*T)) - 1)

        Bnu = planck(nu, T)

        return Bnu/(h * eps * c)

    def cos_theta(psi):

        return np.cos(np.pi-psi)

    mc2 = 510.9989461 #electron mass (keV)

    costheta = cos_theta(psi)
    epsc = np.sqrt(eps * E/2 * (1 - costheta))/mc2 #it gives epsc in keV/mc2

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

    sigma = cross_section(epsc)
    dn = density_n(eps, T)

    return sigma * dn * (1 - costheta) * np.sin(psi)


#Global constants
r0 =  2.818e-13 #classical electron radius (cm)
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)
T = 2.7 #temperature of the CMB (K)

#For the vector eps, E
number_bin_E = 80
number_bin_eps = 40.0

#Distance to the source
R = 0 #distance to the Galactic center of the source (kpc)
Rs = 8.5 #distance to the Galactic center of Sun (kpc)
alpha_r = 0 #right ascension of the source (radian)
z = 0 #height above the Galactic plane (kpc)
L = np.sqrt(z**2 + R**2 + Rs**2 - 2*R*Rs*np.cos(alpha_r)) #distance to the source (kpc)
L = L * 3.085678e21 #in cm

#energy of the gamma-photon
Emin = 1e8 #we choose Emin = 10^-1 TeV (keV)
Emax = 1e14 #we choose Emax = 10^5 TeV (keV)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E) # keV
print(len(E))
E_tev = E*1e-9 #TeV

integral_eps = np.zeros_like(E)

for i in range (len(E)):

    epsmin = mc2**2/E[i]
    epsmax = 10*k*T

    #Because epsmin must be lower than epsmax
    if epsmin > epsmax:
        continue
    else:
        eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

    integral_psi = np.zeros_like(eps)
    psi = np.linspace(0, np.pi, 100) #azimutal angle if z is the direction of the gamma-photon

    for j in range (len(eps)):

        integrand = f(psi, eps[j], E[i], T)
        integrand = np.nan_to_num(integrand)
        idx=(integrand > 0.0)

        #Because the function integral_log works only if there is more than two elements not zero
        if sum(idx) > 2:
       	    integral_psi[j] = integration_log(psi[idx], integrand[idx])

    #Because the function integral_log works only if there is more than two elements not zero
    idy=(integral_psi > 0.0)
    if sum(idy) > 2:
        integral_eps[i] = integration_log(eps[idy], integral_psi[idy])

tau = 1/2.0 * np.pi * r0**2 * L * 2 * np.pi * integral_eps #the intergation in phi gives 2*pi and the integration in x gives L

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
plt.plot(E_tev, np.exp(-tau))
plt.show()
