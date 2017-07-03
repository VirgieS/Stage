import matplotlib.pyplot as plt
import numpy as np

def integrand(epsc, mc2, E, r0, k, hbar, c):

    def cross_section(epsc, mc2, r0):

        def parameter_beta(s):
            return np.sqrt(1 - 1/s) #equation 4 from Gould's article

        s = (epsc/mc2)**2 #epsc = (1/2 * eps * E * (1 - cos(theta)))^(1/2), the center-of-momentum system energy of a photon-photon
        #ind = np.where(s >=1)
        #s = s[ind[0]]
        #epsc = epsc[ind[0]]
        #print(len(epsc))
        beta = parameter_beta(s)

        return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))#, ind

    sigma = cross_section(epsc, mc2, r0)
    #epsc = epsc[ind[0]]
    log = np.log(1 - np.exp(- epsc**2/(E*k*T)))
    #print(len(sigma))

    return -4 * k * T/((hbar*c)**3 * np.pi**2 * E**2) * epsc**3 * sigma * log #f(epsc) = epsc^3 * sigma_int(epsc) * log(1 - exp(-epsc^2/(E*k*T))) (formula 3 of Moskalenko's article)

#Integral of f(x) = a*log(x) + b
def integration_semilog(x, y):

    #Looking for a and b for y = a*x^b
    def calculate_ab(xi, xf, yi, yf):
        logxi = np.log(xi)
        logxf = np.log(xf)
        a = (yf - yi)/(logxf - logxi)
        b = yi - a*logxi
        return a, b

    #Calculate deltaS from deltaS = int from xi to xf a*x^b
    def delta_S(xi, xf, yi, yf):
        [a, b] = calculate_ab(xi, xf, yi, yf)
        return a * (xf * np.log(xf) - xi * np.log(xi) - (xf - xi)) + b *(xf - xi)

    deltaS = np.zeros(len(x)-1)
    integral = 0

    for i in range (1, len(x)):
        deltaS[i-1] = delta_S(x[i-1], x[i], y[i-1], y[i])
        integral = integral + deltaS[i-1]

    return integral

#Constants
mc2 = 510.9989461E+3 #electron mass (eV)
r0 =  2.818E-13 #classical elctron radius (cm)
k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV/K
hbar = 6.6260755e-34/1.602e-19/(2*np.pi) #Planck's constant in eV/s
c = 2.99792458e+10 #light speed in cm/s
T = 2.7 #temperature of the CMB in K

#Parameters for the integral
epsc = np.linspace(mc2+1e3, 1e7, 1000000) #energy of the target-photon in eV from mc^2 to 1TeV
E = 1e15 #energy of the first photon in eV
f = np.zeros_like(epsc)
for i in range (0, len(epsc)):
    f[i] = integrand(epsc[i], mc2, E, r0, k, hbar, c)
    if f[i] == 0: #break condition, if f is zero then the integral is zero
        break

ind = np.where(f>1e-30)
f = f[ind[0]]
epsc = epsc[ind[0]]

plt.xscale('log')
plt.plot(epsc, f)
plt.show()
integral_x = integration_semilog(epsc, f)
R = 0 #distance to the centre of the object in kpc
Rs = 8.5 #distance to the centre of the Sun in kpc
alpha = 0 #right ascension in radian

rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))

L = np.sqrt(R**2 + rho**2)
tau = integral_x * L #because all caracteristc are constants along the line of sight
print(tau)
print(np.exp(-tau))
