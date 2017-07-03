import matplotlib.pyplot as plt
import numpy as np

def integrand(epsc, mc2, E, r0, k, hbar, c):

    def cross_section(epsc, mc2, r0):

        def parameter_beta(s):
            return np.sqrt(1 - 1/s) #equation 4 from Gould's article

        s = (epsc/mc2)**2 #epsc = (1/2 * eps * E * (1 - cos(theta)))^(1/2), the center-of-momentum system energy of a photon-photon
        beta = parameter_beta(s)

        return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))#, ind

    sigma = cross_section(epsc, mc2, r0)
    log = np.log(1 - np.exp(- epsc**2/(E*k*T)))

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

    deltaS = np.zeros(len(x))
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

R = 0 #distance to the centre of the object in kpc
Rs = 8.5 #distance to the centre of the Sun in kpc
alpha = 0 #right ascension in radian
rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))
L = np.sqrt(R**2 + rho**2)

#Parameters for the integral
epsc = np.linspace(mc2+1e3, 1e7, 10000) #energy of the target-photon in eV from mc^2 to 1TeV
E = np.linspace(1e12, 1e17, 100) #energy of the first photon in eV
x = np.linspace(0, L, 100) #the length of the line of sight

integral_x = np.zeros_like(x)
tau = np.zeros_like(E)

for j in range (0, len(E)):

    for k in range (0, len(x)):

        epsc = np.linspace(mc2+1e3, 1e7, 10000) #energy of the target-photon in eV from mc^2 to 1TeV
        f = np.zeros_like(epsc)

        for i in range (0, len(epsc)):
            f[i] = integrand(epsc[i], mc2, E[j], r0, k, hbar, c)
            if f[i] == 0: #break condition, if f is zero then the integral is zero
                break

        ind = np.where(f>0)
        f = f[ind[0]]
        epsc = epsc[ind[0]]

        integral_x[k] = integration_semilog(epsc, f)

    tau[j] = integral_x[k] * L #because all characteristics are constants along the line of sight

transmittance = np.exp(-tau)

E = E*1e-12 #the energy of the first photon in TeV

#plt.plot(E, transmittance)
#plt.xscale('log')
#plt.xlabel('E'r'$_\gamma$''(TeV)')
#plt.ylim(0, 1.05)
#plt.ylabel('exp'r'($- \tau_{\gamma\gamma}$)')
#plt.show()

print(len(integral_x))
#plt.plot(epsc, integral_x[1])
#plt.xscale('log')
#plt.show()
