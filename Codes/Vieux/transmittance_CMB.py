#librairies
import matplotlib.pyplot as plt
import numpy as np

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
        return a, b

    #Calculate deltaS from deltaS = int from xi to xf a*x^b
    def delta_S(xi, xf, yi, yf):
        [a, b] = calculate_ab(xi, xf, yi, yf)
        return a/(b+1)*(xf**(b+1) - xi**(b+1))

    #Calculate total integral from init to final a*x^b
    integral = 0
    deltaS = np.zeros(len(x)-1)

    for i in range (1, len(x)):
        deltaS[i-1] = delta_S(x[i-1], x[i], y[i-1], y[i])
        integral = integral + deltaS[i-1]

    integral = np.nan_to_num(integral)

    return integral

def cos_theta(psi, phi): #alpha = colatitude and beta is the polar angle of the background photon if z is the direction of the gamma
    return np.cos(np.pi - psi)

def density_n(eps, T):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
    c = 2.99792458e+10 #light speed in cm/s
    l = h*c/eps

    #spectral radiance of a black body (in wavelength)
    def B_lambda(l, T, h, c):
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in eV
        return 2*h*c**2/(l**5)*1/(np.exp(h*c/(l*k*T))-1)

    Bl = B_lambda(l, T, h, c)
    return Bl * h/(eps**3)

def Wien(T):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s
    b = 2.8977729e-1 #Wien's displacement constant (cm*K)
    lmax = b/T
    return h*c/lmax

def cross_section(eps, E, psi, phi): #sigma_int  = 1/2.0 * pi * r0**2 * sigma (equation 1 of Gould's article), we calculate sigma and not sigma_int
                                        #eps and E must be given in keV

    costheta = cos_theta(psi, phi)

    def parameter_s(eps, E, costheta):
        mc2 = 510.9989461 #electron mass (keV)
        return eps * E/(2* mc2**2) * (1 - costheta)

    def parameter_beta(eps, E, costheta):
        s = parameter_s(eps, E, costheta)
        return np.sqrt(1 - 1/s)

    beta = parameter_beta(eps, E, costheta)
    beta = np.nan_to_num(beta)
    return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

def f(psi, eps, E, T):
    sigma = cross_section(eps, E, psi, phi)
    dn = density_n(eps, T)
    costheta = cos_theta(psi, phi)
    return sigma * dn * (1 - costheta) * np.sin(psi)

#Global constants
r0 =  2.818E-13 #classical elctron radius (cm)
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV/K
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)

#Parameters
T = 2.7 #temperature of the radiation (here the CMB at 2.7 K)

#distance from the source to the Sun
R = 0 #distance to the Galactic center of the source (kpc)
Rs = 8.5 #distance to the Galactic center of Sun (kpc)
alpha_r = 0 #right ascension of the source
z = 0
L = np.sqrt(z**2 + R**2 + Rs**2 - 2*R*Rs*np.cos(alpha_r)) #distance to the source (kpc)
L = L * 3.085678e21 #in cm

#energy of the VHE gamma-photons (keV)
E = np.linspace(1e8, 1e14, 100)
lxmin = np.log10(E.min())
lxmax = np.log10(E.max())
E = 10**np.linspace(lxmin, lxmax, 10)
Ef = E*1e-12*1e3
integral_eps = np.zeros_like(E)
epsmax = Wien(T)

for i in range (len(E)):

    eps = np.linspace(mc2**2/E[i], 10 * k * T, 10) #energy of background photon (keV)
    integral_psi = np.zeros_like(eps)

    for j in range(len(eps)):
        psi = np.linspace(0, np.pi , 10)
        phi = np.linspace(0, 2*np.pi, 20)

        #integrand = f(eps[j], E[i], T, psi, phi)
        integral_psi[j] = quad(f, 0, np.pi, args = (eps[j], E[i], T))[0]#integration_log(psi, integrand)
        print(integral_psi[i])

    integral_eps[i] = integration_log(eps, integral_psi)
tau = 2*np.pi * L * 1/2.0 * np.pi * r0**2 * integral_eps


plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title('Transmittance of VHE 'r'$\gamma$' '-rays in interaction with CMB photons')
plt.plot(Ef, np.exp(-tau))
plt.show()
