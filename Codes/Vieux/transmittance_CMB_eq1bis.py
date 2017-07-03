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

def density_IR(eps, T, uIR):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
    c = 2.99792458e+10 #light speed in cm/s
    s = 5.67051/1.602e-16 #Stefan-Boltzmann constant (keV cm^-2 K^-4 s^1)
    a = 4 * s/c**3 #radiation constant (keV cm^-3 K^-4)

    #spectral radiance of a black body (in wavelength)
    def B_eps(eps, T, h, c):
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV
        return 2*eps**3/(h**3 * c**2) *1/(np.exp(eps/(k * T))-1)

    Beps = B_eps(eps, T, h, c)
    return Beps * 1/(eps * c) * uIR/(a * T**4)

def density_CMB(eps, T):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
    c = 2.99792458e+10 #light speed in cm/s

    #spectral radiance of a black body (in wavelength)
    def B_eps(eps, T, h, c):
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV
        return 2*eps**3/(h**3 * c**2) *1/(np.exp(eps/(k * T))-1)

    Beps = B_eps(eps, T, h, c)
    return Beps * 1/(eps * c)

def f(eps, T, E, R, Rs, alpha, z, costheta1, phi1, dn): #esp, E must be given in keV, phi1 and alpha in rad, R, Rs and z in kpc

    def cos_theta(R, Rs, alpha, z, costheta1, phi1): #from equation 2 of Moskalnko's article
        rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))
        sintheta2 = rho/(np.sqrt(rho**2 + z**2))
        costheta2 = -1*np.sqrt(1 - sintheta2**2)
        sinphi2 = -(Rs*np.sin(alpha))/rho
        cosphi2 = -(rho**2 + R**2 - Rs**2)/(2*R*rho)
        sintheta1 = np.sqrt(1 - costheta1**2)
        cosphi12 = np.cos(phi1)*cosphi2 + np.sin(phi1)*sinphi2
        return costheta1*costheta2 + sintheta1*sintheta2*cosphi12

    costheta = cos_theta(R, Rs, alpha, z, costheta1, phi1)
    epsc = np.sqrt(1/2.0 * eps * E * (1-costheta)) #variable for the integration (mc2^-1)

    def cross_section(epsc): #sigma_int  = 1/2.0 * pi * r0**2 * sigma (equation 1 of Gould's article), we calculate sigma and not sigma_int

        def parameter_s(epsc):

            mc2 = 510.9989461 #electron mass (keV)
            return epsc**2

        def parameter_beta(epsc):

            s = parameter_s(epsc)
            print(s)
            ind = np.where(s>=1)
            s = s[ind[0]]
            return np.sqrt(1 - 1/s), ind

        [beta, ind] = parameter_beta(epsc)
        print(beta)
        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2)), ind

    [sigma, ind] = cross_section(epsc)

    return dn * sigma * (1 - costheta[ind[0]]), ind

#Constants

r0 =  2.818E-13 #classical elctron radius (cm)
mc2 = 510.9989461 #electron mass (keV)
h = 6.6260755e-34/1.602e-16/mc2 #Planck's constant in keV*s/mc2
c = 2.99792458e+10 #light speed in cm/s
k = 1.380658e-23/1.602e-16/mc2 #Boltzmann's constant in keV/K/mc2


#Parameters

R = 5 #distance to the galactic center (kpc)
Rs = 8.5 #distance of the Sun to the galactic center (kpc)
z = 0 #height above the Galactic plan (kpc)
alpha = 0 #Right ascension (rad)

T = 2.7 #temperature of the CMB (K)

#engergy of the VHE gamma-rays
E = np.linspace(1e8, 1e14, 100) #in keV
lxmin = np.log10(E.min())
lxmax = np.log10(E.max())
E = 10**np.linspace(lxmin, lxmax, 10) #energy-vector with a number of points that increase like a power of 10
costheta1 = np.linspace(-1, 1, 10)
phi1 = np.linspace(0, 2*np.pi, 10)

#positions
rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha)) #in kpc
L = np.sqrt(R**2 + rho**2) #in kpc
L = L * 3.085678e21 #in cm

#for the thirst integral
integral_phi = np.zeros_like(phi1)
integral_theta = np.zeros_like(costheta1)
integral_eps = np.zeros_like(E)

for i in range(len(E)):

    #energy of the background photon (keV/mc2)
    eps = np.linspace(mc2/E[i] , 10*k*T/mc2, 100)
    lxmin = np.log10(eps.min())
    lxmax = np.log10(eps.max())
    eps = 10**np.linspace(lxmin , lxmax, 100) #energy-vector with a number of points that increase like a power of 10

    for j in range(len(eps)):
        dn = density_CMB(eps[j], T)

        for k in range (len(phi1)):
            [integrand, ind] = f(eps[j], T, E[i], R, Rs, alpha, z, costheta1, phi1[k], dn)

            integral_phi[k] = integration_log(phi1, integrand)

        integral_theta[j] = integration_log(costheta1, integral_phi)

    integral_eps[i] = integration_log(eps, integral_theta)

tau = L * integral_eps * 1/2.0 * np.pi * r0**2

#plt.xscale('log')
#plt.yscale('log')
#plt.plot(eps, integrand)
#plt.show()
