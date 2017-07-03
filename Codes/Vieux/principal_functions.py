import matplotlib.pyplot as plt
import numpy as np

def integration(x, y):
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

    return integral

#spectral radiance of a black body (in wavelength)
def B_lambda(l, T):
    k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV
    c = 2.99792458e+10 #light speed in cm/s
    h = 6.6260755e-34/1.602e-19 #Plamck's constant in eV
    return 2*h*c**2/(l**5)*1/(np.exp(h*c/(l*k*T))-1)

def density_n(l, T, eps):
    h = 6.6260755e-34/1.602e-19 #Planck's constant in eV

    #spectral radiance of a black body (in wavelength)
    def B_lambda(l, T, h):
        k = 1.380658e-23/1.602e-19 #Boltzmann's constant in eV
        c = 2.99792458e+10 #light speed in cm/s
        return 2*h*c**2/(l**5)*1/(np.exp(h*c/(l*k*T))-1)

    Bl = B_lambda(l, T, h)
    return Bl * h/eps**3

def cos_theta(R, Rs, alpha, z, theta1, phi1):
    rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))
    sintheta2 = rho/(np.sqrt(rho**2 + z**2))
    costheta2 = -1*np.sqrt(1 - sintheta2**2)
    sinphi2 = -(Rs*np.sin(alpha))/rho
    cosphi2 = -(rho**2 + R**2 - Rs**2)/(2*R*rho)
    cosphi12 = np.cos(phi1)*cosphi2 + np.sin(phi1)*sinphi2
    return np.cos(theta1)*costheta2 + np.sin(theta1)*sintheta2*cosphi12

def sigma_func_gen(R, Rs, alpha, z, theta1, phi1, eps, E):

    mc2 = 510.9989461E+3 #electron mass (eV)
    r0 =  2.818E-13 #classical elctron radius (cm)
    rho = np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))
    sintheta2 = rho/(np.sqrt(rho**2 + z**2))
    costheta2 = -1*np.sqrt(1 - sintheta2**2)
    sinphi2 = -(Rs*np.sin(alpha))/rho
    cosphi2 = -(rho**2 + R**2 - Rs**2)/(2*R*rho)
    cosphi12 = np.cos(phi1)*cosphi2 + np.sin(phi1)*sinphi2
    costheta = np.cos(theta1)*costheta2 + np.sin(theta1)*sintheta2*cosphi12
    epsc = np.sqrt(1/2.0*eps*E*(1-costheta))

    #Parametre s (formula 3 from Gould's article)
    def parametre_s(epsc, E, mc2):
        s = epsc**2/mc2**2
        #ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition.
        return s#, ind

    #Parametre beta (formula 4 from Gould's article)
    def beta_func(s):
        return np.sqrt(1-1/s)
    #voir si les deux fonctions fonctionnent

    #Cross section of the interaction photon-photon (formula 1 from Gould's article)
    def cross_section(s, r0):
        beta = beta_func(s)
        return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

    #Vector for the plot of the total cross section
    def vec_plot(epsc, E, mc2):
        s = np.zeros_like(epsc) #initialization
        s = parametre_s(epsc, E, mc2)
        #s = s[ind[0]]
        return cross_section(s, r0)#, ind

    sigma = vec_plot(epsc, E, mc2)

    return sigma

def sigma_func_CMB(epsc, E): #from the formula 3 of Moskalenko's article

    mc2 = 510.9989461E+3 #electron mass (eV)
    r0 =  2.818E-13 #classical elctron radius (cm)

    #Parametre s (formula 3 from Gould's article)
    def parametre_s(epsc, E, mc2):
        s = epsc**2/mc2**2
        print(s)
        ind = np.where(s>1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition.
        return s, ind

    #Parametre beta (formula 4 from Gould's article)
    def beta_func(s):
        return np.sqrt(1-1/s)

    #Cross section of the interaction photon-photon (formula 1 from Gould's article)
    def cross_section(s, r0):
        beta = beta_func(s)
        print(beta)
        return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

    #Vector for the plot of the total cross section
    def vec_plot(epsc, E, mc2):
        s = np.zeros_like(epsc) #initialization
        [s, ind] = parametre_s(epsc, E, mc2)
        s = s[ind[0]]
        return cross_section(s, r0), ind

    [sigma, ind] = vec_plot(epsc, E, mc2)

    return sigma, ind
