import matplotlib.pyplot as plt
import numpy as np

h = 6.6260755e-34/1.602e-19
c = 2.99792458e10
theta1 = 0
phi1 = 0
R = 5
Rs = 8.5
alpha = 0
z = 0
l = np.array([1e-4, 10e-4])
eps = h*c/l
E = 1e12

def sigma_func(R, Rs, alpha, z, theta1, phi1, eps, E):

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
        ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition.
        return s, ind

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
        [s,ind] = parametre_s(epsc, E, mc2)
        s = s[ind[0]]
        return cross_section(s, r0), ind

    [sigma, ind] = vec_plot(epsc, E, mc2 )

    return sigma

sigma = sigma_func(R, Rs, alpha, z, theta1, phi1, eps, E)
print(sigma)
