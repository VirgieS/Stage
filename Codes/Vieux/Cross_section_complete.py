#librairies
import matplotlib.pyplot as plt
import numpy as np

#Wavelengh (cm) to energy (eV)
def energy(l, c):
    h = 6.6260755e-34/1.602e-19 #Planck's constant (eV/K)
    return h*c/l

#Parametre s
def parametre_s(eps, E, mc2, theta):
    return eps*E/(2*mc2**2)*(1-np.cos(theta))

#Parametre beta
def beta_func(s):
    v = 1-1/s
    ind = np.where(v > 0)
    return np.sqrt(v[ind[0]]), ind

#Cross section of the interaction photon-photon (formula 1 from Gould's article)
def cross_section(s, r0):
    beta = beta_func(s)[0]
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#Constants
m = 9.109E-28 #electron mass (g)
c = 2.99792458E+10 #speed of light (cm/s)
mc2 = m*c**2/1.602E-12 #electron mass (eV)
r0 =  2.818E-13 #classical elctron radius (cm)

#Parametres for the photons (target and gamma)

theta = np.linspace(0.00001,np.pi,100) #direction of the second photon (rad)
E = np.array([1e9, 10e9, 100e9, 1e12, 10e12, 100e12]) #energy of the first photon (eV)
l = np.array([1e-4, 100e-4, 1000e-4]) #wavelengh of the target (cm)
eps = energy(l, c) #energy of the second photon (eV)

#parametre s (formula 3 from Gould's article)

for i in range (1,len(E)):
    for j in range (1, len(eps)):

        s = np.zeros_like(theta) #initialization
        s = parametre_s(eps[j], E[i], mc2, theta)

        sigma = cross_section(s, r0)
        ind = beta_func(s)[1]
        print(ind[0])
        theta = theta[ind[0]]

        plt.plot(theta, sigma)
        

plt.xlabel('Angle of the second photon (rad)')
plt.ylabel('Cross section (cm^2)')
plt.show()
