#librairies
import matplotlib.pyplot as plt
import numpy as np

e = 1.6E-19 #electronic charge (C)
m = 9.109E-28 #electron mass (g)
c = 2.99792458E+10 #speed of light (cm/s)
mc2 = m*c**2/1.602E-12 #electron mass in eV
L = 3.86E-11 #electron Compton wavelength (cm)
kB = 1.38E-16/1.602E-12 #Boltzmann constant in eV/K

r0 =  2.818E-13 #classical elctron radius (cm)
theta = np.linspace(0,np.pi,100) #direction of the second photon (rad)

E = 1E9 #energy of the first photon (eV)
eps = 2 #energy of the second photon (eV)

s = np.zeros_like(theta) #initialize the parameter s from the formula 3 in the article of Gould
s = eps*E/(2*mc2**2)*(1-np.cos(theta)) #expression of s using theta

beta = np.sqrt(1-1/s)
sigma = 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))  #formula (1) from the article of Gould

print(sigma)

plt.plot(theta, sigma)
plt.xlabel('Angle of the second photon')
plt.ylabel('Cross section')
plt.show()
