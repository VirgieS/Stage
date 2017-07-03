import matplotlib.pyplot as plt
import numpy as np

from principal_functions import integration, density_n, cos_theta, sigma_func

#Parameters
T = 2.7 #Temperature of CMB in Kelvin
mc2 = 510.9989461e+3 #electron mass (eV)
l = np.array([1e-1, 10e-4]) #wavelength in cm
h = 6.6260755e-34/1.602e-19 #Planck's constant in eV
c = 2.99792458e+10 #light speed in cm/s
eps = h*c/l
E = 1e14


dn = density_n(l, T, eps)

theta1 = np.linspace(0, np.pi, 100)
phi1 = np.linspace(0, 2*np.pi, 100)
R = 5
Rs = 8.5
alpha = 0
z = 0


#sigma = sigma_func(R, Rs, alpha, z, theta1, phi1, eps, E)
#print(sigma)
costheta = np.zeros_like(theta1)
dn = np.zeros_like(theta1)
sigma = np.zeros_like(theta1)

for j in range (0, len(eps)):
    for i in range (0, len(theta1)):
        costheta[i] = cos_theta(R, Rs, alpha, z, theta1[i], phi1[i])
        sigma[i] = sigma_func(R, Rs, alpha, z, theta1[i], phi1[i], eps, E)
        dn[i] = density_n(l, T, eps[j])
        print(costheta)
