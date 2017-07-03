#librairies
import matplotlib.pyplot as plt
import numpy as np

#Wavelength (cm) to energy (eV)
def energy(l, c):
    h = 6.6260755e-34/1.602e-19 #Planck's constant (eV/K)
    return h*c/l

#Parametre s (formula 3 from Gould's article)
def parametre_s(eps, E, mc2, theta):
    s = eps*E/(2*mc2**2)*(1-np.cos(theta))
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
def vec_plot(theta, eps, E, mc2):
    s = np.zeros_like(theta) #initialization
    [s,ind] = parametre_s(eps, E, mc2, theta)
    s = s[ind[0]]
    return cross_section(s, r0), ind

#Constants
c = 2.99792458E+10 #speed of light (cm/s)
mc2 = 510.9989461E+3 #electron mass (eV)
r0 =  2.818E-13 #classical elctron radius (cm)

#Parametres for the photons (target and gamma)

theta = np.linspace(0.000001,np.pi,100) #direction of the second photon (rad)
E = np.array([1e9, 100e9, 1e12, 10e12]) #energy of the first photon (eV)
l = np.array([1e-4, 10e-4, 100e-4, 1000e-4]) #wavelengh of the target (cm)
eps = energy(l, c) #energy of the second photon (eV)
l = l*1e4 #wavelength in micrometer

#For E=1Gev
[sigma, ind] = vec_plot(theta, eps[0], E[0], mc2)
un, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[0])

[sigma, ind] = vec_plot(theta, eps[1], E[0], mc2)
dix, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[1])

[sigma, ind] = vec_plot(theta, eps[2], E[0], mc2)
cent, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[2])

[sigma, ind] = vec_plot(theta, eps[2], E[0], mc2)
mille, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[3])

Eg = E[0]*1E-9 #energy of the gamma-photon (GeV)
plt.title('Cross section for the first photon at %d GeV' %Eg)
plt.xlabel(r'$\theta$''(rad)')
plt.ylabel(r'$\sigma_{int}$''(cm' r'$^2$'')')
plt.xlim(0,np.pi)
plt.legend(handles=[un, dix, cent, mille])
plt.show()

#For E = 100GeV

[sigma, ind] = vec_plot(theta, eps[0], E[1], mc2)
un, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[0])

[sigma, ind] = vec_plot(theta, eps[1], E[1], mc2)
dix, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[1])

[sigma, ind] = vec_plot(theta, eps[2], E[1], mc2)
cent, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[2])

[sigma, ind] = vec_plot(theta, eps[2], E[1], mc2)
mille, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[3])

Eg = E[1]*1E-9 #energy of the gamma-photon (GeV)
plt.title('Cross section for the first photon at %d GeV' %Eg)
plt.xlabel(r'$\theta$''(rad)')
plt.ylabel(r'$\sigma_{int}$''(cm' r'$^2$'')')
plt.xlim(0,np.pi)
plt.legend(handles=[un, dix, cent, mille])
plt.show()

#For E = 1TeV

[sigma, ind] = vec_plot(theta, eps[0], E[2], mc2)
un, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[0])

[sigma, ind] = vec_plot(theta, eps[1], E[2], mc2)
dix, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[1])

[sigma, ind] = vec_plot(theta, eps[2], E[2], mc2)
cent, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[2])

[sigma, ind] = vec_plot(theta, eps[2], E[2], mc2)
mille, = plt.plot(theta[ind[0]], sigma, label="%d $\mu$m" %l[3])

Eg = E[2]*1E-12 #energy of the gamma-photon (GeV)
plt.title('Cross section for the first photon at %d TeV' %Eg)
plt.xlabel(r'$\theta$''(rad)')
plt.ylabel(r'$\sigma_{int}$''(cm' r'$^2$'')')
plt.xlim(0,np.pi)
plt.legend(handles=[un, dix, cent, mille])
plt.show()

#For E = 10TeV

[sigma, ind] = vec_plot(theta, eps[0], E[3], mc2)
un, = plt.plot(theta[ind[0]], sigma, label="target-photon at %d $\mu$m" %l[0])

[sigma, ind] = vec_plot(theta, eps[1], E[3], mc2)
dix, = plt.plot(theta[ind[0]], sigma, label="target-photon at %d $\mu$m" %l[1])

[sigma, ind] = vec_plot(theta, eps[2], E[3], mc2)
cent, = plt.plot(theta[ind[0]], sigma, label="target-photon at %d $\mu$m" %l[2])

[sigma, ind] = vec_plot(theta, eps[2], E[3], mc2)
mille, = plt.plot(theta[ind[0]], sigma, label="target-photon at %d $\mu$m" %l[3])

Eg = E[3]*1E-12 #energy of the gamma-photon (GeV)
plt.title('Cross section for the first photon at %d TeV' %Eg)
plt.xlabel(r'$\theta$''(rad)')
plt.ylabel(r'$\sigma_{int}$''(cm' r'$^2$'')')
plt.xlim(0,np.pi)
plt.legend(handles=[un, dix, cent, mille])
plt.show()
