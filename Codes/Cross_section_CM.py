import matplotlib.pyplot as plt
import numpy as np

#parametre s (formula 3 from Gould's article)
def parametre_s(Ee, mc2):
    s = (Ee/mc2)**2
    ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition
    return s, ind

#Parametre beta (formula 4 from Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)

#Cross section of the interaction photon-photon (formula 1 from Gould's article) (cm^2)
def cross_section(s, r0):
    beta = beta_func(s)
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#Parametres
r0 =  2.818E-13 #classical elctron radius (cm)
mc2 = 510.9989461e+3 #electron mass (eV)
Ee = np.linspace(mc2, 1e7, 10000) #energy in the CM frame

[s, ind] = parametre_s(Ee, mc2)
s = s[ind[0]]

sigma = cross_section(s, r0)
plt.plot(Ee[ind[0]]*1e-6, sigma)
plt.title('Cross section for different energy in the CM frame')
plt.xlabel(r'$E_{cm,e}$''(MeV)')
plt.ylabel(r'$\sigma_{int}$''(cm' r'$^2$'')')
plt.show()
