"""
Now we consider that the star, not along the line of sight, is a sphere of uniform brightness (black body at T) with a radius R.

Our coordinate are theta, angle formed by the ray from the star and the direction of the centre from a position on the line of sight and phi, polar angle around this axis.

Parameters that we can give for this code are:
    L      : distance to the gamma-source (au)
    zb     : position along the line of sight nearly the star (au)
    b      : impact parameter (au)
    R      : radius of the star (au)
    T      : temperature of the star (K)
"""

# Librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import *

# For the vector eps and E
number_bin_E = 40

# Parameters for the code
L = 20 * AU2cm                          # the distance to the gamma-source (cm)
zb = 10 * AU2cm                         # position along the line of sight nearly the star (cm)
b = 5 * AU2cm                          # impact parameter (cm)
R = 100 * Rsun2cm                       # radius of the star (cm)
T = np.array([3000, 6000, 10000])       # temperature of the star (K)
z = np.linspace(0, L, 100)              # position along the line of sight (cm)
phi = np.linspace(0, 2*np.pi, 10)       # angle polar

# Energy of the gamma-photon
Emin = 1e-2*TeV2erg                     # Emin = 1e-2 TeV (erg)
Emax = 1e5*TeV2erg                      # Emax = 1e5 TeV (erg)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E)     # erg
E_tev = E/TeV2erg                                           # TeV

f = plt.figure()
ax = f.add_subplot(111)

# Calculation of the transmittance for each temperature
for i in range (len(T)):
    tau = calculate_tau(E, z, phi, b, R, T[i], zb)
    plt.plot(E_tev, np.exp(-tau), label = "T = %.2f K" %T[i])

b_au = b/AU2cm                                              # au
L_au = L/AU2cm                                              # au
R_Rsun = R/Rsun2cm                                          # Rsun

plt.xscale('log')
plt.xlabel(r'$E_\gamma$' '(TeV)')
plt.ylabel(r'$\exp(-\tau_{\gamma \gamma})$')
plt.title(u'Transmittance of VHE 'r'$\gamma$' '-rays in interaction \n with a star with a radius '+str(round(R_Rsun,2))+' 'r'$R_\bigodot$')
plt.text(0.5, 0.5,'L = %.2f au, b = %.2f au' %(L_au, b_au))
plt.legend(loc='best')
plt.show()
