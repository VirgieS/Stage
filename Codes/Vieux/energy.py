import matplotlib.pyplot as plt
import numpy as np

#spectral radiance of a black body (in wavelength)
def B_lambda(l, T, h, c):
    k = 1.380658e-23/1.602e-16 #Boltzmann's constant in eV
    return 2*h*c**2/(l**5)*1/(np.exp(h*c/(l*k*T))-1)

def Wien(T):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV*s
    c = 2.99792458e+10 #light speed in cm/s
    b = 2.8977729e-1 #Wien's displacement constant (cm*K)
    return b/T

T = 2.7 #temperature in K
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)

E = 1e14 #energy of gamma photon (keV)
eps = np.linspace(mc2**2/E, 10 * k * T, 100) #energy of background photon
lxmin = np.log10(eps.min())
lxmax = np.log10(eps.max())
eps = 10**np.linspace(lxmin, lxmax, 100)
l = h * c/eps

Blmax = Wien(T)
Bl = B_lambda(l, T, h, c)
Blmax1 = 0
for i in range(len(Bl)):
    if Bl[i] > Blmax1:
        Blmax1 = Bl[i]
    else:
        break

plt.plot(l[i-1], Blmax, 'r')
plt.plot(l, Bl)
plt.show()
