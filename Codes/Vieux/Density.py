import matplotlib.pyplot as plt
import numpy as np

def density_eps(eps, T):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
    c = 2.99792458e+10 #light speed in cm/s

    #spectral radiance of a black body (in wavelength)
    def B_eps(eps, T, h, c):
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV
        return 2*eps**3/(h**3 * c**2) *1/(np.exp(eps/(k * T))-1)

    Beps = B_eps(eps, T, h, c)
    return Beps * 1/(eps * c)


def density_l(l, T):
    h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
    c = 2.99792458e+10 #light speed in cm/s
    eps = h*c/l

    #spectral radiance of a black body (in wavelength)
    def B_lambda(l, T, h, c):
        k = 1.380658e-23/1.602e-16 #Boltzmann's constant in eV
        return 2*h*c**2/(l**5)*1/(np.exp(h*c/(l*k*T))-1)

    Bl = B_lambda(l, T, h, c)
    return Bl * h/(eps**3)

T = 2.7 #temperature in K
k = 1.380658e-23/1.602e-16 #Boltzmann's constant in keV
h = 6.6260755e-34/1.602e-16 #Planck's constant in keV
c = 2.99792458e+10 #light speed in cm/s
mc2 = 510.9989461 #electron mass (keV)

print(h * c/(10*k*T))

E = 1e14 #energy of gamma photon (keV)
eps = np.linspace(mc2**2/E, 10 * k * T, 100) #energy of background photon
lxmin = np.log10(eps.min())
lxmax = np.log10(eps.max())
eps = 10**np.linspace(lxmin, lxmax, 100)
l = h * c/eps

dneps = density_eps(eps, T)
dnl = density_l(l, T)
ind = np.where(dnl>0)
print(dnl)
dnl = dnl[ind[0]]
l = l[ind[0]]
plt.plot(l, dnl)
plt.plot(h * c/eps, dneps)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency ' r'$\nu$'' (Hz)')
plt.plot('Density (cm'r'$^{-3}$' ' sr'r'$^{-1}$' ' keV'r'$^{-1}$'')')
plt.show()
