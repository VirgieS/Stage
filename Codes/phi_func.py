#librairies
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

s0 = np.linspace(1.001, 9.999, 1000) #parameter s0 for 1<s0<10

#beta function (formula 4 from the Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)

#cross section for the interaction gamma-gamma (formula 1 from the Gould's article) (cm^2)
def cross_section(beta):
    r0 =  2.818E-13 #classical electron radius (cm)
    return 1/2.0*np.pi*r0**2*(1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#phi function (formula 9 from the Gould's article)

def integrand(s):
    r0 =  2.818E-13 #classical electron radius (cm)
    beta = beta_func(s)
    sigma = cross_section(beta) #calculation of the total cross section
    sigmab = 2*sigma/(np.pi*r0**2) #dimensionless cross section
    return s*sigmab

n=len(s0)
phis = np.zeros_like(s0)

for i in range (0,n):
    phis[i] = quad(integrand, 1, s0[i])[0] #integration from 1 to s0 for each s0

#Graph phi(s0)/(s0-1)
plt.title('Graph of the phi function')
plt.plot(s0, phis/(s0-1))
plt.xlabel('Parameter s0')
plt.ylabel('phi/(s0-1)')
plt.show()
