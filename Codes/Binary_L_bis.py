"""
For a binary stellar system.

Parameters that we can give for this code are:
    alpha_o         : polar angle of the observator in the orbital frame (rad)
    beta_o          : colatitude of the observator in the orbital frame (rad)
    alpha_gamma     : polar angle of the gamma-source in the orbital frame (rad)
    beta_gamma      : colatitude of the gamma-source in the orbital frame (rad)
    r_gamma         : distance between the WD and the gamma-source (au)
    R_WD            : radius of the WD (au)
    T_WD            : temperature of the WD (K)
    R_RG            : radius of the RG (au)
    T_RG            : temperature of the RG (K)

Our coordinate to compute tau are theta, angle formed by the ray from the secundary source and the direction of the centre from a position on the line of sight and phi, polar angle around this axis.
"""

# Librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factor import *
from Functionsbis import *

# Parameters for the system

    # Position of the observator
alpha_o = np.pi/4                  # polar angle of the observator (rad)
beta_o = np.pi/2                    # colatitude of the observator (rad)

    # Position of the gamma-source
alpha_gamma = np.pi                 # polar angle of the gamma-source (rad)
beta_gamma = np.pi/2                # colatitude of the gamma-source (rad)
r_gamma = 2 * aucm                  # distance to the gamma-source (cm)
r_gamma_au = r_gamma/aucm           # in au

    # For WD
R_WD = 0.5 * aucm                   # radius of WD (cm)
T_WD = 10000                        # temperature of WD (K)

    # For RG
R_RG = 2 * aucm                     # radius of RG (cm)
T_RG = 3000                         # temperature of RG (K)

d_orb = 16 * aucm                   # orbital separation (cm)

r_RG = np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(beta_gamma)*np.sin(alpha_gamma) + r_gamma**2)

theta_max_WD = np.arcsin(R_WD/r_gamma)
theta_max_RG = np.arcsin(R_RG/r_RG)

# Parameters for the integration
step_L = 0.1 * aucm                                     # step for integration over z
L = np.logspace(log10(10), log10(100), int(90/5)) * aucm        # maximum length for the integration about z (cm)
step_phi = 0.1                                          # step for integration over phi
phi = np.linspace(0, 2*np.pi, int(2*np.pi/step_phi))    # angle polar of the one source (rad)

# For the vector eps and E
number_bin_E = 40

# Energy of the gamma-photon
E = 1e9/ergkev          # erg
E_tev = E*ergkev*1e-9   # TeV

# Calculation of the transmittance
tau = np.zeros_like(L)

for i in range (len(L)):

    z = np.linspace(0, L[i], int(L[i]/step_L))          # position along the line of sight (cm)
    print(i)

    [b_WD, z_WD, condition_WD] = compute_WD(beta_gamma, beta_o, alpha_gamma, alpha_o, r_gamma, theta_max_WD)
    [b_RG, z_RG, condition_RG] = compute_RG(beta_gamma, beta_o, alpha_gamma, alpha_o, r_gamma, d_orb, theta_max_RG)

    tau_WD = calculate_tau_L(E, z, phi, L[i], b_WD, R_WD, T_WD, z_WD, condition_WD)
    tau_RG = calculate_tau_L(E, z, phi, L[i], b_RG, R_RG, T_RG, z_RG, condition_RG)

    tau[i] = 1.0/2 * np.pi * r0**2 * (tau_WD + tau_RG)

R_WD_au = R_WD/aucm     # au
R_RG_au = R_RG/aucm     # au
d_orb_au = d_orb/aucm   # au

L_au = L/aucm
plt.plot(L_au, tau)
plt.xlabel('L (au)')
plt.ylabel(r'$\tau_{\gamma \gamma}$')
plt.title(u'Optical depth of 'r'$\gamma$' '-rays at %.2f TeV in interaction \n with photons from a binary stellar system' %E_tev)
#plt.text(100, 0.5,u'T$_{WD}$ = %.2f K, R$_{WD}$ = %.2f au \nT$_{RG}$ = %.2f K, R$_{RG} =$ %.2f au \nd$_{orb}$ = %.2f au \n' r'$\alpha_o$' ' = %.2f, 'r'$\beta_o$'' = %.2f \n'r'$\alpha_\gamma$'' = %.2f, 'r'$\beta_\gamma$'' = %.2f \nr$_\gamma$ = %.2f au' %(T_WD, R_WD_au, T_RG, R_RG_au, d_orb_au, alpha_o, beta_o, alpha_gamma, beta_gamma, r_gamma_au))
#plt.legend(loc='lower right')
plt.show()
