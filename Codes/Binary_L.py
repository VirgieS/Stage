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

# Librairies and functions
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import *

# Parameters for the system

    # Position of the observator
alpha_o = np.pi/4                   # polar angle of the observator (rad)
beta_o = np.pi/2                    # colatitude of the observator (rad)

    # Position of the gamma-source
alpha_gamma = 0                     # polar angle of the gamma-source (rad)
beta_gamma = np.pi/2                # colatitude of the gamma-source (rad)
r_gamma = 2 * AU2cm                 # distance to the gamma-source (cm)
r_gamma_au = r_gamma/AU2cm          # au

    # For WD
R_WD = 0.5 * AU2cm                  # radius of WD (cm)
T_WD = 10000                        # temperature of WD (K)

    # For RG
R_RG = 2 * AU2cm                    # radius of RG (cm)
T_RG = 3000                         # temperature of RG (K)

d_orb = 16 * AU2cm                  # orbital separation (cm)

# Parameters for the integration

    # Integration over z
step_z = 0.1 * AU2cm                                                       # step for z
Lmin = 5                                                                   # choosen L_min at 10 au
Lmax = 100                                                                 # choosen L_max at 100 au
step_L = 0.5                                                               # step between each L-value (au)
L = np.linspace(Lmin, Lmax, int((Lmax-Lmin)/step_L) + 1) * AU2cm           # maximum length for the integration about z (cm)

    # Integration over phi
step_phi = 0.1                                                             # step for phi
phi = np.linspace(0, 2*np.pi, int(2*np.pi/step_phi))                       # angle polar of the one source (rad)

# Energy of the gamma-photon
E = 1e10/erg2kev          	# erg
E_tev = E*erg2kev/TeV2kev    	# TeV

# Calculation of the transmittance

    # Initialization of tau for each source and for the system
tau_WD = np.zeros_like(L)
tau_RG = np.zeros_like(L)
tau = np.zeros_like(L)

    # Distance from the RG to the source gamma
r_RG = np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(beta_gamma)*np.sin(alpha_gamma) + r_gamma**2)

    # For the condition about the eclipse
theta_max_WD = np.arcsin(R_WD/r_gamma)
theta_max_RG = np.arcsin(R_RG/r_RG)

    # Parameters b, zb and condition for the WD and RG
[b_WD, z_WD, condition_WD] = compute_WD(beta_gamma, beta_o, alpha_gamma, alpha_o, r_gamma, theta_max_WD)
[b_RG, z_RG, condition_RG] = compute_RG(beta_gamma, beta_o, alpha_gamma, alpha_o, r_gamma, d_orb, theta_max_RG)

if condition_WD or condition_RG:

    print("There is an eclipse")

else :

    z = np.linspace(0, L[0], int(L[0]/step_z))              # position along the line of sight (cm)
	
    #optical depth for WD, RG and both
    tau_WD[0] = calculate_tau_L(E, z, phi, b_WD, R_WD, T_WD, z_WD)
    tau_RG[0] = calculate_tau_L(E, z, phi, b_RG, R_RG, T_RG, z_RG)
    tau[0] = tau_WD[0] + tau_RG[0]

    print(L[0]/AU2cm)
    print(tau_WD[0], tau_RG[0], tau[0])

    for i in range (1, len(L)):

        z = np.linspace(L[i-1], L[i], int((L[i]-L[i-1])/step_z))

        tau_WD[i] = tau_WD[i-1] + calculate_tau_L(E, z, phi, b_WD, R_WD, T_WD, z_WD)
        tau_RG[i] = tau_RG[i-1] + calculate_tau_L(E, z, phi, b_RG, R_RG, T_RG, z_RG)

        tau[i] = tau_WD[i] + tau_RG[i]

        print(L[i]/AU2cm)
        print(tau_WD[i], tau_RG[i], tau[i])

    R_WD_au = R_WD/AU2cm     # au
    R_RG_au = R_RG/AU2cm     # au
    d_orb_au = d_orb/AU2cm   # au
    L_au = L/AU2cm

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.text(0.5,0.5,u'T$_{WD}$ = %.2f K, R$_{WD}$ = %.2f au \nT$_{RG}$ = %.2f K, R$_{RG} =$ %.2f au \nd$_{orb}$ = %.2f au \n' r'$\alpha_o$' ' = %.2f, 'r'$\beta_o$'' = %.2f \n'r'$\alpha_\gamma$'' = %.2f, 'r'$\beta_\gamma$'' = %.2f \nr$_\gamma$ = %.2f au'  %(T_WD, R_WD_au, T_RG, R_RG_au, d_orb_au, alpha_o, beta_o, alpha_gamma, beta_gamma, r_gamma_au), horizontalalignment='left',
     verticalalignment='center', transform = ax.transAxes)
    plt.plot(L_au, tau_WD, label="WD")
    plt.plot(L_au, tau_RG, label="RG")
    plt.plot(L_au, tau, label="both")
    plt.xlabel('L (au)')
    plt.ylabel(r'$\tau_{\gamma \gamma}$')
    plt.title(u'Optical depth of 'r'$\gamma$' '-rays at %.2f TeV in interaction \n with photons from a binary stellar system' %E_tev)
    plt.legend(loc='best')
    plt.save('optical_depth_L.eps')
    plt.show()
"""
z = np.linspace(0, L[0], int(L[0]/step_z))              # position along the line of sight (cm)
tau_WD[0] = calculate_tau_L(E, z, phi, b_WD, R_WD, T_WD, z_WD)
tau_RG[0] = calculate_tau_L(E, z, phi, b_RG, R_RG, T_RG, z_RG)
tau[0] = tau_WD[0] + tau_RG[0]

print(L[0]/AU2cm)
print(tau_WD[0], tau_RG[0], tau[0])

for i in range (1, len(L)):

    z = np.linspace(L[i-1], L[i], int((L[i]-L[i-1])/step_z))

    tau_WD[i] = tau_WD[i-1] + calculate_tau_L(E, z, phi, b_WD, R_WD, T_WD, z_WD)
    tau_RG[i] = tau_RG[i-1] + calculate_tau_L(E, z, phi, b_RG, R_RG, T_RG, z_RG)

    tau[i] = tau_WD[i] + tau_RG[i]

    print(L[i]/aucm)
    print(tau_WD[i], tau_RG[i], tau[i])

R_WD_au = R_WD/AU2cm     # au
R_RG_au = R_RG/AU2cm     # au
d_orb_au = d_orb/AU2cm   # au
L_au = L/AU2cm

f = plt.figure()
ax = f.add_subplot(111)
plt.text(0.5,0.5,u'T$_{WD}$ = %.2f K, R$_{WD}$ = %.2f au \nT$_{RG}$ = %.2f K, R$_{RG} =$ %.2f au \nd$_{orb}$ = %.2f au \n' r'$\alpha_o$' ' = %.2f, 'r'$\beta_o$'' = %.2f \n'r'$\alpha_\gamma$'' = %.2f, 'r'$\beta_\gamma$'' = %.2f \nr$_\gamma$ = %.2f au'  %(T_WD, R_WD_au, T_RG, R_RG_au, d_orb_au, alpha_o, beta_o, alpha_gamma, beta_gamma, r_gamma_au), horizontalalignment='left',
 verticalalignment='center', transform = ax.transAxes)
plt.plot(L_au, tau_WD, label="WD")
plt.plot(L_au, tau_RG, label="RG")
plt.plot(L_au, tau, label="both")
plt.xlabel('L (au)')
plt.ylabel(r'$\tau_{\gamma \gamma}$')
plt.title(u'Optical depth of 'r'$\gamma$' '-rays at %.2f TeV in interaction \n with photons from a binary stellar system' %E_tev)
plt.legend(loc='lower right')
plt.savefig('Bla.png')
plt.show()
"""
