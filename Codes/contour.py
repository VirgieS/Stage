# Librairies and Functions
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *
from Functions import *

# Parameters for the system

    # Position of the observator
alpha_o = 0                         # polar angle of the observator (rad)
beta_o = np.pi/2                    # colatitude of the observator (rad)

    # Positions of the gamma-source
    # WD is at (0,0) and RG at (16,0)
#delta = 0.5                         # step for each position in X and Z (au)

xmin = -10.0                        # in au
xmax = 30.0                         # in au

zmin = 0.0                          # in au
zmax = 40.0                         # in au

delta = 0.5                         # in ua

xvec = np.linspace(xmin, xmax, int((xmax-xmin)/delta))
zvec = np.linspace(zmin, zmax, int((zmax-zmin)/delta))

xvec_cm = xvec * AU2cm              # in cm
zvec_cm = zvec * AU2cm              # in cm

X, Z = np.meshgrid(xvec, zvec)      # grid of positions of the gamma-source (au)

# For WD
R_WD = 0.5 * AU2cm                  # radius of WD (cm)
T_WD = 10000                        # temperature of WD (K)

# For RG
R_RG = 2 * AU2cm                    # radius of RG (cm)
T_RG = 3000                         # temperature of RG (K)

d_orb = 16 * AU2cm                  # orbital separation (cm)

# Energy of the gamma-photon
E = 1e9/erg2kev                     # erg (1 TeV)
E_tev = E*erg2kev*1e-9              # TeV

# Parameters for the integration
    # Integration over z
step_z = 0.1 * AU2cm                                                        # step for z
Lmin = 5                                                                    # choosen L min at 10 au
Lmax = 100                                                                  # choosen L max at 100 au
step_L = 0.5                                                                # step between each L-value
L = np.linspace(Lmin, Lmax, int((Lmax-Lmin)/step_L) + 1) * AU2cm            # maximum length for the integration about z (cm)
z = np.linspace(0, L[0], int(L[0]/step_z))                                  # position along the line of sight (cm)

    # Integration over phi
step_phi = 0.1                                                              # step for phi
phi = np.linspace(0, 2*np.pi, int(2*np.pi/step_phi))                        # angle polar of the one source (rad)

# Calculation of the transmittance

trans = np.zeros((len(zvec), len(xvec)), float)

for i in range (len(xvec)):

    for j in range (len(zvec)):

        [alpha_gamma, beta_gamma, r_gamma] = gamma(xvec_cm[i], zvec_cm[j])

        r_RG = np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(beta_gamma)*np.sin(alpha_gamma) + r_gamma**2)

        theta_max_WD = np.arcsin(R_WD/r_gamma)
        theta_max_RG = np.arcsin(R_RG/r_RG)

        [b_WD, z_WD, condition_WD] = compute_WD(beta_gamma, beta_o, alpha_gamma, alpha_o, r_gamma, theta_max_WD)
        [b_RG, z_RG, condition_RG] = compute_RG(beta_gamma, beta_o, alpha_gamma, alpha_o, r_gamma, d_orb, theta_max_RG)
        print(j,i)

        if condition_WD or condition_RG:

            transmittance = 0

        else :

            tau_WD = calculate_tau_L(E, z, phi, b_WD, R_WD, T_WD, z_WD)
            tau_RG = calculate_tau_L(E, z, phi, b_RG, R_RG, T_RG, z_RG)
            tau_tot = tau_WD + tau_RG
            transmittance = np.exp(-tau_tot)

        trans[j,i] = transmittance

WD = plt.Circle((0, 0), R_WD/AU2cm, color='black')
RG = plt.Circle((d_orb/AU2cm, 0), R_RG/AU2cm, color='r')

plt.xlabel(r'$X$'' coordinate (au)')
plt.ylabel(r'$Z$'' coordinate (au)')
plt.title('Transmittance of a gamma-photon at '+str(round(E_tev,2))+' TeV for each position')
plt.contourf(X, Z, trans, 100)
plt.colorbar()

fig = plt.gcf()
ax = fig.gca()

ax.add_artist(WD)
ax.add_artist(RG)

plt.savefig('Map_1TeV.eps')

plt.show()
