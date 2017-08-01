"""
For a binary stellar system. Here only the white draft has an importance, so we consider only this.

Parameters that we can give for this code are:
    alpha_o         : polar angle of the observator in the orbital frame (rad)
    beta_o          : colatitude of the observator (rad)
    v_gamma         : velocity of the shock (km/s)
    time            : time since the shock (d)
    R_WD            : radius of the WD (au)
    T_WD            : temperature of the WD (K)

Our coordinate to compute tau are theta, angle formed by the ray from the secundary source and the direction of the centre from a position on the line of sight and phi, polar angle around this axis.
"""

# Librairies
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factor import *
from Functions import *
from LoadDataVirginie import load_results

# Parameters for the system

    # Fixed position of the observator
alpha_o = 0                         # polar angle of the observator (rad)
beta_o = np.pi/2                    # colatitude of the observator (rad)

    # Position of the gamma-source
v_gamma = 1000 * km2cm                   # velocity of the shock (cm/s)
time_min = 0                            # minimum of the time scale (d)
time_max = 20                           # maximum of the time scale (d)
delta_time = 0.5                        # step of time (d)
time = np.linspace(time_min, time_max, int((time_max - time_min)/delta_time)) * day2sec   # time scale (s)
r_gamma_0 = np.zeros_like(time)         # initial radius of the shock (cm)
r_gamma = r_gamma_0 + v_gamma * time    # distance to the gamma-source (cm)
r_gamma_au = r_gamma/aucm               # in au
beta_gamma = np.linspace(0, np.pi/2, 10) # colatitude of the gamma-source (rad)
alpha_gamma = np.array([0, np.pi])         # polar angle of the gamma-source (rad)

    # For WD
R_WD = 0.5 * AU2cm                   # radius of WD (cm)
T_WD = 10000                        # temperature of WD (K)
L_WD = 3e38                         # luminosity of WD (erg/s)

# Parameters for the integration
L = 30 * AU2cm                                           # maximum length for the integration about z (cm)
step_z = 0.5 * AU2cm                                     # step for z (cm)
z = np.linspace(0, L, int(L/step_z))                    # position along the line of sight (cm)
step_phi = 0.1                                          # step for phi (rad)
phi = np.linspace(0, 2*np.pi, int(2*np.pi/step_phi))    # angle polar of the one source (cm)

# For the vector eps and E
number_bin_E = 30

# Energy of the gamma-photon
Emin = 1e7/ergkev                   # Emin = 1e-2 TeV (erg)
Emax = 1e14/ergkev                  # Emax = 1e5 TeV (erg)
E = np.logspace(log10(Emin), log10(Emax), number_bin_E)  # erg
E_tev = E*ergkev*1e-9               # TeV

trans = np.zeros_like(time)

for i in range (len(time)):

    if r_gamma[i] < R_WD :

        continue

    else :

        theta_max_WD = np.arcsin(R_WD/r_gamma[i])
        transmittance = 0

        for j in range (len(beta_gamma)):

            for k in range (len(alpha_gamma)):

                [b_WD, z_WD, condition_WD] = compute_WD(beta_gamma[j], beta_o, alpha_gamma[k], alpha_o, r_gamma[i], theta_max_WD)

                if condition_WD :

                    transmittance += 0

                else :

                    tau_WD = calculate_tau(E, z, phi, b_WD, R_WD, T_WD, z_WD)
                    tau_tot = tau_WD
                    transmittance += np.exp(-tau_tot)

        trans[i] = transmittance

R_WD_au = R_WD/AU2cm     # au
print(trans)
