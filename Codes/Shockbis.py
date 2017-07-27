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
from Conversion_factors import *
from Functions import *
from Load import load_results

# Parameters for the system

    # Fixed position of the observator
alpha_o = 0                         # polar angle of the observator (rad)
beta_o = np.pi/2                    # colatitude of the observator (rad)

if __name__ == '__main__':

    """

	Plot results of nova acceleration+radiation simulation series

	Notes:
	- ...
    """

    # Path to result directories and run ID
    path='/Users/stage/Documents/Stage/Codes/'
    runid='Datas'

    # Load data
    hydro_data,gamma_data = load_results(path+runid)

    # time scale
    time = hydro_data[0]['Time'] * day2sec      # in sec

    # shock coordinate
    r_gamma = hydro_data[0]['Rshock'] * AU2cm                       # in cm
    step_beta = 0.5                                                 # delta(beta) in rad
    beta_gamma = np.linspace(0, np.pi/2, int(np.pi/2/step_beta))    # colatitude of the gamma-source (rad)
    delta_beta = step_beta/3
    alpha_gamma = np.array([0, np.pi])                              # polar angle of the gamma-source (rad)

    # Energy of the gamma photon
    E = gamma_data.gam * MeV2erg                # in erg
    E_MeV = E/MeV2erg                           # in MeV

    # For WD
    R_WD = 0.5 * AU2cm                          # radius of WD (cm)
    T_WD = 10000                                # temperature of WD (K)

    # Parameters for the integration
    L = 30 * AU2cm                                          # maximum length for the integration about z (cm)
    step_z = 0.5 * AU2cm                                    # step for z (cm)
    z = np.linspace(0, L, int(L/step_z))                    # position along the line of sight (cm)
    step_phi = 0.1                                          # step for phi (rad)
    phi = np.linspace(0, 2*np.pi, int(2*np.pi/step_phi))    # angle polar of the one source (cm)

    L_gamma_trans = np.zeros_like(time)

    for i in range (len(time)):

        if r_gamma[i] < R_WD :

            continue

        else :

            theta_max_WD = np.arcsin(R_WD/r_gamma[i])

        for j in range (len(beta_gamma)):

            for k in range (len(alpha_gamma)):

                b_WD, z_WD, condition_WD = compute_WD(beta_gamma[j], beta_o, alpha_gamma[k], alpha_o, r_gamma[i], theta_max_WD)

                if condition_WD :

                    transmittance = 0

                else :

                    tau = calculate_tau(E, z, phi, b_WD, R_WD, T_WD, z_WD)
                    print(len(tau))
                    transmittance = np.exp(-tau)

                # luminosity
                L_gamma = gamma_data.spec_tot[i]
                print(len(L_gamma))

                #L_gamma_i = elementary_luminosity(beta_gamma[j], delta_beta, r_gamma[i], L_gamma)

                #L_gamma_trans[i] = L_gamma_trans[i] + L_gamma_i * transmittance

        #print(L_gamma_trans)
