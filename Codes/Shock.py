"""
Computation of the gamma emission after the absorption for a shock in a binary system where the gamma photons only interact with the photons of the white dwarft.

Parameters that we can give for this code are:
    alpha_o         : polar angle of the observator in the orbital frame (rad)
    beta_o          : colatitude of the observator (rad)
    R_WD            : radius of the WD (AU)
    T_WD            : temperature of the WD (K)

Parameters loaded from a Pickle:
    time            : vector in days
    r_gamma         : vector given in AU
    E               : energy of the gamma photon in MeV

Parameters for the integration :
    beta_gamma      : linspace vector from 0 to pi/2
    delta_beta      : step for beta_gamma
    alpha_gamma     : only two values, 0 and pi (simplification)
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
alpha_o = 0                                                     # polar angle of the observator (rad)
beta_o = np.pi/2                                                # colatitude of the observator (rad)

    # Load data

        # Path to result directories and run ID
path= '/home/vivi/Documents/IRAP/Stage/Codes/'
#path= '/Users/stage/Documents/Stage/Codes/'
runid='Datas'

        # Values
hydro_data,gamma_data = load_results(path+runid)

    # time scale
time = hydro_data[0]['Time'] * day2sec      # in sec

    # shock coordinate
r_gamma = hydro_data[0]['Rshock'] * AU2cm                       # in cm
delta_beta = 0.01                                               # delta(beta) in rad
beta_gamma = np.linspace(0, np.pi/2, int(np.pi/2/delta_beta))   # colatitude of the gamma-source (rad)
alpha_gamma = np.array([0, np.pi])                              # polar angle of the gamma-source (rad)

    # Energy of the gamma photon
E_MeV = gamma_data.gam                	    # in MeV
E = E_MeV*MeV2erg                           # in erg

    # For WD
R_WD = 0.5 * AU2cm                          # radius of WD (cm)
T_WD = 10000                                # temperature of WD (K)

    # Parameters for the integration
L = 30 * AU2cm                                          # maximum length for the integration about z (cm)
step_z = 0.5 * AU2cm                                    # step for z (cm)
z = np.linspace(0, L, int(L/step_z))                    # position along the line of sight (cm)
step_phi = 0.1                                          # step for phi (rad)
phi = np.linspace(0, 2*np.pi, int(2*np.pi/step_phi))    # angle polar of the one source (cm)

# Computes the transmitted luminosity

Lum_gamma_trans = np.zeros_like(E)                        # initialization

theta_max_WD = np.arcsin(R_WD/r_gamma[100])             # for the condition of eclipse

for j in range (len(beta_gamma)):

    for k in range (len(alpha_gamma)):

        b_WD, z_WD, condition_WD = compute_WD(beta_gamma[j], beta_o, alpha_gamma[k], alpha_o, r_gamma[50], theta_max_WD)

        # if eclipse, then transmittance is nul else transmittance can be computed
        if condition_WD :

            transmittance = 0

        else :

            tau = calculate_tau(E, z, phi, b_WD, R_WD, T_WD, z_WD)
            transmittance = np.exp(-tau)


        # luminosity
        Lum_gamma = gamma_data.spec_tot[50]                                                     # total luminosity of the shock (without absorption) (erg/s/eV)
        Lum_gamma_i = elementary_luminosity(beta_gamma[j], delta_beta)*Lum_gamma                # luminosity of the surface dS (without absorption)  (erg/s/eV)
        #transmittance = 1
        Lum_gamma_trans += Lum_gamma_i * transmittance                                          # total luminosity of the shock (with absorption)    (erg/s/eV)

        if beta_gamma[j] == 0:

            break

E_ev = E * MeV2eV
Lum_gamma_trans = Lum_gamma_trans * E_ev              # in erg/s
Lum_gamma = Lum_gamma * E_ev                        # in erg/s

# Plotting the figure

f = plt.figure()
ax = f.add_subplot(111)
plt.text(0.5, 0.5,u'$r_\gamma$ = '+str(round(r_gamma[50]/AU2cm,2))+' au and t ='+str(round(time[100]/day2sec,2))+' days', horizontalalignment='left',
     verticalalignment='center', transform = ax.transAxes)
plt.xscale('log')
plt.title('Gamma emission of a classical nova')
plt.xlabel(r'$E_\gamma$' ' (MeV)')
plt.ylabel(r'$L_\gamma$'' (E) (erg/s)')
plt.plot(E_MeV, Lum_gamma, label="without absorption")
plt.plot(E_MeV, Lum_gamma_trans, label="with absorption")
plt.legend(loc='best')
plt.show()
