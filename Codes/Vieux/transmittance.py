import matplotlib.pyplot as plt
import numpy as np

#Function to calculate rho
def rho_func(R, Rs, alpha):
    return np.sqrt(R**2 + Rs**2 - 2*R*Rs*np.cos(alpha))

#Function to calculate theta et phi2
def theta_2(R, Rs, alpha, z):
    rho = rho_func(R, Rs, alpha)
    sin_theta2 = rho*(rho**2 + z**2)
    return sin_theta2, -np.sqrt(1 - sin_theta2**2)

def phi_2(R, Rs, alpha, z):
    rho = rho_func(R, Rs, alpha)
    return -(Rs*np.sin(alpha))/rho, -(rho**2 + R**2 -Rs**2)/(2*R*rho)


#Function to calulate theta
def theta(theta1, phi1, R, Rs, alpha, z):
    [sin_theta2, cos_theta2] = theta_2(R, Rs, alpha, z)
    [sin_phi2, cos_phi2] = phi_2(R, Rs, alpha, z)
    cos_phi12 = np.cos(phi1)*cos_phi2 + np.sin(phi1)*sin_phi2
    return np.cos(theta1)*cos_theta2 + np.sin(theta1)*sin_theta2*cos_phi12

#parametre s (formula 3 from Gould's article)
def parametre_s(Ee, mc2):
    s = (Ee/mc2)**2
    ind = np.where(s>=1) #for pair production to occur, s>=1 and if s=1, it is the threshold condition
    return s, ind

#Parametre beta (formula 4 from Gould's article)
def beta_func(s):
    return np.sqrt(1-1/s)

#Cross section of the interaction photon-photon (formula 1 from Gould's article) (cm^2)
def cross_section(s):
    r0 =  2.818E-13 #classical elctron radius (cm)
    beta = beta_func(s)
    return 1/2.0 * np.pi * r0**2 * (1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

#Function to calculate the differential number density of background photon
def density(1):
    return 0     #it depends on the photons that we consider
     
#Integrand
def integrand(cos_theta, eps_c, mc2, dn):
    s = (eps_c/mc2)**2
    sigma = cross_section(eps_c)
    return dn*sigma*(1-cos_theta)

#Function to calculate the optical depth
def optical_depth(dn, sigma, theta):
    return integrate.nquad(integrand, [[0, L], [mc2, inf], [0, 4*np.pi]])



#Parametres
mc2 = 510.9989461e+3 #electron mass (eV)
