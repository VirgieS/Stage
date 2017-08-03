#librairies
import numpy as np
from math import *
from Physical_constants import *
from Conversion_factors import *

def integration_log(x, y):

    """
    Return the integrale of a function in log-log scale

    Parameters :
        x           : abscisse of the function
        y           : function that we integrate along x : y = f(x)
    """

    #Looking for a and b for y = a*x^b
    def calculate_ab(xi, xf, yi, yf):
        logxi = np.log(xi)
        logxf = np.log(xf)
        logyi = np.log(yi)
        logyf = np.log(yf)
        b = (logyf - logyi)/(logxf - logxi)
        loga = logyi - b*logxi
        a = np.exp(loga)
        a = np.nan_to_num(a)
        return a, b

    #Calculate deltaS from deltaS = int from xi to xf a*x^b
    def delta_S(xi, xf, yi, yf):
        [a, b] = calculate_ab(xi, xf, yi, yf)
        return a/(b+1)*(xf**(b+1) - xi**(b+1))

    integral = 0

    # Because the function integral_log works only if there is more than two elements not zero
    idx=(y > 0.0)
    #idy=(y < 0.0) # only used if the function has negative value
    idt = idx #+ idy
    if sum(idt) > 2:

        x = x[idt]
        y = y[idt]

        #Calculate total integral from init to final a*x^b
        deltaS = 0

        for i in range (1, len(x)):
            deltaS = delta_S(x[i-1], x[i], y[i-1], y[i])
            integral = integral + deltaS

            integral = np.nan_to_num(integral)

    return integral

def compute_WD(psi_gamma, psi_o, phi_gamma, phi_o, r_gamma, theta_max_WD):

    """
    Return z_WD, b_WD and condition_WD

    Parameters :
        psi_gamma       : colatitude of the gamma-source (rad)
        psi_o           : colatitude of the observator (rad)
        phi_gamma       : polar angle of the gamma-source (rad)
        phi_o           : polar angle of the observator (rad)
        r_gamma         : distance to the gamma-source (cm)
	theta_max	: maximal angle for theta from the gamma source (rad)
    """

    condition_WD = False		# first we suppose that there is no eclipse

    # Compute of the angle gamma_WD (see figure) (rad)
    gamma_WD = np.arccos(-(np.sin(psi_gamma)*np.sin(psi_o)*np.cos(phi_gamma)*np.cos(phi_o) + np.sin(psi_gamma)*np.sin(psi_o)*np.sin(phi_gamma)*np.sin(phi_o) + np.cos(psi_gamma)*np.cos(psi_o)))

    # Impact parameter (cm)
    b_WD = r_gamma * np.sin(gamma_WD)

    # Position along the line of sight closely the WD (cm)
    if gamma_WD <= np.pi/2:

        z_WD = np.sqrt(r_gamma**2 - b_WD**2)

    else:

        z_WD = - np.sqrt(r_gamma**2 - b_WD**2)

    if gamma_WD <= theta_max_WD: # if there is an eclispe

            condition_WD = True

    return b_WD, z_WD, condition_WD

def compute_RG(psi_gamma, psi_o, phi_gamma, phi_o, r_gamma, d_orb, theta_max_RG):

    """
    Return z_RG, b_RG qnd condition_RG

    Parameters :
        psi_gamma       : colatitude of the gamma-source (rad)
        psi_o           : colatitude of the observator (rad)
        phi_gamma       : polar angle of the gamma-source (rad)
        phi_o           : polar angle of the observator (rad)
        r_gamma         : distance to the gamma-source (cm)
        d_orb           : orbital separation (cm)
	theta_max	: maximal angle for theta from the gamma source (rad)
    """

    condition_RG = False		# first we suppose that there is no eclipse

    # Compute of the angle gamma_RG (see figure) (rad)
    gamma_RG = np.arccos((d_orb*np.sin(psi_o)*np.cos(phi_o) - r_gamma * (np.sin(psi_gamma)*np.sin(psi_o)*np.cos(phi_gamma)*np.cos(phi_o) + np.sin(psi_gamma)*np.sin(psi_o)*np.sin(phi_gamma)*np.sin(phi_o) + np.cos(psi_gamma)*np.cos(psi_o)))/(np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(psi_gamma)*np.cos(phi_gamma) + r_gamma**2)))

    # Impact parameter (cm)
    b_RG = sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(psi_gamma)*np.sin(phi_gamma) + r_gamma**2) * np.sin(gamma_RG)

    # Position along the line of sight closely the RG (cm)
    if gamma_RG <= np.pi/2:

        z_RG = np.sqrt(d_orb**2 - 2*r_gamma*d_orb*np.sin(psi_gamma)*np.sin(phi_gamma) + r_gamma**2- b_RG**2)

    else:

        z_RG = - np.sqrt(r_gamma**2 - b_RG**2)

    if gamma_RG <= theta_max_RG: # if there is an eclispe

            condition_RG = True

    return b_RG, z_RG, condition_RG

def distance(zb, z, b):

    """
    Return for each position z along the line of sight the distance between this position and the centre of the star (cm)

    Parameters:
        zb       : position on the line of sight closely the star (cm)
        z        : position along the line of sight (cm)
        b        : impact parameter (cm)
    """

    return np.sqrt((zb - z)**2 + b**2)

def angle_alpha(b, D, z, zb, theta, phi):

    """
    Return the cosinus of the angle (alpha) between the two momenta of the two photons in the observer's frame

    Parameters :
        b     : impact parameter (cm)
        D     : distance to the star from each position along the line of sight (cm)
        z     : position along the line of sight (cm)
        zb    : position along the line of sight nearly the star (cm)
        theta : angle formed by the ray (from the star) and the straight line connecting the centre of the star and a position z on the line of sight (rad)
            careful theta = 0 when the ray 'comes' from the centre of the star and theta = theta_max when the ray is tangent to the sphere
        phi   : angle around the direction between the centre of the star and the position along the line of sight (rad)
    """

    #Return the cosinus of the angle formed by the direction between the centre of the star and the position along the line of sight, and the line of sight (rad)
    def angle_beta(b, D, z, zb):

	"""
	Return the angle (betq) formed by the direction from the position along the line of sight and the centre of the star and the direction of the gamma photon

	Parameters:
	    b     : impact parameter (cm)
            D     : distance to the star from each position along the line of sight (cm)
            z     : position along the line of sight (cm)
            zb    : position along the line of sight nearly the star (cm)
	"""

        if z <= zb:
            beta = np.arcsin(b/D)

        else:
            beta = np.pi - np.arcsin(b/D)

        return beta

    beta = angle_beta(b, D, z, zb)

    return  - np.sin(beta) * np.sin(theta) * np.sin(phi) - np.cos(beta) * np.cos(theta)

def density_n(eps, T, theta):

    """
    Density of the photons is not isotropic : dn = Bnu * cos(theta)/(c * h**2 *nu) (cm^3/sr/erg)

    Parameters:
        eps   : energy of the target-photon (erg)
        theta : angle formed by the ray (from the star) and the line connecting the centre of the star and a position z on the line of sight (rad)
        T     : temperature of the star (K)
    """

    nu = eps/hp

    def planck(nu, T):

        return (2*hp*(nu**3))/(cl**2) * 1/(np.exp((hp*nu)/(kb*T)) - 1)

    Bnu = planck(nu, T)

    return Bnu/(cl * hp**2 * nu) * np.cos(theta) #return dn in cm^3/sr/erg

def gamma(X, Z):

    """
    Return the angles alpha, beta and the distance from the WD to the gamma-source in the perpendicular plane of the orbital plane

    Parameters:
        X   : X coordinate of the gamma-source (cm)
        Z   : Z coordinate of the gamma-source (cm)

    """

    r = np.sqrt(X**2 + Z**2)		# cm

    # azimutal angle (rad): in the plane alpha is only 0 or pi
    if X >=0:

        alpha = 0

    else:

        alpha = np.pi

    # Polar angle (rad)
    beta = np.arccos(Z/r)

    return alpha, beta, r

def f(theta, phi, eps, z, D, b, R, E, T, zb):

    """
    Return the function for the integration in phi : f = dn * sigma * (1 - cos(alpha)) * sin(theta)
        where dn is the density of photons, sigma is the dimensionless cross section of the interaction,
              alpha is the between the two momenta of the two photons in the observer's frame

    Parameters:
        theta   : angle formed by the ray (from the star) and the straight line connecting the centre of the star and a position z on the line of sight (rad)
        phi     : angle around the direction between the centre of the star and the position along the line of sight (rad)
        eps     : energy of the target-photon (keV)
        z       : position along the line of sight (cm)
        D       : distance to the gamma-source (cm)
        b       : impact parameter (cm)
        E       : energy of the gamma-photon (erg)
        T       : temperature of the star (K)
        zb      : position along the line of sight nearly the star (cm)
    """

    cos_alpha = angle_alpha(b, D, z, zb, theta, phi)
    epsc = np.sqrt(eps * E/2 * (1 - cos_alpha))/(mc2/erg2kev) # epsc/(mc2 in erg)

    #First : sigma (dimensionless)
    def cross_section(epsc):

	"""
	Return the dimensionless cross section of the interaction gamma-gamma (formula 1 of Gould's article)

	Parameter:
	    epsc	: center-of-momentum system energy of a photon
	"""

        def parameter_s(epsc):

	    """
	    Return the parameter s (formula 3 of Gould's article)

	    Parameter:
	        epsc	: center-of-momentum system energy of a photon
	    """

            return epsc**2

        def parameter_beta(epsc):

	    """
	    Return the parameter beta (formula 4 of Gould's article)

	    Parameter:
	        epsc	: center-of-momentum system energy of a photon
	    """

            s = parameter_s(epsc)
            return np.sqrt(1 - 1/s)

        beta = parameter_beta(epsc)
        beta = np.nan_to_num(beta)

        return (1 - beta**2) * ((3 - beta**4) * np.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2))

    dn = density_n(eps, T, theta)	# differential density of target photons (cm^3/sr/erg)
    sigma = cross_section(epsc)		# dimensionless cross section of the interaction gamma-gamma

    return dn * sigma * (1 - cos_alpha) * np.sin(theta)

def calculate_tau(E, z, phi, b, R, T, zb):

    """
    Return tau(E) for all E
    Parameters :
        E           : energy of the gamma-photon (erg)
        z           : position along the line of sight (cm)
        phi         : angle around the direction between the centre of the star and the position along the line of sight (rad)
        zb          : position along the line of sight nearly the star (cm)
        L           : maximum distance for the integration about z (cm)
        b           : impact parameter (cm)
        R           : radius of the secundary source (cm)
        T           : temperature of the secundary source (cm)
    """

    integral = np.zeros_like(E)
    number_bin_eps = 40.0

    for i in range(len(E)): # integration over z

        integral_eps = np.zeros_like(z)

        # Energy of the target-photon (erg)
        epsmin = (mc2/erg2kev)**2/E[i]		# threshold condition for alpha = pi
        epsmax = 10*kb*T			# do not need to compute to infinity

        # Because epsmin must be lower than epsmax
        if epsmin > epsmax:

            continue

        else:

	    # we make a energy-grid with a constant energy step
            eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))

            print(i)

        for j in range (len(z)): # integration over eps

            integral_theta = np.zeros_like(eps)
            D = distance(zb, z[j], b)		# compute the distance to the star for one position z[j] (cm)
            theta_max = np.arcsin(R/D)		# compute theta_max for each position (rad)
            #step_theta = 0.001
            theta = np.linspace(0, theta_max, 10) #int(theta_max/step_theta))

            for l in range (len(eps)): # integration over theta

                integral_phi = np.zeros_like(theta)

                for m in range (len(theta)): # integration over phi

                    integrand = f(theta[m], phi, eps[l], z[j], D, b, R, E[i], T, zb)
                    integrand = np.nan_to_num(integrand)

                    integral_phi[m] = integration_log(phi, integrand)

                integral_theta[l] = integration_log(theta, integral_phi)

            integral_eps[j] = integration_log(eps, integral_theta) # you get d(tau)/dx

        integral[i] = integration_log(z, integral_eps) # you get tau

    return  1/2.0 * np.pi * r0**2 * integral#, step_theta

def calculate_tau_L(E, z, phi, b, R, T, zb):

    """
    Return tau(E) for each distance along the line of sight
    Parameters :
        E           : energy of the gamma-photon (erg)
		careful E is selected where there is absorption
        z           : position along the line of sight (cm)
        phi         : angle around the direction between the centre of the star and the position along the line of sight (rad)
        zb          : position along the line of sight nearly the star (cm)
        L           : maximum distance for the integration about z (cm)
        b           : impact parameter (cm)
        R           : radius of the secundary source (cm)
        T           : temperature of the secundary source (cm)
    """

    number_bin_eps = 20.0

    # integration over z

    integral_eps = np.zeros_like(z)

    # Energy of the target-photon (erg)
    epsmin = (mc2/erg2kev)**2/E
    epsmax = 10*kb*T

    eps = np.logspace(log10(epsmin), log10(epsmax), int(log10(epsmax/epsmin)*number_bin_eps))	# to have an energy-grid with a constant energy step

    for j in range (len(z)): # integration over eps

        integral_theta = np.zeros_like(eps)
        D = distance(zb, z[j], b)
        theta_max = np.arcsin(R/D)
        theta = np.linspace(0, theta_max, 10)

        for l in range (len(eps)): # integration over theta

            integral_phi = np.zeros_like(theta)

            for m in range (len(theta)): # integration over phi

                integrand = f(theta[m], phi, eps[l], z[j], D, b, R, E, T, zb)
                integrand = np.nan_to_num(integrand)

                integral_phi[m] = integration_log(phi, integrand)

            integral_theta[l] = integration_log(theta, integral_phi)

        integral_eps[j] = integration_log(eps, integral_theta) # you get d(tau)/dx

    integral = integration_log(z, integral_eps) # you get tau(E)

    return  1/2.0 * np.pi * r0**2 * integral

def elementary_luminosity(beta_gamma, delta_beta, r_gamma, L_gamma):

    """
    Return the luminosity of a surface of the spherical shock

    Parameters:
        beta_gamma          : colatitude of the gamma-source (rad)
        delta_beta          : range for beta_gamma (rad)
        r_gamma             : distance from the gamma-source to WD (cm)
        L_gamma             : luminosity of the shock (erg/s)
    """

    beta_min = beta_gamma
    beta_max = beta_gamma + delta_beta
    dS = 2*np.pi*r_gamma**2*(np.cos(beta_min) - np.cos(beta_max))
    S = 4*np.pi*r_gamma**2

    return L_gamma * dS/S
