#! /Applications/Anaconda/bin/python

from math import *
import numpy as np
from scipy.interpolate import interp1d as itp

from Conversion_factors import *
from Physical_constants import *

from PlotLib import *

class HydroTableInternalShock:

	kind=''
	obj=''
	arglist=()
	num_dim=0
	num_time=0
	num_theta=0
	num_phi=0
	num_ray=0
	mui=1.3000
	mue=1.1818
	mu=0.65
	gamma=5.0/3.0
	H_frac=0.9
	He_frac=0.1

	# Constructor
	def __init__(self):
			pass

	# Set gas properties
	def set_gas(self,mui,mue,gamma,H_frac,He_frac):
		self.mui=mui
		self.mue=mue
		self.mu=1.0/(1.0/mui+1.0/mue)
		self.gamma=gamma
		self.H_frac=H_frac
		self.He_frac=He_frac

	# Set arrays that will contain variables for each ray
	def set_arrays(self,t_min,t_max,num_time):
		self.v_sh=np.zeros(num_time)        # Velocity of the shock in the binary system frame
		self.v_s=np.zeros(num_time)         # Velocity of the upstream flow in the shock frame
		self.r_s=np.zeros(num_time)         # Radius
		self.m_s=np.zeros(num_time)         # Swept-up mass
		self.m_sh=np.zeros(num_time)        # Total mass in shell
		self.vol_s=np.zeros(num_time)       # Swept-up volume
		self.mach_s=np.zeros(num_time)      # Sonic Mach number
		self.b_s=np.zeros(num_time)         # Upstream magnetic field
		self.j_s=np.zeros(num_time)         # Compression ratio
		self.e_s=np.zeros(num_time)         # Kinetic energy
		self.u_mag_up=np.zeros(num_time)    # Magnetic energy density upstream
		self.u_mag_down=np.zeros(num_time)  # Magnetic energy density downstream
		self.u_rad=np.zeros(num_time)       # Radiation energy density (total)
		self.u_rad_star=np.zeros(num_time)    # Radiation energy density downstream (red giant)
		self.u_rad_nova=np.zeros(num_time)  # Radiation energy density downstream (nova)
		self.rho_up=np.zeros(num_time)      # Density upstream
		self.rho_down=np.zeros(num_time)    # Density downstream immediately behind shock
		self.rho_avg=np.zeros(num_time)     # Density downstream as average of swept-up material
		self.gas_column=np.zeros(num_time)  # Gas column number density above shock (to compute absorption)
		if t_max > t_min:
			self.time=np.linspace(t_min,t_max,num_time)
		else:
			self.time=np.zeros(num_time)

	# Reset arrays
	def clear_arrays(self):
		self.v_sh.fill(0.0)
		self.v_s.fill(0.0)
		self.r_s.fill(0.0)
		self.m_s.fill(0.0)
		self.m_sh.fill(0.0)
		self.vol_s.fill(0.0)
		self.mach_s.fill(0.0)
		self.b_s.fill(0.0)
		self.j_s.fill(0.0)
		self.e_s.fill(0.0)
		self.u_mag_up.fill(0.0)
		self.u_mag_down.fill(0.0)
		self.u_rad.fill(0.0)
		self.u_rad_star.fill(0.0)
		self.u_rad_nova.fill(0.0)
		self.rho_up.fill(0.0)
		self.rho_down.fill(0.0)
		self.rho_avg.fill(0.0)
		self.gas_column.fill(0.0)

	# Make dictionary to store arrays
	def make_dict(self):
		dict={'Time':self.time,'Theta':self.thetarr,'Phi':self.phiarr,'Volume':self.vol_s, \
		'Vframe':self.v_sh,'Vshock':self.v_s,'Rshock':self.r_s,'Mshell':self.m_sh,'Mshock':self.m_s,'Mach':self.mach_s,'Jump':self.j_s,'Bshock':self.b_s,'Eshock':self.e_s, \
		'RHOup':self.rho_up,'RHOdown':self.rho_down,'RHOavg':self.rho_avg,'Umagup':self.u_mag_up,'Umagdown':self.u_mag_down,'Urad':self.u_rad, 'Uradstar':self.u_rad_star,'Uradnova':self.u_rad_nova, \
		'Gascolumn':self.gas_column}
		return dict

	# Set up for forward radiative shock
	def set_internalforwardshock(self,m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size):
		self.kind='InternalForwardShock'
		self.obj='ClassicalNova'
		self.arglist=(m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size)
		self.num_dim=2
		self.num_time=int((t_max-0.0)/t_step)+1
		self.num_theta=num_theta
		self.num_ray=num_theta
		# Theta angle from 0 to pi (direction from WD to star corresponding to theta=0)
		self.thetarr=np.linspace(0.5*pi/num_theta,pi-0.5*pi/num_theta,num_theta)
		self.phiarr=np.empty([])
		self.set_arrays(0.0,t_max,self.num_time)

	# Set up for forward radiative shock
	def set_internalreverseshock(self,m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size):
		self.kind='InternalReverseShock'
		self.obj='ClassicalNova'
		self.arglist=(m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size)
		self.num_dim=2
		self.num_time=int((t_max-0.0)/t_step)+1
		self.num_theta=num_theta
		self.num_ray=num_theta
		# Theta angle from 0 to pi (direction from WD to star corresponding to theta=0)
		self.thetarr=np.linspace(0.5*pi/num_theta,pi-0.5*pi/num_theta,num_theta)
		self.phiarr=np.empty([])
		self.set_arrays(0.0,t_max,self.num_time)

	# Compute nova properties at given time and position
	def nova_light(self,t,r):
		# V407Cyg (constant luminosity)
		if self.obj == 'V407Cyg':
			# Set parameters (obtained by Guillaume from fit to light curve)
			l_nova=2e38/Lsun2erg  # in Lsun
			r_nova=5e12/AU2cm     # Nova evolution starting radius in AU
			t_nova=0.1		 	  # Nova evolution starting time in d
			i_nova=-0.5           # Nova evolution index
			# Compute pseudophotosphere radius in AU
			if t >= t_nova:
 				r_photo=r_nova*(t/t_nova)**i_nova
 			# If position outside of photosphere...
 			if r_photo < r:
 				lum_nova=l_nova
 				temp_nova=(l_nova*Lsun2erg/4.0/pi/r_photo/r_photo/sb)**0.25
 			else:
 				lum_nova=0.0
 				temp_nova=0.0
 		# Typical classical nova
 		elif self.obj == 'ClassicalNova':
			# Set parameters
			lum_nova=1e38/Lsun2erg   # constant luminosity in Lsun
			r_photo=100*Rsun2AU      # constant radius 100Rsun in AU
			temp_nova=1e4            # constant pseudophotosphere temperature in K

 		# Return results
 		return r_photo,lum_nova,temp_nova

	# Compute hydrodynamical variables along given ray
	def get_ray(self,i):
		if self.kind == 'InternalForwardShock':
			dict=self.get_ray_internalforwardshock(i,*self.arglist)
		# Internal reverse radiative shock
		elif self.kind == 'InternalReverseShock':
			dict=self.get_ray_internalreverseshock(i,*self.arglist)
		# Return arrays of variables for given ray
		return dict

	# Compute hydrodynamical variables along given ray
	def get_ray_internalforwardshock(self,iray,m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size):

		# Calculation factors
		k_therm=1.5*Msun2g/(AU2cm*AU2cm*AU2cm)/amu/self.mu*kb*temp_ej/MeV2erg  # To compute thermal energy density in MeV/cm3 from a density in Msun/AU^3
		k_lum=Lsun2erg/4.0/pi/c/(AU2cm*AU2cm)/MeV2erg                          # To compute radiation energy density in MeV/cm3 from luminosity in Lsun and distance in AU
		k_mag=0.5*Msun2g/(AU2cm*AU2cm*AU2cm)*km2cm*km2cm/MeV2erg               # To compute magnetic energy density in MeV/cm3 from kinetic energy density with density in Msun/AU3 and velocity in km/s
		k_part=Msun2g/(AU2cm*AU2cm*AU2cm)/amu/self.mui                         # To compute density in particle/cm3 from density in Msun/AU^3

		# Clear arrays
		self.clear_arrays()

		# Set direction angle and solid angle
  		theta=self.thetarr[iray]
  		omega=2.0*pi*sin(theta)*pi/self.num_theta if self.num_theta > 1 else 4.0*pi

  		# Set sound speed (in km/s)
  		cs_ej=sqrt(self.gamma*kb*temp_ej/self.mu/amu)/km2cm
  		cs_wind=sqrt(self.gamma*kb*temp_wind/self.mu/amu)/km2cm

  		# Convert input quantities in cgs units
  		Ncsm=n_csm
  		pej=n_ej
  		Mej=m_ej  *Msun2g      # Msun to g
		Vej=v_ej  *km2cm	   # km/s to cm/s
		Vin=v_in  *km2cm	   # km/s to cm/s
		Mw=m_wind *Msun2g/yr2s # Msun/yr to g/s
		Vw=v_wind *km2cm	   # km/s to cm/s
		Dw=d_wind *day2s       # day to s
		tw=t_wind *day2s       # day to s

		# Solve shell+shock system evolution equation
		ts,tstart,rs,vs,ve,ms,mes,mws,ne,nw,ke,kw = self.solve_shell_evolution(Mej,Vej,Vin,pej,Mw,Vw,Dw,tw,Ncsm)
		tstart=tstart/day2s

		# Set up interpolation functions for all quantities
		# (after converting from log-log to linear-linear and to nova typical units)
		frs=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,rs/AU2cm)(log10(t)))
		fvs=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,vs/km2cm)(log10(t)))
		fve=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ve/km2cm)(log10(t)))
		fms=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ms/Msun2g)(log10(t)))
		fmes=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,mes/Msun2g)(log10(t)))
		fmws=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,mws/Msun2g)(log10(t)))
		fne=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ne/k_part)(log10(t)))
		fnw=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,nw/k_part)(log10(t)))
		fke=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ke)(log10(t)))
		fkw=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,kw)(log10(t)))

		# Loop over time steps
  		for j in range(self.num_time):

  			# Set current time (in d)
  			t=self.time[j]

  			# If shock not yet triggered, move on
  			if t <= tstart:
  				continue

  			# Radius (in AU)
  			self.r_s[j]=frs(t)
  			# Volume (in AU3)
  			self.vol_s[j]=omega/3.0*((self.r_s[j])**3.0-(self.r_s[j-1])**3.0)
  			# Velocity (in km/s)
  			self.v_sh[j]=fvs(t)
  			self.v_s[j]=fvs(t)-fve(t)
  			# Shock Mach number and compression ratio
  			self.mach_s[j]=self.v_s[j]/cs_ej
  			self.j_s[j]=((self.gamma+1)*self.mach_s[j]*self.mach_s[j])/(2+(self.gamma-1)*self.mach_s[j]*self.mach_s[j])
  			# Swept-up and total mass (in Msun)
  			self.m_s[j]=fmes(t)
  			self.m_sh[j]=fms(t)
  			# Upstream and downstream densities (in Msun/AU3)
  			self.rho_up[j]=fne(t)
  			self.rho_down[j]=self.rho_up[j]*self.j_s[j]
  			# Average density of swept-up material for a shell with user-defined fractional thickness
  			if cool_size > 0.0 and cool_size < 1.0:
  				self.rho_avg[j]=self.m_s[j]/(omega/3.0*((self.r_s[j])**3.0-((1.0*cool_size)*self.r_s[j])**3.0))
  			# Average density in radiatively-cooled shell assuming maximum compression by mach number squared at both shocks
  			elif cool_size < 0.0:
  				mach_f=(fvs(t)-fve(t))/cs_ej
  				mach_r=(v_wind-fvs(t))/cs_wind
  				rho_fd=fne(t)*mach_f*mach_f
  				rho_rd=fnw(t)*mach_r*mach_r
  				self.rho_avg[j]=(rho_fd-rho_rd)/log(rho_fd/rho_rd)         # in Msun/AU3
  				drs=fms(t)/4.0/pi/self.r_s[j]/self.r_s[j]/self.rho_avg[j]  # in AU
  				# Debug
  				#print 'At time=%.2f shell radius=%.2e thickness=%.2e density=%.2e versus upstream densities forward=%.2e reverse=%.2e' % (t,self.r_s[j],drs,k_rho*self.rho_avg[j],rho_i,rho_w)
			# Average density by default is downstream one
			else:
				self.rho_avg[j]=self.rho_down[j]
			# Kinetic energy
  			self.e_s[j]=fke(t)
  			# Magnetic field energy density upstream
  			if mag_frac <= 0.0 or mag_frac > 1.0:	     # ...from equipartition with wind thermal energy (in muG)
  				self.u_mag_up[j]=k_therm*self.rho_up[j]
  			else:                                        # ...or from inflowing kinetic energy
  				self.u_mag_up[j]=mag_frac*k_mag*0.5*self.rho_up[j]*(self.v_s[j])**2.0             # WARNING ! The 0.5 factor is superfluous, already included in k_mag !
  			# Magnetic field strength
  			self.b_s[j]=sqrt(8.0*pi*self.u_mag_up[j]*MeV2erg)*1.0e6
  			# Magnetic field energy density downstream including some compression
  			self.u_mag_down[j]=self.u_mag_up[j]*self.j_s[j]*self.j_s[j]
  			# Compute radiation energy density is contributed by star and nova (in MeV/cm3)
  			r_photo,lum_nova,temp_nova=self.nova_light(t,self.r_s[j])
  			self.u_rad_star[j]=k_lum*l_star/(self.r_s[j]*self.r_s[j])    # Star
 			self.u_rad_nova[j]=k_lum*lum_nova/(self.r_s[j]*self.r_s[j])  # Nova
			self.u_rad[j]=self.u_rad_star[j]+self.u_rad_nova[j]          # Total
  			# Compute opacity due to pair production in field of nuclei
  			# Approximation: compute average density in unshocked ejecta and multiply by distance to ejecta outer surface
  			r_ej=v_ej*t*day2s/AU2km
  			if self.r_s[j] < r_ej:
  				upmass=m_ej*omega/4.0/pi-self.m_s[j]            # Unshocked ejecta mass in Msun
  				upvol=omega/3.0*(r_ej**3.0-(self.r_s[j])**3.0)  # Unshocked ejecta volume in AU^3.0
  				uplen=r_ej-self.r_s[j]                          # Distance between shock and outer ejecta bpoundary in AU
  				self.gas_column[j]=k_part*upmass/upvol*uplen*AU2cm  # in ion/cm3

 		# Convert upstream density from Msun/AU3 into nuclei/cm3
 		# Immediate downstream density is upstream one multiplied by compression ratio
 		# Average downstream density is computed if required otherwise immediate downstream density is used
 		self.rho_up=k_part*self.rho_up
 		self.rho_down=k_part*self.rho_down
 		self.rho_avg=k_part*self.rho_avg

 		# Make dictionary of arrays
 		dict=self.make_dict()

 		# Return dictionary
		return dict

	# Compute hydrodynamical variables along given ray
	def get_ray_internalreverseshock(self,iray,m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size):

		# Calculation factors
		k_therm=1.5*Msun2g/(AU2cm*AU2cm*AU2cm)/amu/self.mu*kb*temp_ej/MeV2erg  # To compute thermal energy density in MeV/cm3 from a density in Msun/AU^3
		k_lum=Lsun2erg/4.0/pi/c/(AU2cm*AU2cm)/MeV2erg                          # To compute radiation energy density in MeV/cm3 from luminosity in Lsun and distance in AU
		k_mag=0.5*Msun2g/(AU2cm*AU2cm*AU2cm)*km2cm*km2cm/MeV2erg               # To compute magnetic energy density in MeV/cm3 from kinetic energy density with density in Msun/AU3 and velocity in km/s
		k_part=Msun2g/(AU2cm*AU2cm*AU2cm)/amu/self.mui                         # To compute density in particle/cm3 from density in Msun/AU^3

		# Clear arrays
		self.clear_arrays()

		# Set direction angle and solid angle
  		theta=self.thetarr[iray]
  		omega=2.0*pi*sin(theta)*pi/self.num_theta if self.num_theta > 1 else 4.0*pi

  		# Set sound speed (in km/s)
  		cs_ej=sqrt(self.gamma*kb*temp_ej/self.mu/amu)/km2cm
  		cs_wind=sqrt(self.gamma*kb*temp_wind/self.mu/amu)/km2cm

  		# Convert input quantities in cgs units
  		Ncsm=n_csm
  		pej=n_ej
  		Mej=m_ej  *Msun2g      # Msun to g
		Vej=v_ej  *km2cm	   # km/s to cm/s
		Vin=v_in  *km2cm	   # km/s to cm/s
		Mw=m_wind *Msun2g/yr2s # Msun/yr to g/s
		Vw=v_wind *km2cm	   # km/s to cm/s
		Dw=d_wind *day2s       # day to s
		tw=t_wind *day2s       # day to s

		# Solve shell+shock system evolution equation
		ts,tstart,rs,vs,ve,ms,mes,mws,ne,nw,ke,kw = self.solve_shell_evolution(Mej,Vej,Vin,pej,Mw,Vw,Dw,tw,Ncsm)
		tstart=tstart/day2s

		# Set up interpolation functions for all quantities
		# (after converting from log-log to linear-linear and to nova typical units)
		frs=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,rs/AU2cm)(log10(t)))
		fvs=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,vs/km2cm)(log10(t)))
		fve=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ve/km2cm)(log10(t)))
		fms=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ms/Msun2g)(log10(t)))
		fmes=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,mes/Msun2g)(log10(t)))
		fmws=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,mws/Msun2g)(log10(t)))
		fne=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ne/k_part)(log10(t)))
		fnw=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,nw/k_part)(log10(t)))
		fke=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,ke)(log10(t)))
		fkw=lambda t : 10.0**(self.loglog_interpolation(ts/day2s,kw)(log10(t)))

		# Loop over time steps
  		for j in range(self.num_time):

  			# Set current time (in d)
  			t=self.time[j]

  			# If shock not yet triggered, move on
  			if t <= tstart:
  				continue

  			# Radius (in AU)
  			self.r_s[j]=frs(t)
  			# Volume (in AU3)
  			self.vol_s[j]=omega/3.0*((self.r_s[j])**3.0-(self.r_s[j-1])**3.0)
  			# Velocity (in km/s)
  			self.v_sh[j]=fvs(t)
  			self.v_s[j]=v_wind-fvs(t)
  			# Shock Mach number and compression ratio
  			self.mach_s[j]=self.v_s[j]/cs_ej
  			self.j_s[j]=((self.gamma+1)*self.mach_s[j]*self.mach_s[j])/(2+(self.gamma-1)*self.mach_s[j]*self.mach_s[j])
  			# Swept-up and total mass (in Msun)
  			self.m_s[j]=fmws(t)
  			self.m_sh[j]=fms(t)
  			# Upstream and downstream densities (in Msun/AU3)
  			self.rho_up[j]=fnw(t)
  			self.rho_down[j]=self.rho_up[j]*self.j_s[j]
  			# Average density of swept-up material for a shell with user-defined fractional thickness
  			if cool_size > 0.0 and cool_size < 1.0:
  				self.rho_avg[j]=self.m_s[j]/(omega*self.r_s[j]*self.r_s[j]*self.r_s[j]*cool_size)
  			# Average density in radiatively-cooled shell assuming maximum compression by mach number squared at both shocks
  			elif cool_size < 0.0:
  				mach_f=(fvs(t)-fve(t))/cs_ej
  				mach_r=(v_wind-fvs(t))/cs_wind
  				rho_fd=fne(t)*mach_f*mach_f
  				rho_rd=fnw(t)*mach_r*mach_r
  				self.rho_avg[j]=(rho_fd-rho_rd)/log(rho_fd/rho_rd)         # in Msun/AU3
  				drs=fms(t)/4.0/pi/self.r_s[j]/self.r_s[j]/self.rho_avg[j]  # in AU
  				# Debug
  				#print 'At time=%.2f shell radius=%.2e thickness=%.2e density=%.2e versus upstream densities forward=%.2e reverse=%.2e' % (t,self.r_s[j],drs,k_rho*self.rho_avg[j],rho_i,rho_w)
			# Average density by default is downstream one
			else:
				self.rho_avg[j]=self.rho_down[j]
			# Kinetic energy
  			self.e_s[j]=fkw(t)
  			# Magnetic field energy density upstream
  			if mag_frac <= 0.0 or mag_frac > 1.0:	     # ...from equipartition with wind thermal energy (in muG)
  				self.u_mag_up[j]=k_therm*self.rho_up[j]
  			else:                                        # ...or from inflowing kinetic energy
  				self.u_mag_up[j]=mag_frac*k_mag*0.5*self.rho_up[j]*(self.v_s[j])**2.0            # WARNING ! The 0.5 factor is superfluous, already included in k_mag !
  			# Magnetic field strength
  			self.b_s[j]=sqrt(8.0*pi*self.u_mag_up[j]*MeV2erg)*1.0e6
  			# Magnetic field energy density downstream including some compression
  			self.u_mag_down[j]=self.u_mag_up[j]*self.j_s[j]*self.j_s[j]
  			# Compute radiation energy density is contributed by star and nova (in MeV/cm3)
  			r_photo,lum_nova,temp_nova=self.nova_light(t,self.r_s[j])
  			self.u_rad_star[j]=k_lum*l_star/(self.r_s[j]*self.r_s[j])        # Star
 			self.u_rad_nova[j]=k_lum*lum_nova/(self.r_s[j]*self.r_s[j])  # Nova
			self.u_rad[j]=self.u_rad_star[j]+self.u_rad_nova[j]            # Total
  			# Compute opacity due to pair production in field of nuclei
  			# Approximation: compute average density in unshocked ejecta and multiply by distance to ejecta outer surface
  			r_ej=v_ej*t*day2s/AU2km
  			if self.r_s[j] < r_ej:
  				upmass=m_ej*omega/4.0/pi                        # Entire ejecta mass in Msun
  				upvol=omega/3.0*(r_ej**3.0-(self.r_s[j])**3.0)  # Unshocked ejecta volume in AU^3.0
  				uplen=r_ej-self.r_s[j]                          # Distance between shock and outer ejecta bpoundary in AU
  				self.gas_column[j]=k_part*upmass/upvol*uplen*AU2cm  # in ion/cm3

 		# Convert upstream density from Msun/AU3 into nuclei/cm3
 		# Immediate downstream density is upstream one multiplied by compression ratio
 		# Average downstream density is computed if required otherwise immediate downstream density is used
 		self.rho_up=k_part*self.rho_up
 		self.rho_down=k_part*self.rho_down
 		self.rho_avg=k_part*self.rho_avg

 		# Make dictionary of arrays
 		dict=self.make_dict()

 		# Return dictionary
		return dict

	# Solve shell+shock system evolution equation
	def solve_shell_evolution(self,Mej,Vej,Vin,pej,Mw,Vw,Dw,tw,Ncsm):

		# Time grid
		tmin=0.1 *day2s
		tmax=1e3 *day2s
		ntperdec=100
		nt=int(ntperdec*log10(tmax/tmin))
		time=np.logspace(log10(tmin),log10(tmax),nt)

		# Initialize variables
		vs=0
		rs=0
		ms=0
		ne=0
		nw=0
		rej=0
		rin=0
		rw=0.0

		# Useful stuff
		fourpi=4.0*pi
		meanmass=self.mui*amu

		# Define functions for terms in equation
		# Wind mass loss rate
		def Mw_point(t,rs):
			return Mw*np.exp(-(t-tw-rs/Vw)/Dw)
		# Mass flux from wind into shell
		def Mws_point(t,vs,rs):
			return max( Mw_point(t,rs)*(1-vs/Vw) , 0.0 )
		# Mass flux from ejecta into shell
		def Mes_point(t,vs,rs):
			return fourpi*rs*rs*ne(t,rs)*meanmass*(vs-rs/t) if rs < rej else fourpi*rs*rs*ne(t,rs)*meanmass*vs
		# Mass flux into shell
		def Ms_point(t,vs,rs):
			return Mws_point(t, vs, rs) + Mes_point(t,vs,rs)
		# Wind swept-up mass
		def Mws_sum(t,rs):
			return Dw*Mw*(1.0-exp(-(t-tw-rs/Vw)/Dw))
		# Ejecta swept-up mass
		def Mes_sum(t,rs):
			m=3.0-pej
			return Mej*(rs**m-rin**m)/(rej**m-rin**m) if rs < rej else Mej
		# Wind density
		def nw(t,rs):
			return Mw_point(t,rs)/meanmass/fourpi/rs/rs/Vw
		# Ejecta density
		def ne(t,rs):
			m=3.0-pej
			return Mej*m/meanmass/fourpi/(rej**m-rin**m)/rs**pej if rs < rej else Ncsm
		# Wind swept-up momentum
		def Pws_sum(t,rs):
			return Dw*Mw*Vw*(1.0-exp(-(t-tw-rs/Vw)/Dw))
		# Ejecta swept-up momentum
		def Pes_sum(t,rs):
			m=3.0-pej
			l=4.0-pej
			return Mej*m/l*(rs**l-rin**l)/(rej**m-rin**m)/t	if rs < rej else Mej*m/l*(rej**l-rin**l)/(rej**m-rin**m)/t
		# Wind swept-up kinetic energy
		def Kws_point(t,vs,rs):
			return 0.5*fourpi*rs*rs*nw(t,rs)*meanmass*(Vw-vs)**3.0
		# Ejecta swept-up kinetic energy
		def Kes_point(t,vs,rs):
			return 0.5*fourpi*rs*rs*ne(t,rs)*meanmass*(vs-rs/t)**3.0
		# Right-hand side of momentum conservation equation
		def dvsdt(t,rs,vs,ms):
			return (Mws_point(t,vs,rs)*(Vw-vs) + Mes_point(t,vs,rs)*(rs/t-vs))/ms if ms > 0.0 else 0.0

		# Save data to arrays
		Rc=np.zeros(nt)
		Vc=np.zeros(nt)
		Vu=np.zeros(nt)
		Mc=np.zeros(nt)
		Mce=np.zeros(nt)
		Mcw=np.zeros(nt)
		Rhoe=np.zeros(nt)
		Rhow=np.zeros(nt)
		Ke=np.zeros(nt)
		Kw=np.zeros(nt)

		# Loop over time steps and solve equation via RK4
		tc=0.0
		Collision=False
		for i in range(nt) :

			# Time
			t=time[i]
			h= t-time[i-1] if i > 0 else 0.0
			# Update ejecta and free wind boundaries
			rej=Vej*t
			rin=Vin*t
			rw=Vw*(t-tw) if t > tw else 0.0

			# If wind has not yet reached the ejecta, move on...
			if rw < rin:
				continue
			else:
				# Collision takes place !
				if not Collision:
					rs=rin
					vs=(Vin+sqrt(nw(t,rs)/ne(t,rs))*Vw)/(1.0+sqrt(nw(t,rs)/ne(t,rs))) # dMsVs/dt=Vs*Ms_point when shock starts and has a velocity but no mass
					ms=0.0
					Collision=True
					tc=t
				# Shock propagates
				else:
					# Solve for velocity and position over this time step
					# (with special case k_1=0 for first time step)
					k_1 = dvsdt( time[i-1], Rc[i-1], Vc[i-1], Mc[i-1] )
					k_2 = dvsdt( time[i-1] + h/2, Rc[i-1] + (h/2)*Vc[i-1], Vc[i-1] + (h/2)*k_1, Mc[i-1] + (h/2)*Ms_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_1, Rc[i-1] + (h/2)*Vc[i-1]) )
					k_3 = dvsdt( time[i-1] + h/2, Rc[i-1] + (h/2)*Vc[i-1] + ((h**2)/4)*k_1, Vc[i-1] + (h/2)*k_2, Mc[i-1] + (h/2)*Ms_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_2, Rc[i-1] + (h/2)*Vc[i-1] + ((h**2)/4)*k_1) )
					k_4 = dvsdt( time[i-1] + h, Rc[i-1] + h*Vc[i-1] + ((h**2)/2)*k_2, Vc[i-1] + h*k_3, Mc[i-1] + h*Ms_point(time[i-1] + h, Vc[i-1] + h*k_3, Rc[i-1] + h*Vc[i-1] + ((h**2)/2)*k_2) )
					rs = Rc[i-1] + h*Vc[i-1] + ((h**2)/6)*(k_1 + k_2 + k_3) if k_1 != 0.0 else Rc[i-1] + h*Vc[i-1] + ((h**2)/4)*(k_2 + k_3)
					vs = Vc[i-1] + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4) if k_1 != 0.0 else Vc[i-1] + (h/5)*(2*k_2 + 2*k_3 + k_4)
					# Solve for mass over this time step
					m_1 = Ms_point(time[i-1], Vc[i-1], Rc[i-1])
					m_2 = Ms_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_1, Rc[i-1] + (h/2)*Vc[i-1])
					m_3 = Ms_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_2, Rc[i-1] + (h/2)*Vc[i-1] + ((h**2)/4)*k_1)
					m_4 = Ms_point(time[i-1] + h, Vc[i-1] + h*k_3, Rc[i-1] + h*Vc[i-1] + ((h**2)/2)*k_2)
					ms = Mc[i-1] + (h/6)*(m_1 + 2*m_2 + 2*m_3 + m_4)
					Mce[i]=Mes_sum(t,rs)
					Mcw[i]=Mws_sum(t,rs)
					# Solve for swept-up kinetic energy over this time step
					e_1 = Kes_point(time[i-1], Vc[i-1], Rc[i-1])
					e_2 = Kes_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_1, Rc[i-1] + (h/2)*Vc[i-1])
					e_3 = Kes_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_2, Rc[i-1] + (h/2)*Vc[i-1] + ((h**2)/4)*k_1)
					e_4 = Kes_point(time[i-1] + h, Vc[i-1] + h*k_3, Rc[i-1] + h*Vc[i-1] + ((h**2)/2)*k_2)
					Ke[i] = Ke[i-1] + (h/6)*(e_1 + 2*e_2 + 2*e_3 + e_4)
					e_1 = Kws_point(time[i-1], Vc[i-1], Rc[i-1])
					e_2 = Kws_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_1, Rc[i-1] + (h/2)*Vc[i-1])
					e_3 = Kws_point(time[i-1] + h/2, Vc[i-1] + (h/2)*k_2, Rc[i-1] + (h/2)*Vc[i-1] + ((h**2)/4)*k_1)
					e_4 = Kws_point(time[i-1] + h, Vc[i-1] + h*k_3, Rc[i-1] + h*Vc[i-1] + ((h**2)/2)*k_2)
					Kw[i] = Kw[i-1] + (h/6)*(e_1 + 2*e_2 + 2*e_3 + e_4)

			# Save data to arrays
			Rc[i]=rs
			Vc[i]=vs
			Vu[i]=rs/t if rs < rej else Vej
			Mc[i]=ms
			Rhoe[i]=ne(t,rs)
			Rhow[i]=nw(t,rs)

		# Return
		return time,tc,Rc,Vc,Vu,Mc,Mce,Mcw,Rhoe,Rhow,Ke,Kw

	# Interpolation of tabulated data in log-log space
	def loglog_interpolation(self,x,y):

		# Find non-zero values
		nzidx=y > 0.0
		# Return interpolation function
		return itp(np.log10(x[nzidx]),np.log10(y[nzidx]),kind='linear',bounds_error=False,fill_value=0.0)

	# Savitsky-Golay smoothing/filtering
	def savitzky_golay(self,y, window_size, order, deriv=0, rate=1):
		"""
		Gotten from http://scipy-cookbook.readthedocs.org/items/SavitzkyGolay.html
		"""

		# Start
		try:
		    window_size = np.abs(np.int(window_size))
		    order = np.abs(np.int(order))
		except ValueError, msg:
			raise ValueError("window_size and order have to be of type int")

	    # Check window size is odd number and wide enough given polynomila order
		if window_size % 2 != 1 or window_size < 1:
			raise TypeError("window_size size must be a positive odd number")
		if window_size < order + 2:
			raise TypeError("window_size is too small for the polynomials order")
		order_range = range(order+1)
		half_window = (window_size -1) // 2

	    # Precompute coefficients
		b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
		m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

	    # Pad the signal at the extremes with values taken from the signal itself
		firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
		lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
		y = np.concatenate((firstvals, y, lastvals))

	    # Return
		return np.convolve( m[::-1], y, mode='valid')

	# Plot shock properties
	def plot_shock(self,iray,m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size,shock_type='Forward'):

		# Example: hydro.plot_shock(0,1e-4,1e3,1e2,2.0,1e4,5e-4,2e3,1e1,2.0,1e4,10.0,1.0,1e2,1.0,1,0.01,0.0,'Forward')

		# Define gas properties
		self.set_gas(1.3000,1.1818,1.6666,0.9,0.1)
		# Set class
		if shock_type == 'Forward':
			self.set_internalforwardshock(m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size)
		elif shock_type == 'Reverse':
			self.set_internalreverseshock(m_ej,v_ej,v_in,n_ej,temp_ej,m_wind,v_wind,d_wind,t_wind,temp_wind,n_csm,l_star,t_max,t_step,num_theta,mag_frac,cool_size)
		# Get first ray
		h=self.get_ray(iray)
		# Plot
		set_graphics(False,'jpg')
		plot_simple(h['Time'],h['Rshock'],xlabel='Time (days)',ylabel='Shock radius (AU)',xlog=False,ylog=True,ybds=[1e-1,1e3],color='blue')
		plot_simple(h['Time'],h['Vframe'],xlabel='Time (days)',ylabel='Shock speed (km/s)',xlog=False,ylog=True,ybds=[1e1,1e4],color='blue')
		plot_simple(h['Time'],h['Vshock'],xlabel='Time (days)',ylabel='Upstream flow velocity (km/s)',xlog=False,ylog=True,ybds=[1e1,1e4],color='blue')
		plot_simple(h['Time'],h['RHOup'],xlabel='Time (days)',ylabel='Upstream density (p/cm3)',xlog=False,ylog=True,ybds=[1e5,1e15],color='blue')
		plot_simple(h['Time'],h['Mshock'],xlabel='Time (days)',ylabel='Swept-up mass (Msun)',xlog=False,ylog=True,ybds=[1e-7,1e-3],color='blue')
		plot_simple(h['Time'],h['Bshock']/G2muG,xlabel='Time (days)',ylabel='Upstream magnetic field (G)',xlog=False,ylog=True,ybds=[1e-3,1e3],color='blue')
		plt.show()
