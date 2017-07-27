#! /Applications/Anaconda/bin/python

from PartDistrib import PartDistrib

from copy import deepcopy

from math import pi
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as ptch

import scipy.optimize as opt
import scipy.interpolate as interp

from Conversion_factors import *
from Physical_constants import *

class RadHistory:

	name=''
	gam=np.array([])
	ngam=0
	ntime=0
	nray=0
	spec_all=[]
	spec_ray=[]
	spec_tot=[]
	spec_avg=np.array([])
	lc_nrj=np.array([])
	lc_ph=np.array([])
	nrj_all=[]
	nrj_tot=np.array([])

	# Constructor
	def __init__(self,gam,time,nray,name):
		self.name=name
		self.gam=gam
		self.time=time
		self.ngam=gam.size
		self.ntime=time.size
		self.nray=nray
		self.spec_all=[[np.zeros(self.ngam)]*self.ntime]*nray
		self.spec_ray=[np.zeros(self.ngam)]*self.ntime
		self.spec_tot=[np.zeros(self.ngam)]*self.ntime
		self.spec_avg=np.zeros(self.ngam)
		self.spec_flux=np.zeros(self.ngam)
		self.lc_nrj=np.zeros(self.ntime)
		self.lc_ph=np.zeros(self.ntime)
		self.lc_flux=np.zeros(self.ntime)
		self.nrj_all=[np.zeros(self.ntime)]*nray
		self.nrj_tot=np.zeros(self.ntime)

	# Save spectrum at given time for given ray
	def save_time(self,i,s):
		if i < self.ntime:
			self.spec_ray[i]= deepcopy(s)

	# Save spectrum evolution along given ray
	def save_ray(self,i):
		self.spec_all[i]=deepcopy(self.spec_ray)
		self.spec_ray=[np.zeros(self.ngam)]*self.ntime

	# Update total over rays at given time
	def save_total(self,i,s):
		if i < self.ntime:
			self.spec_tot[i]= np.add(self.spec_tot[i],s)

	# Fill from other class instance
	def add_from(self, radhist):
		# Check that class is compatible
		if radhist.ngam != self.ngam or radhist.ntime != self.ntime or radhist.nray != self.nray or len(radhist.spec_all) != self.nray:
			print 'Cannot add data from class with inconsistent definition.'
			return
		# Add spectra for all rays and times
		for i in range(self.nray):
			for j in range(self.ntime):
				self.spec_all[i][j]= np.add(self.spec_all[i][j],radhist.spec_all[i][j])
		# Add total spectra
		for i in range(self.ntime):
			self.spec_tot[i]=np.add(self.spec_tot[i],radhist.spec_tot[i])

	# Compute light curve in given band or at given energy/frequency
	def compute_lightcurve(self,gmin,gmax):
		# Check energy bounds
		if gmin < self.gam.min() or gmax > self.gam.max() or gmin > gmax:
			print 'Cannot compute light curve with out of range or inconsistent energy bounds.'
			return
		# Flux over a band
		elif gmin != gmax:
			# Find indices for bounds
			i_gmin=np.searchsorted(self.gam,gmin)
			i_gmax=np.searchsorted(self.gam,gmax)
			# Integrate photon and energy flux
			phinerg=self.gam[i_gmin:i_gmax+1]*1e6*eV2erg
			for i in range(self.ntime):
				self.lc_nrj[i]= self.log_integral(self.gam[i_gmin:i_gmax+1],self.spec_tot[i][i_gmin:i_gmax+1])
				self.lc_ph[i]= self.log_integral(self.gam[i_gmin:i_gmax+1],self.spec_tot[i][i_gmin:i_gmax+1]/phinerg)
			# Apply MeV to eV conversion (because spectral luminosity is in erg/s/eV)
			self.lc_nrj=self.lc_nrj*1e6
			self.lc_ph=self.lc_ph*1e6
		# Flux at a given energy or frequency
		else:
			i_g=np.searchsorted(self.gam,gmin)
			for i in range(self.ntime):
				self.lc_nrj[i]= self.spec_tot[i][i_g]
				self.lc_ph[i]= self.spec_tot[i][i_g]/(self.gam[i_g]*1e6*eV2erg)

	# Compute average spectrum over given time interval
	def compute_average_spec(self,tmin,tmax):
		# Check time bounds
		if tmin < self.time.min() or tmax > self.time.max() or tmin > tmax:
			print 'Cannot compute average spectrum with out of range or inconsistent time bounds.'
			return
		# Find indices for bounds
		else:
			i_tmin=np.searchsorted(self.time,tmin)
			i_tmax=np.searchsorted(self.time,tmax)
		# Compute average (assumes constant time bins)
		self.spec_avg=sum(self.spec_tot[i_tmin:i_tmax+1])/(i_tmax-i_tmin+1)

	# Compute emitted energy per ray and total
	def compute_energy(self):
		# Reset arrays
		self.nrj_all=[np.zeros(self.ntime)]*self.nray
		self.nrj_tot=np.zeros(self.ntime)
		# Loop over rays
		for i in range(self.nray):
			# Loop over times
			for j in range(self.ntime):
				if j == 0:
					dt=self.time[0]
					self.nrj_all[i][j]=self.log_integral(self.gam,self.spec_all[i][j])*dt*day2s*1e6
				else:
					dt=self.time[j]-self.time[j-1]
					self.nrj_all[i][j]=self.nrj_all[i][j-1]+self.log_integral(self.gam,self.spec_all[i][j])*dt*day2s*1e6
		for j in range(self.ntime):
			if j == 0:
				dt=self.time[0]
				self.nrj_tot[j]=self.log_integral(self.gam,self.spec_tot[j])*dt*day2s*1e6
			else:
				dt=self.time[j]-self.time[j-1]
				self.nrj_tot[j]=self.nrj_tot[j-1]+self.log_integral(self.gam,self.spec_tot[j])*dt*day2s*1e6

	# Plot fitted spectrum and lightcurve
	def plot_fit(self,d):
		# Fermi-LAT data
		# (format: time or energy / flux / flux+err or 0 if upper limit)
		# (time in days, energy in MeV, fluxes in 10e-7 ph/cm2/s for lightcurve and in erg/cm2/s for spectrum)
		# Light curve
		num_lat_lc=42
		lc_lat=np.zeros((num_lat_lc,3))
		lc_lat[0]=[-3.50000,2.45854,0.0]
		lc_lat[1]=[-2.50000,2.16585,0.0]
		lc_lat[2]=[-1.50000,3.57073,0.0]
		lc_lat[3]=[-5.00000e-1,1.46341,0.0]
		lc_lat[4]=[5.00000e-1,4.27317,5.85366]
		lc_lat[5]=[1.50000,7.20000,8.95610]
		lc_lat[6]=[2.50000,3.68780,5.20976]
		lc_lat[7]=[3.50000,9.07317,1.10049e+1]
		lc_lat[8]=[4.50000,8.72195,1.07707e+1]
		lc_lat[9]=[5.50000,7.31707,9.36585]
		lc_lat[10]=[6.50000,3.27805,4.68293]
		lc_lat[11]=[7.50000,3.74634,5.50244]
		lc_lat[12]=[8.50000,2.69268,3.98049]
		lc_lat[13]=[9.50000,7.60976,9.54146]
		lc_lat[14]=[1.05000e+1,2.57561,3.74634]
		lc_lat[15]=[1.15000e+1,3.62927,5.32683]
		lc_lat[16]=[1.25000e+1,2.75122,0.0]
		lc_lat[17]=[1.35000e+1,6.84878,0.0]
		lc_lat[18]=[1.45000e+1,4.91707,6.73171]
		lc_lat[19]=[1.55000e+1,4.85854,6.75000]
		lc_lat[20]=[1.65000e+1,3.51220,0.0]
		lc_lat[21]=[1.75000e+1,5.67805,0.0]
		lc_lat[22]=[1.85000e+1,1.75610,0.0]
		lc_lat[23]=[1.95000e+1,3.45366,0.0]
		lc_lat[24]=[2.05000e+1,3.21951,0.0]
		lc_lat[25]=[2.15000e+1,3.74634,0.0]
		lc_lat[26]=[2.25000e+1,4.21463,0.0]
		lc_lat[27]=[2.35000e+1,3.10244,0.0]
		lc_lat[28]=[2.45000e+1,1.99024,0.0]
		lc_lat[29]=[2.55000e+1,4.33171,0.0]
		lc_lat[30]=[2.65000e+1,3.27805,0.0]
		lc_lat[31]=[2.75000e+1,3.16098,0.0]
		lc_lat[32]=[2.85000e+1,1.75610,0.0]
		lc_lat[33]=[2.95000e+1,2.75122,0.0]
		lc_lat[34]=[3.05000e+1,3.16098,0.0]
		lc_lat[35]=[3.15000e+1,1.63902,0.0]
		lc_lat[36]=[3.25000e+1,2.16585,0.0]
		lc_lat[37]=[3.35000e+1,4.68293,0.0]
		lc_lat[38]=[3.45000e+1,2.16585,0.0]
		lc_lat[39]=[3.55000e+1,5.56098,0.0]
		lc_lat[40]=[3.65000e+1,1.87317,0.0]
		lc_lat[41]=[3.75000e+1,3.45366,0.0]
		lc_lat[:,1]=lc_lat[:,1]*1.0e-7
		lc_lat[:,2]=lc_lat[:,2]*1.0e-7
		# Average spectrum
		num_lat_spec=10
		spec_lat=np.zeros((num_lat_spec,3))
		spec_lat[0]=[1.29953e+2,1.14686e-10,0.0]
		spec_lat[1]=[2.23764e+2,5.59408e-11,7.80982e-11]
		spec_lat[2]=[3.81574e+2,8.90692e-11,1.06849e-10]
		spec_lat[3]=[6.50680e+2,1.09032e-10,1.26889e-10]
		spec_lat[4]=[1.10957e+3,9.27455e-11,1.10140e-10]
		spec_lat[5]=[1.89210e+3,8.64080e-11,1.04710e-10]
		spec_lat[6]=[3.22652e+3,5.70836e-11,7.80982e-11]
		spec_lat[7]=[5.55568e+3,6.57642e-11,0.0]
		spec_lat[8]=[9.38235e+3,3.92676e-11,0.0]
		spec_lat[9]=[1.58448e+4,6.37992e-11,0.0]
		# Compute model fluxes from distance input in kpc (convert erg/s/eV into erg/s/cm2 for spectrum)
		d_coef=1.0/(4.0*pi*d*d*kpc2cm*kpc2cm)
		self.spec_flux=d_coef*(self.spec_avg*self.gam*1e6)
		self.lc_flux=d_coef*self.lc_ph
		# Prepare non-upper-limit spectral data
		xdata=[]
		ydata=[]
		yerr=[]
		for set in spec_lat:
			if set[2] != 0.0:
				xdata.append(set[0])
				ydata.append(set[1])
				yerr.append(set[2]-set[1])
		xdata=np.asarray(xdata)
		ydata=np.asarray(ydata)
		yerr=np.asarray(yerr)
		# Perform chi-square fit
		fit_pars,fit_covar=opt.curve_fit(self.func_spec_mod, xdata, ydata, p0=[ydata.max()/self.spec_flux.max()], sigma=yerr)
		fit_renorm=fit_pars[0]
		# Filename container
		filename=[]
		# Plot fitted spectrum together with Fermi-LAT points
		x=self.gam
		y=fit_renorm*self.spec_flux
		specfig=self.plot_simple(x,y,xlabel='Photon energy (MeV)',ylabel=r'E $\times$ L(E)'+' (erg/s)',text='Renormalised by %.3e' % (fit_renorm))
		filename.append('fitted_spectrum')
		plt.figure(specfig.number)
		plt.errorbar(xdata,ydata,yerr=[yerr,yerr],fmt='.',color='black')
		# Prepare non-upper-limit time data
		xdata=[]
		ydata=[]
		yerr=[]
		for set in lc_lat:
			if set[2] != 0.0:
				xdata.append(set[0])
				ydata.append(set[1])
				yerr.append(set[2]-set[1])
		xdata=np.asarray(xdata)
		ydata=np.asarray(ydata)
		yerr=np.asarray(yerr)
		# Plot fitted lightcurve together with Fermi-LAT points
		x=self.time
		y=fit_renorm*self.lc_flux
		lcfig=self.plot_simple(x,y,xlabel='Time (days)',ylabel='Flux (ph/s/cm'+r'$^2$'+')',text='Renormalised by %.3e' % (fit_renorm),xlog=False,xbds=[0.0,x.max()])
		filename.append('fitted_lightcurve')
		plt.figure(lcfig.number)
		plt.errorbar(xdata,ydata,yerr=[yerr,yerr],fmt='.',color='black')
		# Return plot name
		return filename

	# Plot emitted energy buildup
	def plot_energy(self,text):
		# Filename container
		filename=[]
		# Plot
		x=self.time
		y=self.nrj_tot
		self.plot_simple(x,y,xlabel='Time (days)',ylabel='Emitted energy (erg)',text=text,xlog=False)
		filename.append('%s_emitted_energy' % (text.replace(' ','_').replace('(','').replace(')','')))
		# Return plot name
		return filename

	# Plot average spectrum
	def plot_average_spec(self,text):
		# Filename container
		filename=[]
		# Plot
		x=self.gam
		y=self.spec_avg
		self.plot_simple(x,x*1e6*y,xlabel='Photon energy (MeV)',ylabel=r'E $\times$ L(E)'+' (erg/s)',text=text)
		filename.append('%s_average_spectrum' % (text.replace(' ','_').replace('(','').replace(')','')))
		# Return plot name
		return filename

	# Plot lightcurve
	def plot_lightcurve(self,text):
		# Filename container
		filename=[]
		# Plot
		x=self.time
		y=self.lc_nrj
		self.plot_simple(x,y,xlabel='Time (days)',ylabel='Flux (erg/s)',text=text,xlog=False)
		filename.append('%s_energy_lightcurve' % (text.replace(' ','_').replace('(','').replace(')','')))
		y=self.lc_ph
		self.plot_simple(x,y,xlabel='Time (days)',ylabel='Flux (ph/s)',text=text,xlog=False)
		filename.append('%s_photon_lightcurve' % (text.replace(' ','_').replace('(','').replace(')','')))
		# Return plot name
		return filename

	# Plot spectrum evolution along selected rays
	def plot_rays(self,plot_raystep,text):
		# Filename container
		filename=[]
		# Weighing for nu.F(nu) plotting
		# (F(nu) is in erg/s/eV so need to convert photon energies from MeV to eV)
		w=self.gam*1e6
		# Loop over rays
		for  i in range(self.nray):
			if float(i+1)%plot_raystep == 0:
				raytext='ray %d' % (i+1)
				self.plot_distrib_series(self.gam,self.spec_all[i],w=w,xlabel='Photon energy (MeV)',ylabel=r'E $\times$ L(E)'+' (erg/s)',text=text+' - '+raytext)
				filename.append('%s_ray%d' % (text.replace(' ','_').replace('(','').replace(')',''),i+1))
		# Return plot names
		return filename

	# Plot total distribution evolution over time
	def plot_totals(self,text):
		# Filename container
		filename=[]
		# Weighing for nu.F(nu) plotting
		# (F(nu) is in erg/s/eV so need to convert photon energies from MeV to eV)
		w=self.gam*1e6
		# Plot totals
		self.plot_distrib_series(self.gam,self.spec_tot,w=w,xlabel='Photon energy (MeV)',ylabel=r'E $\times$ L(E)'+' (erg/s)',text=text)
		filename.append('%s_total' % (text.replace(' ','_').replace('(','').replace(')','')))
		# Return plot names
		return filename

	# Plotting simple curve
	def plot_simple(self,x,y,xlabel='',ylabel='',text='',xlog=True,ylog=True,xbds=[],ybds=[]):
		# Create object
		fig=plt.figure()
		ax=fig.add_subplot(111)
		plt.plot(x,y,'-',color='blue')
		# Axis scales
		xmin=x.min()
		xmax=x.max()
		ymin=y.min()
		ymax=y.max()
		if xlog == True:
			plt.xscale('log')
			plt.xlim([xmin,xmax])
		else:
			plt.xscale('linear')
			plt.xlim([xmin,xmax])

		if ylog == True:
			plt.yscale('log')
			plt.ylim([1e-3*ymax,10.0*ymax])
		else:
			plt.yscale('linear')
			plt.ylim([0.9*ymin,1.1*ymax])
		# User-defined axis bounds
		if len(xbds) > 0:
			plt.xlim(xbds)
		if len(ybds) > 0:
			plt.xlim(ybds)
		# Axis labels
		plt.xlabel(xlabel,labelpad=10)
		plt.ylabel(ylabel,labelpad=10)
		# Text
		if len(text) > 0:
			plt.text(0.08, 0.92,text,horizontalalignment='left',verticalalignment='center',rotation=0,transform=ax.transAxes,color='black')
		# Grid and ticks
		plt.grid(True)
		plt.tight_layout()
		# Return plot object
		return fig

	# Plotting series
	def plot_distrib_series(self,x,ylist,w=np.ndarray,xlabel='',ylabel='',text='',xlog=True,ylog=True,xbds=[],ybds=[]):
		# Create object
		fig=plt.figure()
		ax=fig.add_subplot(111)
		# Set color map
		clist = cm.rainbow(np.linspace(0,1,len(ylist)))
		# Loop over distributions
		xmin=x.min()
		xmax=x.max()
		ymin=1e200
		ymax=0.0
		for y,c in zip(ylist,clist):
			if w.size > 0:
				y=y*w
			ymin=min([ymin,y.min()])
			ymax=max([ymax,y.max()])
			plt.plot(x,y,'-',color=c)
		# Axis scales
		if xlog == True:
			plt.xscale('log')
			plt.xlim([xmin,xmax])
		else:
			plt.xscale('linear')
			plt.xlim([xmin,xmax])
		if ylog == True:
			plt.yscale('log')
			plt.ylim([1e-4*ymax,10.0*ymax])
		else:
			plt.yscale('linear')
			plt.ylim([0.9*ymin,1.1*ymax])
		# User-defined axis bounds
		if len(xbds) > 0:
			plt.xlim(xbds)
		if len(ybds) > 0:
			plt.xlim(ybds)
		# Axis labels
		plt.xlabel(xlabel,labelpad=10)
		plt.ylabel(ylabel,labelpad=10)
		# Text
		if len(text) > 0:
			plt.text(0.08, 0.92,text,horizontalalignment='left',verticalalignment='center',rotation=0,transform=ax.transAxes,color='black')
		# Grid and ticks
		plt.grid(True)
		plt.tight_layout()
		# Return plot object
		return fig

	# Model function for fitting (the model function is just our spectral model to scale up/down)
	def func_spec_mod(self,x, k):
		# Log-log cubic interpolation for given x
		finterp=interp.interp1d(self.gam,self.spec_flux,kind='cubic')
		y=k*finterp(x)
		# Return y value
		return y

	# Integral of x.F(x).dlog(x) (taken from Dubus)
	def log_integral(self,x,y):
		res=0.0
		if x.size != y.size:
			print 'Incompatible arrays for log-integral calculation.'
		else:
			for i in range(1,x.size):
				res += (x[i]-x[i-1])/(x[i]+x[i-1])*(x[i]*y[i]+x[i-1]*y[i-1])
		return res
