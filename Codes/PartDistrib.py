#! /Applications/Anaconda/bin/python

from copy import deepcopy
from math import log10
from math import sqrt
import numpy as np
import scipy.integrate as integ

class PartDistrib:
	
	id=''
	mass=0.0
	A=0
	Z=0
	nbin=0
	total_num=0.0
	total_energy=0.0
	unit=''
	
	# Constructor
	def __init__(self,id):
		if id == 'electron':
			self.id='electron'
			self.mass=0.510998910
			self.A=0
			self.Z=-1
		elif id == 'positron':
			self.id='positron'
			self.mass=0.510998910
			self.A=0
			self.Z=1
		elif id == 'proton':
			self.id='proton'
			self.mass=938.272046
			self.A=1
			self.Z=1
		elif id == 'alpha':
			self.id='alpha'
			self.mass=3.72738e3 
			self.A=4
			self.Z=2
	
	# Set arrays with energy in MeV and momentum in MeV/c
	def set_ekin_mev(self,n,kmin,kmax):
		# Number of bins
		self.nbin=n
		# X arrays
		self.ekin=np.logspace(log10(kmin),log10(kmax),num=n,base=10.0)
		self.etot=self.ekin+self.mass
		self.gam=self.etot/self.mass
		self.bet=np.sqrt(1.0-1.0/(self.gam*self.gam))
		self.mom=np.sqrt(self.etot*self.etot-self.mass*self.mass)
		# Y arrays
		self.fmom=np.zeros(n)
		self.fekin=np.zeros(n)
		self.fgam=np.zeros(n)
		# Totals
		self.total_num=0.0
		self.total_energy=0.0
		# Unit
		self.unit='mev'
	def set_mom_mev(self,n,kmin,kmax):
		# Number of bins
		self.nbin=n
		# X arrays
		self.mom=np.logspace(log10(kmin),log10(kmax),num=n,base=10.0)
		self.etot=np.sqrt(self.mom*self.mom+self.mass*self.mass)
		self.gam=self.etot/self.mass
		self.bet=np.sqrt(1.0-1.0/(self.gam*self.gam))
		self.ekin=self.etot-self.mass
		# Y arrays
		self.fmom=np.zeros(n)
		self.fekin=np.zeros(n)
		self.fgam=np.zeros(n)
		# Totals
		self.total_num=0.0
		self.total_energy=0.0
		# Unit
		self.unit='mev'	
		
	# Load a distribution in MeV or MeV/c and update arrays
	def load_momentum_mev(self,dist):
		if dist.size == self.nbin:
			self.fmom=deepcopy(dist)
			self.fekin=self.fmom*self.etot/self.mom
			self.fgam=self.fekin*self.mass
			self.compute_totals()
	def load_gamma_mev(self,dist):
		if dist.size == self.nbin:
			self.fgam=deepcopy(dist)
			self.fekin=self.fgam/self.mass
			self.fmom=self.fekin*self.mom/self.etot
			self.compute_totals()
	def load_ekin_mev(self,dist):
		if dist.size == self.nbin:
			self.fekin=deepcopy(dist)
			self.fgam=self.fekin*self.mass
			self.fmom=self.fekin*self.mom/self.etot
			self.compute_totals()
	
	# Load a distribution in m_p*c or m_p*c*c and update arrays in MeV/c and MeV
	def load_momentum_mc2(self,dist):
		if dist.size == self.nbin:
			self.fmom=deepcopy(dist)
			self.fmom=self.fmom*938.272046
			self.fekin=self.fmom*self.etot/self.mom
			self.fgam=self.fekin*self.mass
			self.compute_totals()
	def load_gamma_mc2(self,dist):
		if dist.size == self.nbin:
			self.fgam=deepcopy(dist)
			self.fekin=self.fgam/self.mass
			self.fmom=self.fekin*self.mom/self.etot
			self.compute_totals()
	def load_ekin_mc2(self,dist):
		if dist.size == self.nbin:
			self.fekin=deepcopy(dist)
			self.fekin=self.fekin*938.272046
			self.fgam=self.fekin*self.mass
			self.fmom=self.fekin*self.mom/self.etot
			self.compute_totals()	
	
	# Compute total energy and number of particle
	def compute_totals(self):
		# Integration over irregular grid but not particularly adapted to a log grid
		#self.total_num=integ.simps(self.fmom,self.mom,even='first')
		#self.total_energy=integ.simps(self.fekin*self.ekin,self.ekin,even='first')
		# Integration specific to a log grid
		self.total_num=self.log_integral(self.mom,self.fmom)
		self.total_energy=self.log_integral(self.ekin,self.fekin*self.ekin)
	
	# Compute maximum momentum of distribution
	def compute_maximum_mom(self):
		# Create working array
		mom3=np.power(self.mom,3)
		# Find maximum
		max_mom=self.mom[np.argmax(self.fmom*mom3)]	
		# Return it
		return max_mom
	
	# Clear distributions
	def clear_dist(self):
		self.fekin.fill(0.0)
		self.fgam.fill(0.0)
		self.fmom.fill(0.0)
		self.compute_totals()

	# Set up a power law distribution
	def set_pl_mom(self,idx,ntot,mom_min=0.0,mom_max=0.0):
		dist=np.power(self.mom,idx)
		if mom_min >= self.mom.min() and mom_max <= self.mom.max() and mom_min < mom_max:
			i_mmin=np.searchsorted(self.mom,mom_min)
			i_mmax=np.searchsorted(self.mom,mom_max)
			dist=dist/self.log_integral(self.mom[i_mmin:i_mmax],dist[i_mmin:i_mmax])*ntot
		else:
			dist=dist/self.log_integral(self.mom,dist)*ntot
		self.load_momentum_mev(dist)
	def set_pl_ekin(self,idx,ntot,ekin_min=0.0,ekin_max=0.0):
		dist=np.power(self.ekin,idx)
		if ekin_min >= self.ekin.min() and ekin_max <= self.ekin.max() and ekin_min < ekin_max:
			i_kmin=np.searchsorted(self.ekin,ekin_min)
			i_kmax=np.searchsorted(self.ekin,ekin_max)
			dist=dist/self.log_integral(self.ekin[i_kmin:i_kmax],dist[i_kmin:i_kmax])*ntot
		else:
			dist=dist/self.log_integral(self.ekin,dist)*ntot
		self.load_ekin_mev(dist)
	
	# Integral of x.F(x).dlog(x) (taken from Dubus)
	def log_integral(self,x,y):
		res=0.0
		if x.size != y.size:
			print 'Incompatible arrays for log-integral calculation.'
		else:
			for i in range(1,x.size):
				res += (x[i]-x[i-1])/(x[i]+x[i-1])*(x[i]*y[i]+x[i-1]*y[i-1])
		return res
		
		