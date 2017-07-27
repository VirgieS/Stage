#! /Applications/Anaconda/bin/python

from PartDistrib import PartDistrib
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as ptch

class PartHistory:
	
	part=PartDistrib
	nbin=0
	ntime=0
	nray=0
	dist_up_all=[]
	dist_down_all=[]
	dist_up_tot=[]
	dist_down_tot=[]
	dist_up_ray=[]
	dist_down_ray=[]
	nrj_all=[]
	nrj_tot=np.array([])
	
	# Constructor
	def __init__(self,part,time,nray):
		self.part=part
		self.time=time
		self.nbin=part.nbin
		self.ntime=time.size
		self.nray=nray		
		self.dist_up_all=[[np.zeros(self.nbin)]*self.ntime]*nray
		self.dist_down_all=[[np.zeros(self.nbin)]*self.ntime]*nray
		self.dist_up_ray=[np.zeros(self.nbin)]*self.ntime
		self.dist_down_ray=[np.zeros(self.nbin)]*self.ntime
		self.dist_up_tot=[np.zeros(self.nbin)]*self.ntime
		self.dist_down_tot=[np.zeros(self.nbin)]*self.ntime
		self.nrj_all=[[0.0]*self.ntime]*nray
		self.nrj_tot=np.zeros(self.ntime)
	
	# Save distribution at given time for given ray
	def save_time(self,i,u,d):
		self.dist_up_ray[i]=deepcopy(u)
		self.dist_down_ray[i]=deepcopy(d)
		
	# Save distribution evolution along given ray
	def save_ray(self,i):
		self.dist_up_all[i]=deepcopy(self.dist_up_ray)
		self.dist_up_ray=[np.zeros(self.nbin)]*self.ntime
		self.dist_down_all[i]=deepcopy(self.dist_down_ray)
		self.dist_down_ray=[np.zeros(self.nbin)]*self.ntime
		
	# Update total over rays at given time
	def update_total(self,i,u,d):
		if i < self.ntime:
			self.dist_up_tot[i]= self.dist_up_tot[i]+u
			self.dist_down_tot[i]= self.dist_down_tot[i]+d
	
	# Compute maximum momentum for a series of distribution
	def compute_maxima(self):
		# Create working and result arrays
		self.dist_max=[np.zeros(self.ntime)]*self.nray
		mom3=np.power(self.part.mom,3)
		# Loop over rays
		for i in range(self.nray):
			ray_max=np.zeros(self.ntime)
			for j in range(self.ntime):
				ray_max[j]=self.part.mom[np.argmax(self.dist_up_all[i][j]*mom3)]	
			self.dist_max[i]=ray_max
	
	# Compute total particle energy per ray and total
	def compute_energy(self):
		# Convert from MeV to erg
		MeV2erg=1.6022e-6
		# Copy particle distribution instance
		dist=deepcopy(self.part)
		# Reset arrays
		self.nrj_all=[[0.0]*self.ntime]*self.nray
		self.nrj_tot=np.zeros(self.ntime)
		# Loop over rays
		for i in range(self.nray):
			# Loop over times
			for j in range(self.ntime):
				dist.load_momentum_mev(self.dist_up_all[i][j]+self.dist_down_all[i][j])
				self.nrj_all[i][j]=self.log_integral(dist.ekin,dist.fekin*dist.ekin)*MeV2erg
				self.nrj_tot[j]=self.nrj_tot[j]+self.nrj_all[i][j]
		
	# Plot particle energy buildup 
	def plot_energy(self,text):
		# Filename container
		filename=[]
		# Plot
		x=self.time
		y=self.nrj_tot	
		self.plot_simple(x,y,xlabel='Time (days)',ylabel='Particle distribution energy (erg)',text=text,xlog=False)
		filename.append('%s_distrib_energy.' % (text.replace(' ','_').replace('(','').replace(')','')))
		# Return plot name
		return filename
	
	# Plot maxima evolution over time for selected rays
	def plot_maxima(self,plot_raystep):	
		# Filename container
		filename=[]
		# Compute maxima
		self.compute_maxima()
		# Loop over rays to extract only selected rays
		sel_max=[]
		for i in range(self.nray):
			if float(i+1)%plot_raystep == 0:
				sel_max.append(self.dist_max[i])
		# Plot maxima series if any
		if len(sel_max) > 0:
			x=self.time
			y=sel_max		
			self.plot_distrib_series(x,y,w=np.empty(0),xlabel='Time (days)',ylabel='Maximum momentum (MeV/c)',text=self.part.id,xlog=False)
			filename.append('%s_maxima' % (self.part.id))
		# Return plot name
		return filename
		
	# Plot distribution evolution along selected rays
	def plot_rays(self,plot_raystep):	
		# Filename container
		filename=[]
		# Weighing for up and down distributions
		w1=self.part.mom
		w2=self.part.mom*self.part.mom
		# Loop over rays
		for i in range(self.nray):
			if float(i+1)%plot_raystep == 0:
				raytext='ray %d' % (i+1)
				self.plot_distrib_series(self.part.mom,self.dist_up_all[i],w=w1,xlabel='Momentum (MeV/c)',ylabel=r'p $\times$ N(p)',text='%s (acceleration zone) - %s'% (self.part.id,raytext))
				filename.append('%s_up_series_ray%d' % (self.part.id,i+1))			
				self.plot_distrib_series(self.part.mom,self.dist_down_all[i],w=w2,xlabel='Momentum (MeV/c)',ylabel=r'p$^2$ $\times$ N(p)',text='%s (cooling zone) - %s'% (self.part.id,raytext))
				filename.append('%s_down_series_ray%d' % (self.part.id,i+1))
		# Return plot names
		return filename
				
	# Plot total distribution evolution over time
	def plot_totals(self):	
		# Filename container
		filename=[]
		# Weighing for up and down distributions
		w1=self.part.mom
		w2=self.part.mom*self.part.mom
		# Plot totals		
		self.plot_distrib_series(self.part.mom,self.dist_up_tot,w=w1,xlabel='Momentum (MeV/c)',ylabel=r'p $\times$ N(p)',text='%s (acceleration zone) - Total' % (self.part.id))
		filename.append('%s_up_series_total' % (self.part.id))
		self.plot_distrib_series(self.part.mom,self.dist_down_tot,w=w2,xlabel='Momentum (MeV/c)',ylabel=r'p$^2$ $\times$ N(p)',text='%s (cooling zone) - Total' % (self.part.id))
		filename.append('%s_down_series_total' % (self.part.id))
		# Return plot names
		return filename
		
	# Plotting series
	def plot_distrib_series(self,x,ylist,w=np.ndarray,xlabel='',ylabel='',text='',xlog=True,ylog=True):
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
			plt.ylim([1e-6*ymax,10.0*ymax])
		else:
			plt.yscale('linear')
			plt.ylim([0.9*ymin,1.1*ymax])
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
		
	# Integral of x.F(x).dlog(x) (taken from Dubus)
	def log_integral(self,x,y):
		res=0.0
		if x.size != y.size:
			print 'Incompatible arrays for log-integral calculation.'
		else:
			for i in range(1,x.size):
				res += (x[i]-x[i-1])/(x[i]+x[i-1])*(x[i]*y[i]+x[i-1]*y[i-1])
		return res
	
	