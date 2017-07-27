#! /Applications/Anaconda/bin/python

import os
import sys
import time

from math import log
from math import log10
from math import sqrt
from math import sin
from math import cos
from math import exp
from math import pi
from math import floor

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as ptch		


#===================#
# Plot simple curve #
#===================#
def plot_simple(x,y,xlabel='',ylabel='',text='',xlog=True,ylog=True,xbds=[],ybds=[],color='blue',mark='-',label=''):
	# Create object
	fig=plt.figure()
	ax=fig.add_subplot(111)	
	plt.plot(x,y,mark,color=color,label=label)			
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
		plt.ylim([1e-2*ymax,10.0*ymax])
	else:
		plt.yscale('linear')
		plt.ylim([0.9*ymin,1.1*ymax])							
	# User-defined axis bounds
	if len(xbds) > 0:
		plt.xlim(xbds)
	if len(ybds) > 0:
		plt.ylim(ybds)
	# Axis labels
	plt.xlabel(xlabel,labelpad=10)
	plt.ylabel(ylabel,labelpad=10)				
	# Text and legend
	plt.legend(loc='upper right')
	if len(text) > 0:
		plt.text(0.75, 0.80,text,horizontalalignment='left',verticalalignment='center',rotation=0,transform=ax.transAxes,color='black',fontsize=15)	
	# Grid and ticks
	plt.grid(True)
	plt.tight_layout()		
	# Return plot object
	return fig


#==============================#
# Plot multiple series of data #
#==============================#
def plot_distrib_series(x,ylist,w=np.ndarray,xlabel='',ylabel='',text='',xlog=True,ylog=True,labellist=[],clist=[],mlist=[],xbds=[],ybds=[]):
	"""
	Plot series of spectral distributions
	
	Parameters
	- ... 

	Comments
	- ...
	"""
	# Create object
	fig=plt.figure()
	ax=fig.add_subplot(111)	
	# Set marker list
	if len(mlist) != len(ylist):
		mlist = ['-']*len(ylist)
	# Set color map
	if len(clist) != len(ylist):
		clist = cm.rainbow(np.linspace(0,1,len(ylist)))
	# Check label list
	if len(labellist) != len(ylist):
		labellist=map(str,range(1,len(ylist)+1))
	# Loop over distributions
	xmin=x.min()
	xmax=x.max()
	ymin=1e200
	ymax=0.0
	for y,c,l,m in zip(ylist,clist,labellist,mlist):
		if w.size > 0:
			y=y*w
		ymin=min([ymin,y.min()])
		ymax=max([ymax,y.max()])
		plt.plot(x,y,m,color=c,label=l)		
	# Axis scales
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
		plt.ylim(ybds)
	# Axis labels
	plt.xlabel(xlabel,labelpad=10)
	plt.ylabel(ylabel,labelpad=10)				
	# Text and legend
	plt.legend(loc='upper right')
	if len(text) > 0:
		plt.text(0.08, 0.92,text,horizontalalignment='left',verticalalignment='center',rotation=0,transform=ax.transAxes,color='black',fontsize=20)	
	# Grid and ticks
	plt.grid(True)
	plt.tight_layout()		
	# Return plot object
	return fig


#=========================#
# Set graphics properties #
#=========================#
def set_graphics(plot_write,file_type):
	"""
	Set graphics properties for screen or file
	
	Parameters
	- Flag to specify whether plots are to be shown on screen or saved to file
	- Type of graphic file to be written

	Comments
	- ...
	"""
	# File	
	if plot_write:	
		params = {'legend.fontsize': 20,
		  'legend.labelspacing': 0.2,
		  'figure.figsize': (12,9),
		  'figure.facecolor': 'white',
		  'lines.linewidth': 2,
		  'font.size' : 15,
		  'axes.titlesize': 20,
		  'axes.labelsize': 20,
		  'xtick.labelsize': 20,
		  'ytick.labelsize': 20,
		  'axes.formatter.limits': (-3,3),
		  'savefig.dpi':200,
		  'savefig.format': file_type,
		  'savefig.bbox':'tight'}
	# Screen
	else:
		params = {'legend.fontsize': 20,
		  'legend.labelspacing': 0.2,
		  'figure.figsize': (12,9),
		  'figure.facecolor': 'white',
		  'lines.linewidth': 2,
		  'font.size' : 15,
		  'axes.titlesize': 20,
		  'axes.labelsize': 20,
		  'xtick.labelsize': 20,
		  'ytick.labelsize': 20,
		  'axes.formatter.limits': (-3,3)}
	# Load preferences
	plt.rcParams.update(params)
	
	
	
	