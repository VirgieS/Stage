#! /Applications/Anaconda/bin/python

import os
import sys
from HydroTableInternalShock import HydroTableInternalShock
from RadHistory import RadHistory
from PartHistory import PartHistory

import cPickle as pkl

from math import *
import numpy as np

#=========================#
# Load simulation results #
#=========================#
def load_results(resdir):

	# Internal (file structures)
	num_rad_pkl=5
	i_brem=0
	i_icnova=1
	i_pion=2
	i_tot=3
	i_sync=4
	num_part_pkl=6
	i_novapars=0
	i_hydroclass=1
	i_hydrovars=2
	i_elechist=3
	i_prothist=4
	i_effi=5

	# Get particle acceleration data
	infile= resdir+'/distrib.pkl'
	if (os.path.isfile(infile) == False):
		print "File ",infile," does not exist. Exit."
		return np.empty([0]),np.empty([0])

	# Open result pickle file and load data (keeping the same order used for saving them)
	respkl=open(infile, 'rb')
	pkldata=[]
	for i in range(num_part_pkl):
		print(num_part_pkl)
		pkldata.append(pkl.load(respkl))
	respkl.close()
	hydro_data = pkldata[i_hydrovars]

	# Get radiation results file name
	infile= resdir+'/radiation.pkl'
	if (os.path.isfile(infile) == False):
		print "File ",infile," does not exist. Exit."
		return np.empty([0]),np.empty([0]),np.empty([0]),np.empty([0]),np.empty([0]),np.empty([0]),np.empty([0]),np.empty([0]),np.empty([0])

	# Open result pickle file and load data (keeping the same order used for saving them)
	respkl=open(infile, 'rb')
	pkldata=[]
	for i in range(num_rad_pkl):
		pkldata.append(pkl.load(respkl))
	gamma_data= pkldata[i_tot]

	# Return those
	return hydro_data,gamma_data



#=====================#
# Routine entry point #
#=====================#
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

	# Print out what we need
	"""
	print 'Time array (days)'
	print hydro_data[0]['Time']
	print 'Shock radius (AU)'
	print hydro_data[0]['Rshock']
	print 'Shock velocity (km/s)'
	print hydro_data[0]['Vframe']
	print 'Gamma-ray energies (MeV)'
	print gamma_data.gam
	"""

	print 'Gamma-ray spectrum evolution over time (erg/s/eV)'
	for i in range(gamma_data.time.size):
		print 't=%.2f day : ' % gamma_data.time[i]
		print gamma_data.spec_tot[i]
