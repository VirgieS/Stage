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
