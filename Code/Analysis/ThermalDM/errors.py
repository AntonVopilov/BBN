"""
errors.py

Generates the errors plot as a function of experimental error in D/H and Yp

To Note:

- Important: The correct data files need to be used. This is different naming convention for EE and Nu particles. See __main__ for more details.
- Process will open a GUI window where the contour labels are chosen, the plots are then saved from there.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
from matplotlib.patches import Rectangle
from matplotlib import ticker
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline, interp1d, interp2d

from utils import get_data
from utils import get_chisq_grid, get_chisq_marg_data, get_mass_omegab_grid, get_masses
from utils import chisq

from joblib import Parallel, delayed

# Uncomment code in get_bound for Neutrinophilic, then comment if/else clause

def get_bound(data, errors):
	masses = get_masses(data)
	DeltaChisqBBN = get_chisq_marg_data(data, type='BBN', errors=errors)
	stddev = 0.954499736103642
	cl = chi2.ppf(stddev, 1)
	mask = (DeltaChisqBBN <= cl)
	if len(masses[mask]) == 0:
		return 0.0
	else:
		return np.min(masses[mask])
	# BBNInterp = UnivariateSpline(masses, DeltaChisqBBN - cl, s=0)
	# BBNroot = BBNInterp.roots()[0] if any(BBNInterp.roots()) else 0.0
	# return BBNroot

ticks = {'Nu_Maj': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
		 'Nu_Maj_New': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
		 'EE_Maj': [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4],
		 'EE_Maj_Sigma': [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]}

pads = {'Nu_Maj': 12,
		'Nu_Maj_New': 12,
		'EE_Maj': 20,
		'EE_Maj_Sigma': 20}

colors = {'Nu_Maj': '#BF4145',
		  'Nu_Maj_New': '#BF4145',
		  'EE_Maj': 'yellow',
		  'EE_Maj_Sigma': 'yellow'}

edges = {'Nu_Maj': 'w',
		  'Nu_Maj_New': 'w',
		  'EE_Maj': 'k',
		  'EE_Maj_Sigma': 'k'}

cmaps = {'Nu_Maj': 'Greens_r',
		 'Nu_Maj_New': 'Greens_r',
		 'EE_Maj': 'Blues_r',
		'EE_Maj_Sigma': 'Blues_r'}

files = {'Nu_Maj': 'Nu_Maj',
		 'Nu_Maj_New': 'Nu_Maj',
		 'EE_Maj': 'EE_Maj',
		 'EE_Maj_Sigma': 'EE_Maj'}

levels_list = {'Nu_Maj_New': [1.0, 3.0, 5.0, 8.0],
				'EE_Maj': [1.0, 4.0, 8.0],
			   'EE_Maj_Sigma': [0.5, 1.0, 1.5]}

strs_list = {'Nu_Maj_New': ['1 MeV', '3 MeV', '5 MeV', '8 MeV'],
			 'EE_Maj': ['1 MeV', '4 MeV', '8 MeV'],
			   'EE_Maj_Sigma': ['0.5 MeV', '1 MeV', '1.5 MeV']}


if __name__ == '__main__':
	# IMPORTANT: Need to run with EE_Maj_Sigma or Nu_Maj_New for the correct data files
	cases = ['EE_Maj_Sigma']
	for case in cases:
		data = get_data(case + '.txt')
		nptsYp = 400
		nptsDH = 400
		YpMax = -2
		YpMin = -3.6
		DHMax = -5.5
		DHMin = -7.7
		sigmaYp = np.logspace(YpMin, YpMax, nptsYp)
		sigmaDH = np.logspace(DHMin, DHMax, nptsDH)
		SIGMAYP, SIGMADH = np.meshgrid(sigmaYp, sigmaDH)
		sigmaYpR = SIGMAYP.reshape(1, nptsYp*nptsDH)[0, :]
		sigmaDHR = SIGMADH.reshape(1, nptsYp*nptsDH)[0, :]

		bounds = Parallel(n_jobs=4)(delayed(get_bound)(data=data, errors=(sigmaYpR[idx], sigmaDHR[idx])) for idx in range(0, nptsYp*nptsDH))
		bounds = np.array(bounds).reshape(nptsDH, nptsYp)

		# boundsfn = interp2d(sigmaYp, sigmaDH, bounds, kind='cubic')

		# nptsYp = 1000
		# nptsDH = 1000

		# SIGMAYP, SIGMADH = np.meshgrid(np.logspace(YpMin, YpMax, nptsYp), np.logspace(DHMin, DHMax, nptsDH))
		# BOUNDS = boundsfn(np.logspace(YpMin, YpMax, nptsYp), np.logspace(DHMin, DHMax, nptsDH))

		plt.figure(figsize=(6, 5))
		if case == 'EE_Maj_Sigma':
			plt.title(r'$\mathrm{Electrophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$', fontsize=20)
			vmin = 0.1
			vmax = 2.15
		else:
			plt.title(r'$\mathrm{Neutrinophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$', fontsize=20)
			vmin = -1.0
			vmax = 10.0

		levels = [0.1, 0.5, 1.0, 2.0]
		ct = plt.contourf((1/0.247)*SIGMAYP, (1/(2.439*10**(-5)))*SIGMADH, bounds, 
		                 # locator=ticker.LogLocator(),
		                 cmap=cmaps[case],
		                 vmin=vmin,
		                 vmax=vmax,
		                 # interpolation='nearest'
		                 )
		cbar = plt.colorbar(aspect=30)
		cbar.set_ticks(ticks[case])
		cbar.ax.minorticks_off()
		cbar.ax.set_ylabel(r'$m_{\chi} \, \mathrm{[MeV]}$', labelpad=pads[case], rotation=270, fontsize=16)
		ct = plt.contour((1/0.247)*SIGMAYP, (1/(2.439*10**(-5)))*SIGMADH, bounds, 
		                 # locator=ticker.LogLocator(),
		                 levels=levels_list[case],
		                 colors='k',
		                 linewidths=1.2,
		                 linestyles=[(0, (5, 1))],
		                 # vmin=vmin,
		                 # vmax=vmax,
		                 # interpolation='nearest')
		                 )
		fmt = {}
		strs = strs_list[case]
		for l, s in zip(ct.levels, strs):
		    fmt[l] = s
		# ct = plt.contour(SIGMAYP, SIGMADH, BOUNDS, 
		#                  levels=levels,
		#                  linewidths=0.5,
		#                  colors=['#3F7BB6'],
		#                  vmin=vmin,
		#                  vmax=vmax,
		#                  interpolation='nearest')
		
		plt.xlabel(r'$\mathrm{Fractional} \, \mathrm{Error}\,\,Y_{\mathrm{p}}$', fontsize=20)
		plt.ylabel(r'$\mathrm{Fractional} \, \mathrm{Error}\,\,\mathrm{D}/\mathrm{H}|_{\mathrm{p}}$', fontsize=20)
		plt.xscale('log')
		plt.yscale('log')
		plt.scatter([0.003/0.245], [0.045/2.569], 
			marker='*',
			c=colors[case],
			alpha=1.0, 
			s=300,
			linewidths=0.1,
			edgecolors=edges[case],
			)
		plt.xlim(1e-3, 3e-2)
		plt.ylim(1e-3, 1e-1)
		plt.clabel(ct, 
			# fmt=r'%1.0f MeV',
			fmt=fmt,
			fontsize=16, 
			inline=True,
			rightside_up=True,
			manual=True)
		# plt.show()
		#plt.savefig('{}_Errors.pdf'.format(files[case]))




	
	

