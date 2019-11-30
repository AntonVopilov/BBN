"""
neff.py

- compares predictions of PriMiDM and PRIMAT including additional dark radiation
- outputs pdf Neffcheck.pdf
- uses data from DeltaNeffPRIMAT.txt and DeltaNeffPRIMI.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['axes.linewidth'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

if __name__ == '__main__':
	primatData = np.loadtxt('DeltaNeffPRIMAT.txt', skiprows=1, delimiter=',')
	primiData = np.loadtxt('DeltaNeffPRIMI.txt', skiprows=1)

	DeltaNeff = primatData[:, 0]

	primi_df = {'DeltaNeff': DeltaNeff,
				'H': primiData[:, 2],
		        'Yp': primiData[:, 6],
		        'D/H x 10^5': primiData[:, 3]/primiData[:, 2],
		        '3He/H x 10^5': primiData[:, 5]/primiData[:, 2],
		        '7Li/H x 10^11': primiData[:, 7]/primiData[:, 2]}

	primat_df = {'DeltaNeff': DeltaNeff,
				 'H': primatData[:, 2],
		         'Yp': primatData[:, 6],
		         'D/H x 10^5': primatData[:, 3]/primatData[:, 2],
		         '3He/H x 10^5': primatData[:, 5]/primatData[:, 2],
		         '7Li/H x 10^11': primatData[:, 7]/primatData[:, 2]}

	figsize = (8, 5)
	plt.figure(figsize=figsize)

	columns = ['H', 
			   'Yp', 
			   'D/H x 10^5', 
			   '3He/H x 10^5', 
			   '7Li/H x 10^11']
	
	column_labels = [r'$\mathrm{H}$', 
					 r'$Y_p$', 
					 r'$\mathrm{D}/\mathrm{H}$', 
					 r'$^{3}\mathrm{He}/\mathrm{H}$', 
					 r'$^{7}\mathrm{Li}/\mathrm{H}$']

	colors = ['#01295F',
			  '#419D78',
			  '#FFBF00',
			  '#D1495B',
			  '#DCDCDD']

	#linestyles = ['-', '--', '-.', ':', (0, (1, 10))]

	colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

	markers = ["^", "o", "s", "*", "d"]
	sizes = np.array([60, 60, 60, 90, 60])

	for idx, column in enumerate(columns):
		plt.plot(primi_df['DeltaNeff'], np.abs(1 - (primi_df[column]/primat_df[column])), 
			c=colors[idx],
			ls='-',
			lw=1.7,
			label='',
			zorder=1)
		plt.scatter(primi_df['DeltaNeff'], np.abs(1 - (primi_df[column]/primat_df[column])), 
			c=colors[idx],
			alpha=0.9,
			s=sizes[idx],
			linewidths=0.4,
			edgecolors='k',
			marker=markers[idx],
			label=column_labels[idx],
			zorder=2)

	plt.yscale('log')
	plt.xlabel(r'$\Delta N_{\mathrm{eff}}$', fontsize=22)
	plt.ylabel(r'$\mathrm{Rel.}\,\,\mathrm{Diff.}$', fontsize=22)
	plt.legend(fontsize=18, 
		title_fontsize=12, 
		loc=4,
		fancybox=False,
		frameon=False,
		markerfirst=False)
	plt.xlim(0.0, 1.0)
	plt.ylim(3*10**(-6), 10**(-3))

	ax = plt.gca()
	ax.xaxis.set_tick_params(labelsize=20)
	ax.yaxis.set_tick_params(labelsize=20)
	# ax.tick_params(axis='x', which='major', size=4)
	# ax.tick_params(axis='y', which='major', size=4)
	# ax.tick_params(axis='y', which='minor', size=2)
	plt.savefig('Neffcheck.pdf')
