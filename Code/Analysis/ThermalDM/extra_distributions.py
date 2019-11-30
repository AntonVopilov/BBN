"""
Similar to distributions.py but for Lithium and He3 abundances.
"""

from utils import get_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.ticker
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
	scenarios = ['Nu', 'EE']
	for scenario in scenarios:
		figsize = (6, 10)
		plt.figure(figsize=figsize)
		#plt.subplots_adjust(top=0.94)
		ax1 = plt.subplot(2, 1, 1)
		ax2 = plt.subplot(2, 1, 2)
		axes = [ax1, ax2]
		if 'Nu' in scenario:
			ax1.set_title(r'$\mathrm{Neutrinophilic}$', fontsize=22)
		else:
			ax1.set_title(r'$\mathrm{Electrophilic}$', fontsize=22)
		colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']
		linestyles = ['--',
		              (0, (5,1)),
		              (0, (3, 1, 1, 1)),
		              (0, (1, 1)),
		              '-']

		labels = ['Neutral Scalar',
				  'Complex Scalar',
				  'Vector Boson',
				  'Majorana Fermion',
				  'Dirac Fermion']

		cases = [scenario + '_Neutral_Scalar',
				 scenario + '_Complex_Scalar',
				 scenario + '_Zp',
				 scenario + '_Maj',
				 scenario + '_Dirac',]

		for idx, case in enumerate(cases):
			try:
				if scenario == 'Nu':
					index = 30 # Omegab h2 = 0.021875
				else:
					index = 62
				data = get_data(case + '.txt')
				masses = np.unique(data['mass'])
				omegab = data['OmegaB'].reshape((len(masses), -1))
				Yp = data['Yp'].reshape((len(masses), -1))
				Yp = Yp[:, index]
				Neff = data['Neff'].reshape((len(masses), -1))
				Neff = Neff[:, 0]
				DoverH = (10**5)*data['D/H'].reshape((len(masses), -1))
				DoverH = DoverH[:, index]
				LioverH = data['Li/H'].reshape((len(masses), -1))
				LioverH = LioverH[:, index]
				HeoverH = data['3He/H'].reshape((len(masses), -1))
				HeoverH = HeoverH[:, index]
				# extra_data = np.loadtxt(case + '_Extra.txt', skiprows=1)
				# masses = np.append(masses, extra_data[:, 0])
				# Neff = np.append(Neff, extra_data[:, 2])
				# Yp = np.append(Yp, 4*extra_data[:, 8])
				# DoverH = np.append(DoverH, 10**5*extra_data[:, 5]/extra_data[:, 4])
				Neffinterp = interp1d(masses, Neff, kind='cubic')
				Ypinterp = interp1d(masses, Yp, kind='cubic')
				DoverHinterp = interp1d(masses, DoverH, kind='cubic')
				LioverHinterp = interp1d(masses, 10**(10)*LioverH, kind='cubic')
				HeoverDinterp = interp1d(masses, 10**5*HeoverH, kind='cubic')
				massinterp = np.geomspace(0.1, 29.9, 1000)

				ax1.plot(massinterp, LioverHinterp(massinterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[idx],
					label=labels[idx],
					linestyle=linestyles[idx])

				ax2.plot(massinterp, HeoverDinterp(massinterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[idx],
					label=labels[idx],
					linestyle=linestyles[idx])
			except:
				continue

		LiCentre = 1.6
		LiError = 0.3
		HeCentre = 0.098
		HeError = 1.0

		labelsize = 22

		# ax1.add_patch(plt.Rectangle(xy=(0.1, LiCentre - 2*LiError),
	 #                            width=(30.0 - 0.1),
	 #                            height=4*LiError,
	 #                            alpha=0.1,
	 #                            color='k'))
		ax2.add_patch(plt.Rectangle(xy=(0.1, 1.3),
	                            width=(30.0 - 0.1),
	                            height=0.4,
	                            alpha=0.3,
	                            color='k',
	                            fill=None, hatch='///'),
								)

		ax1.set_xscale('log')
		ax1.set_ylabel(r'$10^{10} \times$' + r'$^{7}\mathrm{Li}/\mathrm{H}|_{\mathrm{p}}$', fontsize=labelsize)
		ax2.set_xscale('log')
		
		# ax1.tick_params(axis='x', which='minor', size=2)
		# ax2.tick_params(axis='x', which='minor', size=2)
		# ax1.tick_params(axis='y', which='minor', size=2)
		# ax2.tick_params(axis='y', which='minor', size=2)
		ax1.tick_params(axis='x', which='major', labelsize=0)
		# ax2.tick_params(axis='x', which='major', size=4)
		# ax2.tick_params(axis='y', which='major', size=4)
		# ax2.tick_params(axis='y', which='major', size=4)

		ml = matplotlib.ticker.MultipleLocator(5)
		ax1.yaxis.set_minor_locator(ml)
		plt.rcParams["xtick.minor.visible"] =  True


		if scenario == 'Nu':
			ax2.text(1.0, 1.32, 'Excluded', fontsize=14)
			ax1.set_xlim(0.1, 30.0)
			#ax1.set_ylim(0.23, 0.275)
			ax2.set_xlim(0.1, 30.0)
			ax2.set_ylim(1.05, 1.35)
		if scenario == 'EE':
			ax2.text(1.0, 1.34, 'Excluded', fontsize=14)
			ax1.set_xlim(0.1, 30.0)
			#ax1.set_ylim(0.23, 0.30)
			ax2.set_xlim(0.1, 30.0)
			ax2.set_ylim(0.8, 1.4)
			ax1.legend(markerfirst=False, fontsize=14, loc='upper right', handlelength=3)
		ax2.set_xlabel(r'$m_{\chi} \, [\mathrm{MeV}]$', fontsize=labelsize)
		ax2.set_ylabel(r'$10^5 \times$' + r'$^{3}\mathrm{He}/\mathrm{H}|_{\mathrm{p}}$', fontsize=labelsize)
		ax2.set_xticks([0.1, 1.0, 10.0, 30.0])

		ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
		ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))

		plt.subplots_adjust(wspace=0, hspace=0.1)
		for ax in [ax1, ax2]:
			ax.yaxis.set_tick_params(labelsize=20)
		ax2.xaxis.set_tick_params(labelsize=20)

		plt.savefig(scenario + '_extra_abundance_plot.pdf')