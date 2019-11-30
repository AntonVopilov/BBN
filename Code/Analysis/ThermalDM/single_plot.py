"""
Similar to distributions.py but outputs each panel to a single plot. Useful for presentation purposes. Figure sizes can be changed here.
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
	scenarios = ['Nu']
	for scenario in scenarios:
		figsize = (6, 8)
		fig1 = plt.figure(figsize=figsize)
		fig2 = plt.figure(figsize=figsize)
		fig3 = plt.figure(figsize=figsize)
		ax1 = fig1.add_subplot(1, 1, 1)
		ax2 = fig2.add_subplot(1, 1, 1)
		ax3 = fig3.add_subplot(1, 1, 1)
		axes = [ax1, ax2, ax3]
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
				# extra_data = np.loadtxt(case + '_Extra.txt', skiprows=1)
				# masses = np.append(masses, extra_data[:, 0])
				# Neff = np.append(Neff, extra_data[:, 2])
				# Yp = np.append(Yp, 4*extra_data[:, 8])
				# DoverH = np.append(DoverH, 10**5*extra_data[:, 5]/extra_data[:, 4])
				Neffinterp = interp1d(masses, Neff, kind='cubic')
				Ypinterp = interp1d(masses, Yp, kind='cubic')
				DoverHinterp = interp1d(masses, DoverH, kind='cubic')
				massinterp = np.geomspace(0.1, 29.9, 1000)

				ax1.plot(massinterp, Ypinterp(massinterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[idx],
					label=labels[idx],
					linestyle=linestyles[idx])

				ax2.plot(massinterp, DoverHinterp(massinterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[idx],
					label=labels[idx],
					linestyle=linestyles[idx])

				ax3.plot(massinterp, Neffinterp(massinterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[idx],
					label=labels[idx],
					linestyle=linestyles[idx])
			except:
				continue

		YpCentre = 0.245
		YpError = np.sqrt(0.003**2 + 0.00017**2)
		DHCentre = 2.569
		DHError = np.sqrt(0.027**2 + 0.036**2)
		NeffCentre = 2.89
		NeffError = 0.31

		labelsize = 22

		ax1.add_patch(plt.Rectangle(xy=(0.1, YpCentre - 2*YpError),
	                            width=(30.0 - 0.1),
	                            height=4*YpError,
	                            alpha=0.1,
	                            color='k'))
		ax2.add_patch(plt.Rectangle(xy=(0.1, (DHCentre - 2*DHError)),
	                            width=(30.0 - 0.1),
	                            height=4*DHError,
	                            alpha=0.1,
	                            color='k'))
		ax3.add_patch(plt.Rectangle(xy=(0.1, (NeffCentre - 2*NeffError)),
	                            width=(30.0 - 0.1),
	                            height=4*NeffError,
	                            alpha=0.1,
	                            color='k'))

		ax3.axhline(3.046, color='k', linestyle='--', alpha=0.8)

		ax1.set_xscale('log')
		ax1.set_ylabel(r'$Y_{\mathrm{P}}$', fontsize=labelsize)
		ax2.set_xscale('log')
		ax3.set_xscale('log')
		ax3.set_ylabel(r'$N_{\mathrm{eff}}$', fontsize=labelsize)
		
		# ax1.tick_params(axis='x', which='minor', size=2)
		# ax2.tick_params(axis='x', which='minor', size=2)
		# ax1.tick_params(axis='y', which='minor', size=2)
		# ax2.tick_params(axis='y', which='minor', size=2)
		# ax2.tick_params(axis='x', which='major', size=4)
		# ax2.tick_params(axis='y', which='major', size=4)
		# ax2.tick_params(axis='y', which='major', size=4)

		ml = matplotlib.ticker.MultipleLocator(5)
		ax1.yaxis.set_minor_locator(ml)
		plt.rcParams["xtick.minor.visible"] =  True


		if scenario == 'Nu':
			#ax3.text(2.3, 6.0, 'Neutrinophilic', fontsize=20)
			#ax3.set_title(r'$\mathrm{Neutrinophilic}$', fontsize=26)
			ax3.text(0.12, 3.15, 'SM', fontsize=14)
			ax1.set_xlim(0.1, 30.0)
			ax1.set_ylim(0.23, 0.275)
			ax2.set_xlim(0.1, 30.0)
			ax2.set_ylim(2.4, 3.5)
			ax3.set_xlim(0.1, 30.0)
			ax3.set_ylim(2.0, 6.5)
		if scenario == 'EE':
			#ax3.text(3.2, 3.7, 'Electrophilic', fontsize=20)
			ax3.text(0.12, 3.1, 'SM', fontsize=14)
			#ax3.set_title(r'$\mathrm{Electrophilic}$', fontsize=26)
			ax1.set_xlim(0.1, 30.0)
			ax1.set_ylim(0.23, 0.30)
			ax2.set_xlim(0.1, 30.0)
			ax2.set_ylim(0.9, 2.8)
			ax3.set_xlim(0.1, 30.0)
			ax3.set_ylim(1.5, 4.0)
			ax2.legend(markerfirst=False, fontsize=14, loc='lower right', handlelength=3)
		ax2.set_ylabel(r'$10^5 \times \mathrm{D}/\mathrm{H}|_{\mathrm{P}}$', fontsize=labelsize)
		

		
		ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
		ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))

		plt.subplots_adjust(wspace=0, hspace=0.1)
		for ax in [ax1, ax2, ax3]:
			ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
			ax.set_xlabel(r'$m_{\chi} \, [\mathrm{MeV}]$', fontsize=labelsize)
			ax.yaxis.set_tick_params(labelsize=20)
			ax.xaxis.set_tick_params(labelsize=22)
			ax.set_xticks([0.1, 1.0, 10.0, 30.0])


		fig1.savefig(scenario + '_Yp_plot.pdf')
		fig2.savefig(scenario + '_DH_plot.pdf')
		fig3.savefig(scenario + '_neff_plot.pdf')



