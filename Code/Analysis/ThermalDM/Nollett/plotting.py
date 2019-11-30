"""
plotting.py

- compares the predictions of PriMiDM and Nollett (2014/15)
- outputs pdfs NollettEE.pdf and NollettNU.pdf for two scenarios
- uses data from DH_EE_Maj.txt, DH_NU_Maj.txt, PRIMI_Nollett_EE_Maj, PRIMI_Nollett_NU_Maj, YHe_EE_Maj.txt, YHe_NU_Maj.txt
"""

if __name__ == '__main__':
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
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
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20

	scenarios = ['EE', 'NU']
	label = {'EE': 'Nollett et al. (2014)',
			 'NU': 'Nollett et al. (2015)'}
	for scenario in scenarios:
		primi = np.loadtxt('PRIMI_Nollett_{}_Maj'.format(scenario), skiprows=1)
		mass_arr = primi[:, 0]
		Yp_arr = 4*primi[:, 8]
		DoverH_arr = 10**5*primi[:, 5]/primi[:, 4]

		YpMcCabe = np.loadtxt('YHe_{}_Maj.txt'.format(scenario))
		DMcCabe = np.loadtxt('DH_{}_Maj.txt'.format(scenario))

		arrays = [Yp_arr, DoverH_arr]
		mass_arrays = [YpMcCabe[:, 0], DMcCabe[:, 0]]
		McCabe_arrays = [YpMcCabe[:, 1], DMcCabe[:, 1]]


		figsize = (6,6)
		plt.figure(figsize=figsize)
		
		column_labels = [r'$Y_{\mathrm{P}}$', 
						 r'$10^5 \times \mathrm{D}/\mathrm{H}|_{\mathrm{P}}$', ]

		colors = ['#3F7BB6',
				  '#BF4145',]

		markers = ["o", "s"]
		sizes = [10, 10]

		ax1 = plt.subplot(2, 1, 1)
		if 'nu' in scenario.lower():
			ax1.set_title(r'$\mathrm{Neutrinophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$', fontsize=22)
		else:
			ax1.set_title(r'$\mathrm{Electrophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$', fontsize=22)
		ax2 = plt.subplot(2, 1, 2)
		axes = [ax1, ax2]
		if scenario == 'EE':
			locs = ['upper right', 'lower right']
			markerfirstarr = [False, False]
		else:
			locs = ['lower left', 'lower left']
			markerfirstarr = [True, True]

		for idx, color in enumerate(colors):
			# axes[idx].plot(mass_arr, arrays[idx], 
			# 	c='k',
			# 	ls='--',
			# 	lw=0.6,
			# 	label='')
			interpfn = interp1d(mass_arr, arrays[idx], kind='cubic', fill_value='extrapolate')
			new_masses = np.geomspace(0.01, 30.0)
			axes[idx].plot(new_masses, interpfn(new_masses), 
				c=colors[idx],
				alpha=0.9,
				lw=2.2,
				#s=sizes[idx],
				#linewidths=0.4,
				#edgecolors='k',
				#marker=markers[idx],
				#label=column_labels[idx])
				)
			axes[idx].plot([], [], 
				c='k',
				alpha=0.9,
				lw=2.2,
				#s=sizes[idx],
				#linewidths=0.4,
				#edgecolors='k',
				#marker=markers[idx],
				#label=column_labels[idx],
				label='This Work')
			axes[idx].plot(mass_arrays[idx], McCabe_arrays[idx],
				c='k',
				ls=(0, (5, 2)),
				lw=2.2,
				label=label[scenario])
			
			axes[idx].set_xlim(0.1,30.0)
			axes[idx].set_xscale('log')
			axes[idx].set_ylabel(column_labels[idx], fontsize=22)
			if idx == 0:
				axes[idx].set_xticklabels([])
				if scenario == 'EE':
					axes[idx].set_yticks([0.24, 0.25, 0.26, 0.27, 0.28, 0.29])
					axes[idx].set_ylim(0.24, 0.29)
				else:
					axes[idx].set_ylim(0.246, 0.263)
					axes[idx].set_yticks([0.246, 0.250, 0.254, 0.258, 0.262])
			if idx == 1:
				if scenario == 'EE':
					axes[idx].legend(markerfirst=False, handlelength=3, loc='lower right', fontsize=14)
				else:
					axes[idx].legend(markerfirst=True, handlelength=3, loc='lower left', fontsize=14)
				if scenario == 'EE':
					axes[idx].set_yticks([1.4, 1.8, 2.2, 2.6])
				else:
					axes[idx].set_yticks([2.5, 2.7, 2.9, 3.1])
				axes[idx].set_xticks([0.1, 1.0, 10.0, 30.0])
				axes[idx].set_xticklabels([r'$0.1$', r'$1.0$', r'$10.0$', r'$30.0$'])
				axes[idx].set_xlabel(r'$m_{\chi} \, \mathrm{[MeV]}$', fontsize=22)
		# if scenario == 'EE':
		# 	#ax1.text(0.185, 0.284, 'Electrophilic Majorana Fermion', fontsize=14)
		# else:
		# 	#ax1.text(0.135, 0.261, 'Neutrinophilic Majorana Fermion', fontsize=14)
		#plt.suptitle('Majorana Fermion annihilating into Electrons')
		plt.savefig('Nollett{}.pdf'.format(scenario))	

