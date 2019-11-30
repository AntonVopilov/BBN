"""
br.py

Plots the value of Neff for a range of branching ratios in electrons and neutrinos.

To Note:

- check where the abundance data is stored. If it is in a subfolder of DarkBBN/Data, this will need to be added to get_data
- outputs a plot '{case}_BR.pdf'
"""

if __name__ == '__main__':
	from utils import get_data
	import matplotlib.pyplot as plt
	import numpy as np
	from scipy.interpolate import interp1d
	plt.rcParams['figure.subplot.hspace'] = 0.07

	case = 'Zp'

	scenarios = {'Abundances_Case=EE_Stat=BE_gDM=3._Sigmav=1._BR=M6': r'$e:\nu = 10^{6}:1$',
					 'Abundances_Case=EE_Stat=BE_gDM=3._Sigmav=1._BR=M5': r'$e:\nu = 10^{5}:1$',
					 'Abundances_Case=EE_Stat=BE_gDM=3._Sigmav=1._BR=M4': r'$e:\nu = 10^{4}:1$',
					 'Abundances_Case=EE_Stat=BE_gDM=3._Sigmav=1._BR=M3': r'$e:\nu = 10^{3}:1$',
					 'Abundances_Case=EE_Stat=BE_gDM=3._Sigmav=1._BR=M2': r'$e:\nu = 10^{2}:1$',
					 'Abundances_Case=EE_Stat=BE_gDM=3._Sigmav=1._BR=M1': r'$e:\nu = 10^{1}:1$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=0.5': r'$e:\nu = 1:1$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=M1': r'$e:\nu = 1:10^{1}$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=M2': r'$e:\nu = 1:10^{2}$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=M3': r'$e:\nu = 1:10^{3}$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=M4': r'$e:\nu = 1:10^{4}$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=M5': r'$e:\nu = 1:10^{5}$',
					 'Abundances_Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=M6': r'$e:\nu = 1:10^{6}$',
				}

	fig = plt.figure(figsize=(4*4, 4*5))

	nu_boundary_data = get_data('Nu_{}.txt'.format(case))
	ee_boundary_data = get_data('EE_{}.txt'.format(case))
	nu_masses = np.unique(nu_boundary_data['mass'])
	ee_masses = np.unique(ee_boundary_data['mass'])
	nu_neff = nu_boundary_data['Neff'].reshape(len(nu_masses), -1)[:, 0]
	ee_neff = ee_boundary_data['Neff'].reshape(len(ee_masses), -1)[:, 0]
	nu_interpfn = interp1d(nu_masses, nu_neff, kind='cubic')
	ee_interpfn = interp1d(ee_masses, ee_neff, kind='cubic')


	for idx, scenario in enumerate(scenarios.keys()):
		data = get_data(scenario + '.txt')
		masses = data['mass'].reshape(len(np.unique(data['mass'])), -1)[:, 0]
		Neff = data['Neff'].reshape(len(np.unique(data['mass'])), -1)[:, 0]
		mass_range = np.geomspace(np.min(masses), np.max(masses), 1000)

		interpfn = interp1d(masses, Neff, kind='cubic')
		if idx != 12:
			ax = plt.subplot(5, 3, idx + 1)
		else:
			ax = plt.subplot(5, 3, 14)
		# ax.plot(mass_range, interpfn(mass_range), c='#3F7BB6', lw=2.5)
		# ax.plot(mass_range, nu_interpfn(mass_range), c='#BF4145', lw=2.5, ls=(0, (5, 1)))
		# ax.plot(mass_range, ee_interpfn(mass_range), c='#BF4145', lw=2.5, ls=(0, (5, 1)))
		ax.plot(masses, Neff, c='#3F7BB6', lw=2.5)
		ax.plot(mass_range, nu_interpfn(mass_range), c='#BF4145', lw=2.5, ls=(0, (5, 1)))
		ax.plot(mass_range, ee_interpfn(mass_range), c='#BF4145', lw=2.5, ls=(0, (5, 1)))
		if idx in [9, 11, 12]:	
			ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
		else:
			ax.set_xticklabels([])
		if idx in [0, 3, 6, 9, 12]:
			ax.set_ylabel(r'$N_{\mathrm{eff}}$')
		else:
			ax.set_yticklabels([])
		ax.set_xlim(0.1, 8.0)
		ax.set_ylim(2.0, 4.0)
		ax.text(3.5, 2.1, scenarios[scenario], fontsize=20)
	plt.suptitle('Vector Boson', fontsize=32)
	plt.subplots_adjust(top=0.95)
	plt.savefig('{}_BR.pdf'.format(case))


	

