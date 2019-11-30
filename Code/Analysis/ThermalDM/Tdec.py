"""
Plotting tool for the neutrino decoupling temperature

To Note:

- Check the data location for the two relevant files TdecAbundances.txt and TdecChisq.txt
- Outputs to pdf "Tdec.pdf"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib
from utils import get_data, get_chisq_marg_interpolation
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
	data_dir = '../../../Data/'
	abundanceData = np.loadtxt(data_dir + 'TdecAbundances.txt', skiprows=1)
	chisqData = get_data('TdecChisq.txt')

	data = {}
	data['Tdec'] = abundanceData[:, 0]
	data['Neff'] = abundanceData[:, 2]
	data['Yp'] = 4 * abundanceData[:, 8]
	data['D/H'] = 10**5 * abundanceData[:, 5]/abundanceData[:, 4]

	NeffInterp = interp1d(data['Tdec'], data['Neff'], kind='cubic', fill_value='extrapolate')
	YpInterp = interp1d(data['Tdec'], data['Yp'], kind='cubic', fill_value='extrapolate')
	DHInterp = interp1d(data['Tdec'], data['D/H'], kind='cubic', fill_value='extrapolate')

	figsize = (12, 5)
	plt.figure(figsize=figsize)
	ax1 = plt.subplot(3, 2, 1)
	ax2 = plt.subplot(3, 2, 3)
	ax3 = plt.subplot(3, 2, 5)
	ax4 = plt.subplot(1, 2, 2)
	colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

	TdecInterp = np.geomspace(0.1, 5, 1000)

	ax1.plot(TdecInterp, NeffInterp(TdecInterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[2],
					linestyle='-')
	ax2.plot(TdecInterp, YpInterp(TdecInterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[2],
					linestyle='-')
	ax3.plot(TdecInterp, DHInterp(TdecInterp),
					linewidth=2.5,
					alpha=0.8,
					c=colors[2],
					linestyle='-')

	NeffCentre = 2.89
	NeffError = 0.31
	YpCentre = 0.245
	YpError = np.sqrt(0.003**2 + 0.00017**2)
	DHCentre = 2.569
	DHError = np.sqrt(0.027**2 + 0.036**2)

	labelsize = 18

	ax1.add_patch(plt.Rectangle(xy=(0.1, (NeffCentre - 2*NeffError)),
	                        width=(30.0 - 0.1),
	                        height=4*NeffError,
	                        alpha=0.1,
	                        color='k'))

	ax2.add_patch(plt.Rectangle(xy=(0.1, YpCentre - 2*YpError),
	                        width=(30.0 - 0.1),
	                        height=4*YpError,
	                        alpha=0.1,
	                        color='k'))

	ax3.add_patch(plt.Rectangle(xy=(0.1, (DHCentre - 2*DHError)),
	                        width=(30.0 - 0.1),
	                        height=4*DHError,
	                        alpha=0.1,
	                        color='k'))

	bbninterpfn = get_chisq_marg_interpolation(chisqData, type='BBN')
	bbnominterpfn = get_chisq_marg_interpolation(chisqData, type='BBN+Omegab')
	cmbinterpfn = get_chisq_marg_interpolation(chisqData, type='CMB')
	bbnandcmbinterpfn = get_chisq_marg_interpolation(chisqData, type='BBN+CMB')

	ax4.plot(TdecInterp, np.sqrt(np.abs(bbnandcmbinterpfn(TdecInterp))),lw=2.8,label='BBN+Planck',c='#306B37',ls=(0, (1, 1)))
	ax4.plot(TdecInterp, np.sqrt(np.abs(bbnominterpfn(TdecInterp))),lw=2.8,label=r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$',c='purple',ls=(0, (3, 1, 1, 1)))
	ax4.plot(TdecInterp, np.sqrt(np.abs(cmbinterpfn(TdecInterp))),lw=2.5,label='Planck',c='#BF4145',ls=(0, (5, 2)))
	ax4.plot(TdecInterp, np.sqrt(np.abs(bbninterpfn(TdecInterp))),lw=2.5,label='BBN',c='#3F7BB6',ls='-')

	MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
	col = 'k'
	confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
	ct = ax4.contour(MC, CSQ, CSQ, 
	                levels=confidence_levels,
	                linewidths=1.0,
	                alpha=0.4,
	                colors=['k'])
	levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
	for idx, level in enumerate(levels):
	    ax4.text(0.1082, (idx + 1) + 0.075, level, color='k')

	ax1.axvline(1.9, color='k', linestyle='--', alpha=0.8)
	ax1.text(2.0, 8.7, 'SM', fontsize=14)
	ax2.axvline(1.9, color='k', linestyle='--', alpha=0.8)
	ax3.axvline(1.9, color='k', linestyle='--', alpha=0.8)
	ax1.tick_params(axis='x', which='major', labelsize=0)

	ax1.set_xscale('log')
	ax1.set_ylabel(r'$N_{\mathrm{eff}}$', fontsize=labelsize)

	ax2.set_xscale('log')
	ax2.set_ylabel(r'$Y_{\mathrm{P}}$', fontsize=labelsize)
	ax2.tick_params(axis='x', which='major', labelsize=0)

	ax3.set_xscale('log')
	ax3.set_xlabel(r'$T_{\nu}^{\mathrm{dec}} \, \mathrm{[MeV]}$', fontsize=labelsize)
	ax3.set_ylabel(r'$10^5 \times \mathrm{D}/\mathrm{H}|_{\mathrm{P}}$', fontsize=labelsize)

	ax4.set_xscale('log')
	ax4.set_xlabel(r'$T_{\nu}^{\mathrm{dec}} \, \mathrm{[MeV]}$', fontsize=labelsize)
	ax4.set_ylabel(r'$\sqrt{\Delta \chi^2}$', fontsize=labelsize)

	ax1.set_yticks([2, 4, 6, 8, 10])
	ax2.set_yticks([0.24, 0.25, 0.26])
	ax3.set_yticks([2.5, 3.5, 4.5])
	ax3.set_xticks([0.1, 1.0, 5.0])
	ax3.set_xticklabels([r'$0.1$', r'$1.0$', r'$5.0$'])
	ax4.set_xticks([0.1, 1.0, 5.0])
	ax4.set_xticklabels([r'$0.1$', r'$1.0$', r'$5.0$'])
	ax4.legend(fontsize=13, markerfirst=False, handlelength = 3, loc='upper right')

	ax1.set_ylim(2.0, 10.0)
	ax1.set_xlim(0.1, 5.0)
	ax2.set_xlim(0.1, 5.0)
	ax2.set_ylim(0.235, 0.265)
	ax3.set_xlim(0.1, 5.0)
	ax3.set_ylim(2.0, )
	ax4.set_xlim(0.1, 5.0)
	ax4.set_ylim(0.0, 7.0)

	plt.subplots_adjust(hspace=0.05)
		
	plt.savefig('Tdec.pdf')