"""
abundances.py 

- plots the helium and deuterium abundance from the data file Abundances_Delta_Gravity
- Abundances_Delta_Gravity is stored in 
- outputs to pdf saved as abundances.pdf
"""

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

def dGdT(prefactor):
	return 10**(12)*(1 - prefactor)/(13.8*10**9)

if __name__ == '__main__':
	data = np.loadtxt('Abundances_Delta_Gravity', skiprows=1)
	prefactor = data[:, 0]
	OmegaB = data[:, 1]
	Yp = data[:, 8] * 4
	DoverH = data[:, 5]/data[:, 4]
	gravity = np.unique(prefactor)
	omegab = np.unique(OmegaB)
	GRAV, OMEGAB = np.meshgrid(gravity, omegab)
	YP, DH = Yp.reshape(len(gravity), -1), DoverH.reshape(len(gravity), -1)

	indices = [140, 150, 160, 170, 180]
	colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']
	linestyles = ['--',
	              (0, (5,1)),
	              (0, (3, 1, 1, 1)),
	              (0, (1, 1)),
	              '-']


	figsize= (6, 10)
	plt.figure(figsize=figsize)
	ax1 = plt.subplot(2, 1, 1)
	ax2 = plt.subplot(2, 1, 2)
	axes = [ax1, ax2]
	color = "darkgoldenrod"
	linestyle = "-"

	for index, color, linestyle in zip(indices, colors, linestyles):
		Yp_plot = YP[:, index]
		DH_plot = DH[:, index]

		ax1.plot(gravity, Yp_plot, c=color, linestyle=linestyle, alpha=0.8)
		ax2.plot(gravity, 10**5 * DH_plot, c=color, linestyle=linestyle, alpha=0.8, label=r'${:.4f}$'.format(omegab[index]))

	YpCentre = 0.245
	YpError = np.sqrt(0.003**2 + 0.00017**2)
	DHCentre = 2.569
	DHError = np.sqrt(0.027**2 + 0.036**2)
	NeffCentre = 2.89
	NeffError = 0.31

	labelsize = 22

	axtwin = ax1.twiny()
	axtwin.set_xlim((dGdT(0.8), dGdT(1.2)))
	axtwin.set_xlabel(r'$10^{12} \times \dot{G}/G_0$')

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

	axtwin = ax2.twiny()
	axtwin.set_xlim((dGdT(0.8), dGdT(1.2)))
	axtwin.xaxis.set_tick_params(labelsize=0, zorder=8)

	ax1.set_xlim(0.8, 1.2)
	ax2.set_xlim(0.8, 1.2)
	ax1.set_ylim(0.23, 0.26)
	ax2.set_ylim(2.2, 3.0)
	ax2.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])

	ax2.legend(loc='upper left', markerfirst=True, fontsize=14, title_fontsize=14, title=r'$\mathbf{\Omega_{\mathbf{b}}h^2}$', ncol=2)

	ax2.set_xlabel(r'$G_{\mathrm{BBN}}/G_0$')
	ax1.set_ylabel(r'$Y_{\mathrm{P}}$')
	ax2.set_ylabel(r'$10^5 \times \mathrm{D}/\mathrm{H}$')
	ax1.tick_params(axis='x', which='major', labelsize=0)
	plt.subplots_adjust(wspace=0, hspace=0.1)
	plt.savefig('abundances.pdf')

