"""
bounds.py

- plots the chi squared curves and contours for BBN and BBN+OmegaB scenarios from 
  data file Abundances_Delta_Gravity
- Abundances_Delta_Gravity should be saved in same location
- outputs to pdf saved as DeltaGCtr.pdf
- also prints the interpolated results for the 1 and 2 sigma exclusions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib import ticker
import matplotlib
plt.rcParams['axes.linewidth'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

def dGdT(prefactor):
	return 10**(12)*(1 - prefactor)/(13.8*10**9)

if __name__ == '__main__':
	data = np.loadtxt('Abundances_Delta_Gravity', skiprows=1)
	prefactor = data[:, 0]
	OmegaB = data[:, 1]
	Yp = data[:, 8] * 4
	DoverH = data[:, 5]/data[:, 4]

	YpCentre = 0.245

	YpError = 0.003
	YpErrorTh = 0.00017
	DHCentre = 2.569 * 10**(-5)
	DHError = (0.027) * 10**(-5)
	DHErrorTh = 0.036 * 10**(-5)
	OmegabCentre = 0.02236
	OmegabError = 0.00015*2

	chisqBBN = np.power(Yp - YpCentre, 2)/(YpError**2 + YpErrorTh**2) + np.power(DoverH - DHCentre, 2)/(DHError**2 + DHErrorTh**2)
	chisqBBNOM = chisqBBN + np.power(OmegaB - OmegabCentre, 2)/(OmegabError**2)

	gravity = np.unique(prefactor)
	omegab = np.unique(OmegaB)
	GRAV, OMEGAB = np.meshgrid(gravity, omegab)
	CHISQBBN = chisqBBN.reshape(len(gravity), -1).T
	CHISQBBNOM = chisqBBNOM.reshape(len(gravity), -1).T

	CHISQBBN = CHISQBBN - np.min(CHISQBBN)
	CHISQBBNOM = CHISQBBNOM - np.min(CHISQBBNOM)

	CHISQ_MARG_BBN = []
	CHISQ_MARG_BBNOM = []

	for idx in range(len(gravity)):
		CHISQ_MARG_BBN.append(np.min(CHISQBBN[:, idx]))
		CHISQ_MARG_BBNOM.append(np.min(CHISQBBNOM[:, idx]))

	fig = plt.figure(figsize=(6, 10))

	ax = plt.subplot(2, 1, 1)
	axtwin = ax.twiny()
	zorders = [-1, -1, -1, -1, 2, 3, 0, 1]


	confidence_levels = [0.0, 6.18, 28.74]
	vmin = 1.5
	vmax = 7.0
	ct = ax.contourf(GRAV, OMEGAB, CHISQBBN, 
	                 levels=confidence_levels,
	                 cmap=plt.get_cmap('Purples_r'),
	                 vmin=vmin,
	                 vmax=vmax,
	                 zorder=zorders[0])
	ct = ax.contour(GRAV, OMEGAB, CHISQBBN, 
	                 levels=confidence_levels,
	                 linewidths=0.5,
	                 colors=['indigo'],
	                 vmin=vmin,
	                 vmax=vmax,
	                 zorder=zorders[1])
	ct = ax.contourf(GRAV, OMEGAB, CHISQBBNOM,
	                 levels=confidence_levels,
	                 cmap=plt.get_cmap('Greens_r'),
	                 vmin=vmin,
	                 vmax=vmax,
	                 zorder=zorders[2])
	ct = ax.contour(GRAV, OMEGAB, CHISQBBNOM, 
	                 levels=confidence_levels,
	                 linewidths=0.5,
	                 colors=['#306B37'],
	                 vmin=vmin,
	                 vmax=vmax,
	                 zorder=zorders[3])

	proxy = [plt.Rectangle((0,0),1,1,fc='indigo',alpha=0.8),
	         plt.Rectangle((0,0),1,1,fc='#306B37',alpha=0.8)]


	ax.legend(proxy, [r"BBN", r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$'], 
	    fontsize=14, 
	    loc='upper left',
	    markerfirst=True)

	#ax.set_xlabel(r'$G_{\mathrm{BBN}}/G_{0}$', fontsize=22)
	ax.set_ylabel(r'$\Omega_{\mathrm{b}} h^2$', fontsize=22)
	ax.set_xlim((0.8, 1.2))
	ax.set_ylim((0.018, 0.025))
	ax.set_yticks([0.018, 0.020, 0.022, 0.024])
	axtwin.set_xlim((dGdT(0.8), dGdT(1.2)))
	axtwin.set_xlabel(r'$10^{12} \times \dot{G}/G_0$')
	#axtwinx.set_xticks([10.0, 5.0, 0.0, -5.0, -10.0])
	#axtwin.set_xticklabels([r'$10.0$', r'$5.0$', r'$0.0$', r'$5.0$', r'$10.0$'])
	ax.xaxis.set_tick_params(labelsize=0, zorder=8)
	ax.yaxis.set_tick_params(labelsize=20, zorder=8)

	ax = plt.subplot(2, 1, 2)

	axtwin = ax.twiny()
	axtwin.set_xlim((dGdT(0.8), dGdT(1.2)))

	prefactor_array = np.linspace(0.5, 1.5, 1000)
	interpfn_bbn = interp1d(gravity, CHISQ_MARG_BBN, kind='cubic', fill_value='extrapolate')
	interpfn_bbnom = interp1d(gravity, CHISQ_MARG_BBNOM, kind='cubic', fill_value='extrapolate')

	ax.plot(prefactor_array, np.sqrt(np.abs(interpfn_bbn(prefactor_array))),lw=2.8,label='BBN',c='indigo',ls=(0, (1, 1)))
	ax.plot(prefactor_array, np.sqrt(np.abs(interpfn_bbnom(prefactor_array))),lw=2.8,label=r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$',c='#306B37',ls=(0, (3, 1, 1, 1)))

	MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
	col = 'k'
	confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
	ct = ax.contour(MC, CSQ, CSQ, 
	                levels=confidence_levels,
	                linewidths=1.0,
	                alpha=0.4,
	                colors=['k'])
	levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
	for idx, level in enumerate(levels):
	    ax.text(0.81, (idx + 1) + 0.075, level, color='k')
	#plt.text(0.115, 0.22, descriptions[scenario], color='k', fontsize=11)
	#plt.title(descriptions[scenario], color='k', fontsize=20)
	ax.set_xlabel(r'$G_{\mathrm{BBN}}/G_0$', fontsize=22)
	ax.set_ylabel(r'$\sqrt{\Delta \chi^2}$', labelpad=15, fontsize=22)
	ax.set_xlim(0.8, 1.2)
	ax.set_ylim(0, 6.0)
	ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
	ax.set_yticklabels([r'$0.0$', r'$1.0$', r'$2.0$', r'$3.0$', r'$4.0$', r'$5.0$', r'$6.0$', r'$7.0$'])
	ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
	axtwin.xaxis.set_tick_params(labelsize=0, zorder=8)
	ax.xaxis.set_tick_params(labelsize=20)
	ax.yaxis.set_tick_params(labelsize=20)
	ax.legend(loc='upper center', markerfirst=True, fontsize=14, handlelength=3)
	plt.subplots_adjust(hspace=0.1)

	plt.savefig('DeltaGCtr.pdf')
	BBNcentre = UnivariateSpline(gravity, np.array(CHISQ_MARG_BBN) - 0.0, s=0)
	BBNOMcentre = UnivariateSpline(gravity, np.array(CHISQ_MARG_BBNOM) - 0.0, s=0)
	BBNinterp = UnivariateSpline(gravity, np.array(CHISQ_MARG_BBN) - 2.0, s=0)
	BBNOMinterp = UnivariateSpline(gravity, np.array(CHISQ_MARG_BBNOM) - 2.0, s=0)
	# with open('results.txt', 'w') as f:
	# 	print('BBN G/G0:\t\t\t', BBNinterp.roots(), file=f)
	# 	print('BBN + OmegaB G/G0:\t\t', BBNOMinterp.roots(), file=f)
	# 	print('BBN 10^12 x dG/dt/G0:\t\t', dGdT(BBNinterp.roots()), file=f)
	# 	print('BBN + OmegaB 10^12 x dG/dt/G0:\t', dGdT(BBNOMinterp.roots()), file=f)
	print('BBN G/G0 Central Value:\t\t\t', BBNcentre.roots())
	print('BBN + OmegaB G/G0 Central Value:\t', BBNOMcentre.roots())
	print('BBN G/G0:\t\t\t\t', BBNinterp.roots() - BBNcentre.roots())
	print('BBN + OmegaB G/G0:\t\t\t', BBNOMinterp.roots() - BBNOMcentre.roots())
	print('BBN 10^12 x dG/dt/G0 Centre:\t\t', dGdT(BBNcentre.roots()))
	print('BBN + OmegaB 10^12 x dG/dt/G0 Centre:\t', dGdT(BBNOMcentre.roots()))
	print('BBN 10^12 x dG/dt/G0:\t\t\t', dGdT(BBNinterp.roots()) - dGdT(BBNcentre.roots()))
	print('BBN + OmegaB 10^12 x dG/dt/G0:\t\t', dGdT(BBNOMinterp.roots()) - dGdT(BBNOMcentre.roots()))
