"""
standard_model.py

Plots the classic abundance plot for the SM case as a function of the baryon density
"""

import numpy as np
import matplotlib.pyplot as plt
data_dir = '../../../Data/'
colors = ['#306B37', 'purple', 'darkgoldenrod', '#BF4145']

if __name__ == '__main__':
	data = np.loadtxt(data_dir + 'SBBN.txt', skiprows=1)
	OmegaB = data[:, 0]
	YP = 4 * data[:, 7]
	DH = data[:, 4]/data[:, 3]
	He3H = data[:, 6]/data[:, 3]
	Li7H = data[:, 9]/data[:, 3]

	fig = plt.figure(figsize=(5, 10))
	ax1 = fig.add_subplot(3, 1, 1)
	ax1t = ax1.twiny()
	ax2 = fig.add_subplot(3, 1, 2)
	ax3 = fig.add_subplot(3, 1, 3)
	plt.subplots_adjust(hspace=0.0)
	ax1.tick_params(axis='x', labelsize=0)
	ax2.tick_params(axis='x', labelsize=0)

	ax1.set_ylabel(r'$Y_{\mathrm{P}}$')
	ax2.set_ylabel(r'$^{3}\mathrm{He}/\mathrm{H}\quad\mathrm{D}/\mathrm{H}$')
	ax3.set_xlabel(r'$\Omega_{\mathrm{b}}h^2$')
	ax3.set_ylabel(r'$^{7}\mathrm{Li}/\mathrm{H}$')

	ax1.plot(OmegaB, YP, c=colors[0], lw=4)
	ax2.plot(OmegaB, DH, c=colors[1], lw=4)
	ax2.plot(OmegaB, He3H, c=colors[2], lw=4)
	ax3.plot(OmegaB, Li7H, c=colors[3], lw=4)

	ax2.set_yscale('log')
	ax3.set_yscale('log')

	for ax in [ax1, ax2, ax3]:
		ax.set_xlim(0.02, 0.035)
		ax.set_xticks([0.02, 0.025, 0.030, 0.035])

	ax1.set_ylim(0.230,0.260)
	ax1.set_yticks([0.235, 0.245, 0.255])
	ax2.set_ylim(3e-6,)
	ax3.set_ylim(6e-11, 3e-9)

	axis2 = ax2.axis()
	axis3 = ax3.axis()
	
	YpCentre = 0.245
	YpError = np.sqrt(0.003**2 + 0.00017**2)
	DHCentre = 2.569e-5
	DHError = np.sqrt(0.027**2 + 0.036**2) * 10**(-5)
	LiCentre = 1.58e-10
	LiError = 0.3e-10
	OmegaBCentre = 0.02236
	OmegaBError = 1.5e-4

	ax1.add_patch(plt.Rectangle(xy=(0.02, YpCentre - 2*YpError),
	                        width=0.02,
	                        height=4*YpError,
	                        alpha=0.1,
	                        color='k'))
	ax1.add_patch(plt.Rectangle(xy=(OmegaBCentre - 2*OmegaBError, 0.230),
	                        width=4*OmegaBError,
	                        height=0.03,
	                        alpha=0.25,
	                        color='#3F7BB6'))
	ax2.add_patch(plt.Rectangle(xy=(0.02, (DHCentre - 2*DHError)),
	                        width=0.02,
	                        height=4*DHError,
	                        alpha=0.1,
	                        color='k'))
	ax2.add_patch(plt.Rectangle(xy=(OmegaBCentre - 2*OmegaBError, axis2[2]),
	                        width=4*OmegaBError,
	                        height=axis2[3] - axis2[2],
	                        alpha=0.25,
	                        color='#3F7BB6'))
	ax3.add_patch(plt.Rectangle(xy=(0.02, (LiCentre - 2*LiError)),
	                        width=0.02,
	                        height=4*LiError,
	                        alpha=0.1,
	                        color='k'))
	ax3.add_patch(plt.Rectangle(xy=(OmegaBCentre - 2*OmegaBError, axis3[2]),
	                        width=4*OmegaBError,
	                        height=axis3[3] - axis3[2],
	                        alpha=0.25,
	                        color='#3F7BB6'))

	ax1t.set_xlim(5.475349, 9.5818607)
	ax1t.set_xticks([5.475349, 6.8441862, 8.2130235, 9.5818607])
	ax1t.set_xticklabels([r'$5.5$', r'$6.8$', r'$8.2$', r'$9.6$'])
	ax1t.set_xlabel(r'$10^{10} \times \eta$')

	plt.savefig('standard_model.pdf')