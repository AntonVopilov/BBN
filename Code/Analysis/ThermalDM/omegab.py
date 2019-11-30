"""
omegab.py

Produces the prior on omegab plot

To Note:

- Outputs to pdf "omegabposteriors.pdf"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
	central_values = [0.02223,  
					  0.02251, 
					  0.02251, 
					  0.02236, 
					  0.02205,
					  0.02225]

	sigmas = [0.00028,  
			  0.00027, 
			  0.00024, 
			  0.00027, 
			  0.00027,
			  3*0.00022]

	colors = ['darkgoldenrod',
			  '#BF4145', 
			  '#306B37',
			  '#3F7BB6',
			  'indigo',
			  'black']

	colors = ['darkgoldenrod',
			  'indigo',
			  '#306B37',
			  '#3F7BB6',
			  '#BF4145',
			  'black']

	labels = ['Planck',
			  'Planck+R16+BAO',
			  'Planck+R16+JLA',
			  'Planck+R16+WL',
			  'Planck+R16+Lensing',
			  r'$\textrm{\textbf{Our Range}}$']

	fig = plt.figure(figsize=(10, 6))
	plt.xlabel(r'$\Omega_{\mathrm{b}}h^2$')
	plt.ylabel(r'$\mathrm{Probability}\,\mathrm{Density}$')
	omegab_grid = np.linspace(0.020, 0.025, 1000)
	for c, s, col, l in zip(central_values, sigmas, colors, labels):
		maximum = np.max(norm.pdf(omegab_grid, loc=c, scale=s))
		maximum = 1
		plt.plot(omegab_grid, norm.pdf(omegab_grid, loc=c, scale=s)/maximum,
			color=col,
			label=l,
			linestyle='-',
			alpha=0.8)
		plt.fill(omegab_grid, norm.pdf(omegab_grid, loc=c, scale=s)/maximum,
			color=col,
			alpha=0.05,)
	plt.gca().set_yticks([])
	plt.gca().set_yticklabels([])
	plt.legend(markerfirst=False, fontsize=15)
	plt.savefig('omegabposteriors.pdf')


