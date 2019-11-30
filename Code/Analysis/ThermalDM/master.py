"""
master.py

Master file to run all the base analyses. Saves plots to the subfolder Scenario/ e.g. EE_Maj/sqrt_delta_chi.pdf

To Note:

- Various functions can be uncommented to produce additional plots, but the relevant ones for producing the publication figures are uncommented
- Lists of scenarios can be defined in __main__
"""

from utils import get_data
from utils import plot_distributions, plot_abundances, plot_chisq_distribution, plot_mchi_omegab_contours 
from utils import plot_joint_mchi_omegab, plot_deltachisq
from utils import get_chisq_grid, get_mass_omegab_grid, get_masses
from utils import chisq
from utils import save_results
from utils import plot_sqrtdeltachisq, plot_sqrtdeltachisqmain, plot_joint_mchi_omegab_forecast, plot_sqrtdeltachisq_forecast

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

def run_scenario(scenario):
	filename = scenario + '.txt'
	data = get_data(filename)
	print('Loaded data from: {}\nLength: {}'.format(filename, len(data['mass'])))

	# plot_distributions(data, scenario)
	# print('[{}] Plotted Distrbutions (1/7)'.format(scenario))
	# plot_abundances(data, scenario)
	# print('[{}] Plotted Abundances (2/7)'.format(scenario))
	# plot_chisq_distribution(data, scenario)
	# print('[{}] Plotted Chi Squared Distrbutions (3/7)'.format(scenario))
	# plot_mchi_omegab_contours(data, scenario, 'BBN')
	# plot_mchi_omegab_contours(data, scenario, 'CMB')
	# plot_mchi_omegab_contours(data, scenario, 'BBN+CMB')
	# print('[{}] Plotted Omegab vs Mchi Contours (4/7)'.format(scenario))
	# plot_joint_mchi_omegab(data, scenario)
	# print('[{}] Plotted joint contours (5/7)'.format(scenario))
	# plot_deltachisq(data, scenario, zoom=False)
	# plot_deltachisq(data, scenario, zoom=True)
	plot_sqrtdeltachisq(data, scenario)
	plot_sqrtdeltachisqmain(data, scenario)
	# print('[{}] Plotted Delta Chi curves (6/7)'.format(scenario))
	# print('[{}] Saving results (7/7)'.format(scenario))
	#save_results(data, scenario, save=False)

if __name__ == '__main__':
	scenarios = ['Nu_Maj', 
                 'Nu_Dirac',
                 'Nu_Neutral_Scalar',
                 'Nu_Complex_Scalar',
                 'Nu_Zp', 
                 'EE_Maj',
                 'EE_Dirac',
                 'EE_Neutral_Scalar',
                 'EE_Complex_Scalar',
                 'EE_Zp']
	scenarios = ['EE_Neutral_Scalar']
	Parallel(n_jobs=4)(delayed(run_scenario)(scenario=scenario) for scenario in scenarios)
	

