"""
contours.py

Produces smoothed contours for the Majorana case and delta chisq plots

To Note:

- Uses the {Scenario}_New.txt data files
"""

from utils import get_data
from utils import plot_distributions, plot_abundances, plot_chisq_distribution, plot_mchi_omegab_contours 
from utils import plot_joint_mchi_omegab, plot_deltachisq
from utils import get_chisq_grid, get_mass_omegab_grid, get_masses
from utils import chisq
from utils import save_results
from utils import plot_sqrtdeltachisq, plot_sqrtdeltachisqmain, plot_joint_mchi_omegab_forecast, plot_sqrtdeltachisq_forecast
from utils import plot_cts_and_deltachi

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

def run_scenario(scenario):
	filename = scenario + '.txt'
	data = get_data(filename)
	new_data = get_data(filename.replace("_New", ""))
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
	plot_cts_and_deltachi(data, new_data, scenario)
	# print('[{}] Plotted joint contours (5/7)'.format(scenario))
	# plot_deltachisq(data, scenario, zoom=False)
	# plot_deltachisq(data, scenario, zoom=True)
	# plot_sqrtdeltachisq(data, scenario)
	# plot_sqrtdeltachisqmain(data, scenario)
	# print('[{}] Plotted Delta Chi curves (6/7)'.format(scenario))
	# print('[{}] Saving results (7/7)'.format(scenario))
	# save_results(data, scenario)

if __name__ == '__main__':
	scenarios = ['EE_Neutral_Scalar',
				 'EE_Complex_Scalar', 
			     'EE_Maj',
			     'EE_Dirac',
			     'EE_Zp',
			     'Nu_Neutral_Scalar', 
			     'Nu_Complex_Scalar', 
			     'Nu_Maj',
			     'Nu_Dirac', 
			     'Nu_Zp']
	# NOTE: Uses {Scenario}_New.txt data to smooth out contours
	scenarios = ['EE_Maj_New', 'Nu_Maj_New']
	Parallel(n_jobs=4)(delayed(run_scenario)(scenario=scenario) for scenario in scenarios)