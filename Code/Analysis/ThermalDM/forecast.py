"""
forecast.py

Generates sqrt(delta chisq) plots for CMB forecasts (not used to generate plots in the paper)

To Note:

- IMPORTANT: Requires new data files (EE_Maj_Sigma.txt and Nu_Maj_New.txt for the Majorana case)
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
	plot_abundances(data, scenario, forecast=True)
	# plot_joint_mchi_omegab_forecast(data, scenario)
	plot_sqrtdeltachisq_forecast(data, scenario)

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
	scenarios = ['EE_Maj_Sigma']
	Parallel(n_jobs=-1)(delayed(run_scenario)(scenario=scenario) for scenario in scenarios)