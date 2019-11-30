"""
getresults.py

Outputs the bounds to the terminal for the various scenario e.g. BR cases

To Note:

- IMPORTANT: In the BR case, the data files are not stored in DarkBBN/Data, additional subfolder needs to be specified in get_data or in the list of scenarios
"""

from utils import get_data
from utils import plot_distributions, plot_abundances, plot_chisq_distribution, plot_mchi_omegab_contours 
from utils import plot_joint_mchi_omegab, plot_deltachisq
from utils import get_chisq_grid, get_mass_omegab_grid, get_masses
from utils import chisq
from utils import save_results
from utils import plot_sqrtdeltachisq, plot_sqrtdeltachisqmain

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

def run_scenario(scenario):
	filename = scenario + ".txt"
	data = get_data(filename)
	print('Loaded data from: {}\nLength: {}'.format(filename, len(data['mass'])))

	#plot_sqrtdeltachisq(data, scenario)
	save_results(data, scenario, save=False)

if __name__ == '__main__':

	# IMPORTANT: In the BR case, the files are not stored in DarkBBN/Data, an additional subfolder needs to be added to either get_data or the scenarios below
	scenarios = ["Nu_Maj",
				 "Nu_Dirac",
				 "Nu_Neutral_Scalar",
				 "Nu_Complex_Scalar",
				 "Nu_Zp",
				 "EE_Maj",
				 "EE_Dirac",
				 "EE_Neutral_Scalar",
				 "EE_Complex_Scalar",
				 "EE_Zp",
				 ]
	#Parallel(n_jobs=-1)(delayed(run_scenario)(scenario=scenario) for scenario in scenarios)
	for scenario in scenarios:
		run_scenario(scenario)