"""
cmbforecast.py

Generates the CMB forecast comparisonss for Planck, Simons and CMB-S4.

- If data is not stored in DarkBBN/Data this needs to be modified in get_data
- Information on how to run and filename structure is in __main__ section
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline
from datetime import datetime
plt.rcParams['axes.linewidth'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True



def get_data(filename, data_dir='../../../Data/'):
    """
    Loads data file from data_dir/filename
    """
    data = np.loadtxt(data_dir + filename, skiprows=1)
    mass = data[:, 0]
    OmegaB = data[:, 1]
    Neff = data[:, 2]
    H = data[:, 4]
    D = data[:, 5]
    He = data[:, 8]
    DoverH = D / H
    Yp = 4 * He
    Li = data[:, 10]
    HeT = data[:, 7] 
    return {'mass': mass, 
            'OmegaB': OmegaB, 
            'Neff': Neff, 
            'H': H, 
            'D': D, 
            'He': He, 
            'D/H': DoverH, 
            'Yp': Yp,
            'Li/H': Li/H,
            '3He/H': HeT/H}

def plot_joint_mchi_omegab(planck_data, forecast_data, scenario):
    """
    Plots confidence intervals for different scenarios with planck data and forecast data

    Change figure size here
    """
    confidence_levels = [0.0, 6.18, 28.74]
    descriptions = {'EE_Neutral_Scalar': 'Electrophilic Neutral Scalar',
                    'EE_Complex_Scalar': 'Electrophilic Complex Scalar',
                    'EE_Maj': 'Electrophilic Majorana Fermion',
                    'EE_Dirac': 'Electrophilic Dirac Fermion',
                    'EE_Zp': 'Electrophilic Vector Boson',
                    'Nu_Neutral_Scalar': 'Neutrinophilic Neutral Scalar',
                    'Nu_Complex_Scalar': 'Neutrinophilic Complex Scalar',
                    'Nu_Maj': 'Neutrinophilic Majorana Fermion',
                    'Nu_Dirac': 'Neutrinophilic Dirac Fermion',
                    'Nu_Zp': 'Neutrinophilic Vector Boson'}

    save_names = {'EE_Neutral_Scalar': 'EE_Neutral_Scalar',
                    'EE_Complex_Scalar': 'EE_Complex_Scalar',
                    'EE_Maj': 'EE_Maj',
                    'EE_Dirac': 'EE_Dirac',
                    'EE_Zp': 'EE_Zp',
                    'Nu_Neutral_Scalar': 'Nu_Neutral_Scalar',
                    'Nu_Complex_Scalar': 'Nu_Complex_Scalar',
                    'Nu_Maj': 'Nu_Maj',
                    'Nu_Dirac': 'Nu_Dirac',
                    'Nu_Zp': 'Nu_Zp'}

    PMASS, POMEGAB = get_mass_omegab_grid(planck_data)
    FMASS, FOMEGAB = get_mass_omegab_grid(forecast_data)
    CHISQP = get_chisq_grid(planck_data, type='Planck') 
    CHISQP = CHISQP - np.min(CHISQP)
    CHISQSO = get_chisq_grid(forecast_data, type='SO') 
    CHISQSO = CHISQSO - np.min(CHISQSO)
    CHISQSIV = get_chisq_grid(forecast_data, type='CMBS4') 
    CHISQSIV = CHISQSIV - np.min(CHISQSIV)

    plt.figure(figsize=(6,6))
    #plt.suptitle('Combining Measurements')
    ax = plt.subplot(1,1,1)
    ax.set_xlim(0.1,11)
    if 'nu' in scenario.lower():
        ax.set_xlim((5.0, 30.0))
        ax.set_ylim(0.022, 0.023)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.set_yticks([0.022, 0.0222, 0.0224, 0.0226, 0.0228, 0.023])
        #ax.text(5.8, 0.02294, descriptions[scenario], color='k', fontsize=12)
        plt.title(r'$\mathrm{Neutrinophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$', fontsize=22)
    else:
        ax.set_xlim(2.0, 30.0)
        ax.set_ylim(0.0216, 0.0228)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.set_yticks([0.0216, 0.0218, 0.022, 0.0222, 0.0224, 0.0226, 0.0228])
        plt.title(r'$\mathrm{Electrophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$', fontsize=22)
        #ax.text(2.8, 0.02165, descriptions[scenario], color='k', fontsize=12)
    stddev = np.array([0., 0.682689492137086, 0.954499736103642])
    confidence_levels = []
    for j in range(len(stddev)):
        confidence_levels.append(chi2.ppf(stddev[j], 2))

    planck_cmap = 'Greys_r'
    so_cmap = 'Blues_r'
    s4_cmap = ['firebrick', 'lightcoral']
    planck_vmin, planck_vmax = -3.0, 7.5
    so_vmin, so_vmax = 1.0, 7.0
    s4_vmin, s4_vmax = 1.0, 7.0
    colors = {'Planck': 'dimgray',
              'SO': 'royalblue',
              'S4': '#BF4145'}
    linewidths = 1.8
    alpha = 0.5

    # planck_cmap = ['#ED920F', '#F3BE82']
    # so_cmap = 'Blues_r'
    # s4_cmap = ['firebrick', 'lightcoral']
    # planck_vmin, planck_vmax = -3.0, 7.5
    # so_vmin, so_vmax = 1.0, 7.0
    # s4_vmin, s4_vmax = -3.0, 7.5
    # colors = {'Planck': '#F3BE82',
    #           'SO': '#B87294',
    #           'S4': 'dimgray'}
    # linewidths = 1.8
    # alpha = 0.5

    # planck_cmap = ['teal', 'powderblue']
    # so_cmap = ['rebeccapurple', 'plum']
    # s4_cmap = ['goldenrod', 'palegoldenrod']
    # planck_vmin, planck_vmax = -5.0, 7.0
    # so_vmin, so_vmax = 1.0, 7.0
    # s4_vmin, s4_vmax = 1.0, 7.0
    # colors = {'Planck': 'teal',
    #           'SO': 'rebeccapurple',
    #           'S4': 'khaki'}
    # linewidths = 1.2
    # alpha = 0.5


    if type(planck_cmap) != type([0, 1]):
        ct = ax.contourf(PMASS, POMEGAB, CHISQP, 
                         levels=confidence_levels,
                         cmap=planck_cmap,
                         vmin=planck_vmin,
                         vmax=planck_vmax,
                         interpolation='cubic',
                         zorder=-1)
    else:
        ct = ax.contourf(PMASS, POMEGAB, CHISQP, 
                         levels=confidence_levels,
                         colors=planck_cmap,
                         vmin=planck_vmin,
                         vmax=planck_vmax,
                         interpolation='cubic',
                         zorder=-1)
    ct = ax.contour(PMASS, POMEGAB, CHISQP, 
                     levels=confidence_levels,
                     linewidths=linewidths,
                     colors=colors['Planck'],
                     alpha=alpha,
                     interpolation='cubic',
                     zorder=-1)
    if type(so_cmap) != type([0, 1]):
        ct = ax.contourf(FMASS, FOMEGAB, CHISQSO,
                         levels=confidence_levels,
                         cmap=so_cmap,
                         vmin=so_vmin,
                         vmax=so_vmax,
                         zorder=0,
                         interpolation='cubic')
    else:
        ct = ax.contourf(FMASS, FOMEGAB, CHISQSO,
                         levels=confidence_levels,
                         colors=so_cmap,
                         vmin=so_vmin,
                         vmax=so_vmax,
                         interpolation='cubic',
                         zorder=0)
    ct = ax.contour(FMASS, FOMEGAB, CHISQSO, 
                     levels=confidence_levels,
                     linewidths=linewidths,
                     colors=colors['SO'],
                     alpha=alpha,
                     interpolation='cubic',
                     zorder=0)
    if type(s4_cmap) != type([0, 1]):
        ct = ax.contourf(FMASS, FOMEGAB, CHISQSIV, 
                         levels=confidence_levels,
                         cmap=s4_cmap,
                         vmin=s4_vmin,
                         vmax=s4_vmax,
                         interpolation='cubic',
                         zorder=1)
    else:
        ct = ax.contourf(FMASS, FOMEGAB, CHISQSIV, 
                         levels=confidence_levels,
                         colors=s4_cmap,
                         vmin=s4_vmin,
                         vmax=s4_vmax,
                         interpolation='cubic',
                         zorder=1)
    ct = ax.contour(FMASS, FOMEGAB, CHISQSIV, 
                     levels=confidence_levels,
                     linewidths=linewidths,
                     colors=colors['S4'],
                     alpha=alpha,
                     interpolation='cubic',
                     zorder=1)
    
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$', fontsize=22)
    ax.set_ylabel(r'$\Omega_{\mathrm{b}} h^2$', fontsize=22)
    ax.xaxis.set_tick_params(labelsize=20, zorder=8)
    ax.yaxis.set_tick_params(labelsize=20, zorder=8)

    if 'nu' in scenario.lower():
        proxy = [plt.Rectangle((0,0),1,1,fc='#3F7BB6',alpha=1.0),
                plt.Rectangle((0,0),1,1,fc='#BF4145',alpha=1.0),
             plt.Rectangle((0,0),1,1,fc='darkgrey',alpha=1.0), 
             ]
        ax.legend(proxy, [r"Simons Obs.", r"CMB-S4", r"Planck"], markerfirst=False, fontsize=16, loc='upper right')
    else:
        proxy = [plt.Rectangle((0,0),1,1,fc='darkgrey',alpha=1.0),
             plt.Rectangle((0,0),1,1,fc='#BF4145',alpha=1.0),
             plt.Rectangle((0,0),1,1,fc='#3F7BB6',alpha=1.0),
             ]
        ax.legend(proxy, [r"Planck", r"CMB-S4", r"Simons Obs.",], markerfirst=False, fontsize=16, loc='lower right')
    plt.savefig(save_names[scenario] + '/{}_cmb_exclusion.pdf'.format(save_names[scenario]))

def get_mass_omegab_grid(data):
    """
    Generates mchi, omegab 2d meshgrid
    """
    masses = np.unique(data['mass'])
    omegabs = np.unique(data['OmegaB'])
    MASS, OMEGAB = np.meshgrid(masses, omegabs)
    OMEGABDAT = data['OmegaB'].reshape(len(masses), -1).T
    return MASS, OMEGAB

def get_chisq_grid(data, type, forecast=False, errors=None):
    """
    Gets chisq 2d meshgrid for given type of forecast (i.e. Planck, SO, CMB-S4)
    """
    masses = np.unique(data['mass'])
    omegabs = np.unique(data['OmegaB'])
    MASS, OMEGAB = np.meshgrid(masses, omegabs)
    OMEGABDAT = data['OmegaB'].reshape(len(masses), -1).T
    YP = data['Yp'].reshape(len(masses), -1).T
    DH = data['D/H'].reshape(len(masses), -1).T
    NEFF = data['Neff'].reshape(len(masses), -1).T
    return chisq(YP, OMEGABDAT, NEFF, type)

def get_masses(data):
    """
    Returns 1d array of masses
    """
    return np.unique(data['mass'])

def chisqPlanck(OmegaB, Neff, Yp):
    """
    Calculates Planck chisq given abundances and omegab/mchi meshgrid
    """
    OmegaBCentre = 0.02225
    NeffCentre = 2.89
    YpCentre = 0.246
    dO = OmegaB - OmegaBCentre
    dN = Neff - NeffCentre
    dY = Yp - YpCentre
    rho12 = 0.40
    rho13 = 0.18
    rho23 = -0.69
    Delta1 = 0.00022
    Delta2 = 0.31
    Delta3 = 0.018
    SigmaCMB = np.array([[Delta1**2, Delta1*Delta2*rho12, Delta1*Delta3*rho13], 
                         [Delta1*Delta2*rho12, Delta2**2, Delta2*Delta3*rho23], 
                         [Delta1*Delta3*rho13, Delta2*Delta3*rho23, Delta3**2]])
    inv = np.linalg.inv(SigmaCMB)
    
    return dO*dO*inv[0][0] + dN*dN*inv[1][1] + dY*dY*inv[2][2] + 2*dO*dN*inv[0][1] + 2*dO*dY*inv[0][2] + 2*dN*dY*inv[1][2]

def chisqSO(OmegaB, Neff, Yp):
    """
    Calculates SO chisq given abundances and omegab/mchi meshgrid
    """
    OmegaBCentre = 0.02236
    NeffCentre = 3.046
    YpCentre = 0.24717
    dO = OmegaB - OmegaBCentre
    dN = Neff - NeffCentre
    dY = Yp - YpCentre
    rho12 = 0.072
    rho13 = 0.33
    rho23 = -0.86
    Delta1 = 0.000073
    Delta2 = 0.11
    Delta3 = 0.0066
    SigmaCMB = np.array([[Delta1**2, Delta1*Delta2*rho12, Delta1*Delta3*rho13], 
                         [Delta1*Delta2*rho12, Delta2**2, Delta2*Delta3*rho23], 
                         [Delta1*Delta3*rho13, Delta2*Delta3*rho23, Delta3**2]])
    inv = np.linalg.inv(SigmaCMB)
    
    return dO*dO*inv[0][0] + dN*dN*inv[1][1] + dY*dY*inv[2][2] + 2*dO*dN*inv[0][1] + 2*dO*dY*inv[0][2] + 2*dN*dY*inv[1][2]

def chisqCMBS4(OmegaB, Neff, Yp):
    """
    Calculates CMB-S4 chisq given abundances and omegab/mchi meshgrid
    """
    OmegaBCentre = 0.02236
    NeffCentre = 3.046
    YpCentre = 0.24717
    dO = OmegaB - OmegaBCentre
    dN = Neff - NeffCentre
    dY = Yp - YpCentre
    rho12 = 0.25
    rho13 = 0.22
    rho23 = -0.84
    Delta1 = 0.000047
    Delta2 = 0.081
    Delta3 = 0.0043
    SigmaCMB = np.array([[Delta1**2, Delta1*Delta2*rho12, Delta1*Delta3*rho13], 
                         [Delta1*Delta2*rho12, Delta2**2, Delta2*Delta3*rho23], 
                         [Delta1*Delta3*rho13, Delta2*Delta3*rho23, Delta3**2]])
    inv = np.linalg.inv(SigmaCMB)
    
    return dO*dO*inv[0][0] + dN*dN*inv[1][1] + dY*dY*inv[2][2] + 2*dO*dN*inv[0][1] + 2*dO*dY*inv[0][2] + 2*dN*dY*inv[1][2]

def chisq(Yp, OmegaB, Neff, type):
    """
    Returns generic chisq meshgrid given Planck, SO, or CMB-S4 type
    """
    if type == 'Planck':
        return chisqPlanck(OmegaB, Neff, Yp)
    elif type == 'SO':
        return chisqSO(OmegaB, Neff, Yp)
    elif type == 'CMBS4':
        return chisqCMBS4(OmegaB, Neff, Yp)

def get_chisq_marg_data(data, type):
    """
    Marginalised chisq to extract Planck, SO, CMB-S4 bounds on mchi
    """
    CHISQ = get_chisq_grid(data, type)
    masses = get_masses(data)
    chisq_marg = np.empty(len(masses))
    for idx, mass in enumerate(masses):
        chisq_marg[idx] = np.min(CHISQ[:, idx])
    chisq_marg_rescaled = chisq_marg #- np.min(chisq_marg) # WE HAVE SET UP MIN CHISQ TO BE ZERO
    return chisq_marg_rescaled

def save_results(data, scenario, save=True):
    """
    Computes bounds on the DM mass from interpolation of get_chisq_marg_data
    """
    masses = get_masses(data)
    DeltaChisqSO = get_chisq_marg_data(data, type='SO')
    DeltaChisqCMBSIV = get_chisq_marg_data(data, type='CMBS4')
    stddev = 0.954499736103642
    cl = chi2.ppf(stddev, 1)
    SOInterp = UnivariateSpline(masses, DeltaChisqSO - cl, s=0)
    CMBSIVInterp = UnivariateSpline(masses, DeltaChisqCMBSIV - cl, s=0)
    SOroot = SOInterp.roots()[0] if any(SOInterp.roots()) else 0.0
    CMBSIVroot = CMBSIVInterp.roots()[0] if any(CMBSIVInterp.roots()) else 0.0


    if save:
        with open(scenario + '/results.txt', 'w') as f:
            print(scenario, file=f)
            print("", file=f)
            print("-----------------------------------------------------------", file=f)
            print(" Conf. Lev. \t SO \t\t CMBS4", file=f)
            print("-----------------------------------------------------------", file=f)
            print(" {} sigma \t {:.3f} MeV \t {:.3f} MeV".format(2, SOroot, CMBSIVroot), file=f)
            print("-----------------------------------------------------------", file=f)
            print("")
            print("NOTE: A value of 0.0 MeV means no bound could be determined.", file=f)
            print("Runtime:", datetime.today().strftime("%d.%m.%Y %H:%M:%S"), file=f)
    else:
        print(scenario)
        print("-----------------------------------------------------------")
        print(" Conf. Lev. \t SO \t\t CMBS4")
        print("-----------------------------------------------------------")
        print(" {} sigma \t {:.3f} MeV \t {:.3f} MeV".format(2, SOroot, CMBSIVroot))
        print("-----------------------------------------------------------")
        print("")
        print("NOTE: A value of 0.0 MeV means no bound could be determined.")

if __name__ == '__main__':
    # for the given scenarios, obtains the Planck and Forecasted data
    scenarios = ['Nu_Maj', 'EE_Maj']
    for scenario in scenarios:
        # only have Planck extension for Majorana case
        planck_data = get_data(scenario + '_Planck.txt')
        forecast_data = get_data(scenario + '_Forecast.txt')
        # plots the confidence intervals and outputs the computed bounds
        plot_joint_mchi_omegab(planck_data, forecast_data, scenario)
        # outputs the results to the screen, or a file
        save_results(forecast_data, scenario, save=False)