"""
Main data and plotting utilities, used extensively in e.g. master.py

To Note:

- Likely place to change figure sizes
- Check data_dir argument in get_data for datafile locations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]



def get_data(filename, data_dir='../../../Data/'):
    """
    Gets data from )data_dir + filename) [Note extension should be added]. Returns dictionary of data fields.
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

def plot_distributions(data, scenario):
    """
    Plots histogram of data values for Yp and D/H
    """
    plt.figure(figsize=(10, 5))
    bins = 40
    alpha = 0.6
    ax = plt.subplot(1, 2, 1)
    ax.hist(data['Yp'], 
            bins=bins, 
            density=True, 
            alpha=alpha)
    ax.set_xlim(np.min(data['Yp']), np.max(data['Yp']))
    ax.set_xlabel(r'$Y_p$')
    ax.set_ylabel(r'$\mathrm{Density}$')
    ax = plt.subplot(1, 2, 2)
    ax.hist(data['D/H'] * 10**5, 
            bins=bins, 
            density=True, 
            alpha=alpha)
    ax.set_xlabel(r'$\mathrm{D}/\mathrm{H} \times 10^5$')
    ax.set_xlim(np.min(data['D/H'])*10**5, np.max(data['D/H'])*10**5)
    plt.suptitle('Distributions')
    plt.savefig(scenario + '/abundance_distributions.pdf')

def plot_abundances(data, scenario, forecast=False):
    """
    Plots colorbar plot of abundances to see dependence on omegab and mchi
    """
    YpCentre = 0.245
    YpError = 0.003
    DHCentre = 2.569 * 10**(-5)
    DHError = 0.027 * 10**(-5)
    if forecast:
        YpCentre = 0.247
        YpError = 0.00029286445646252374
        YpErrorTh = 0.0
        DHCentre = 2.439 * 10**(-5)
        DHError = 1.9952623149688786e-08
        DHErrorTh = 0.0
    
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1,100)
    ax1 = fig.add_subplot(gs[0, 0:40])
    ax2 = fig.add_subplot(gs[0, 54:100])
    ax1.scatter(data['mass'], data['Yp'],
               lw=0.0,
               marker='.',
               s=20,
               alpha=0.8,
               c=data['OmegaB'],
               cmap='PiYG')
    ax1.set_xlabel(r'$m_{\chi} \, \mathrm{[MeV]}$')
    ax1.set_ylabel(r'$Y_p$')
    ax1.set_xscale('log')
    ax1.set_xlim(0.1, np.max(data['mass']))
    # ax1.set_xlim(0.5, 10.0)
    # ax1.set_ylim(0.24, 0.25)
    # ax1.set_yticks([0.24, 0.245, 0.25])
    ax1.add_patch(Rectangle(xy=(0.1, YpCentre - YpError),
                            width=(30.0 - 0.1),
                            height=2*YpError,
                            alpha=0.1,
                            color='k'))
    ax2.add_patch(Rectangle(xy=(0.1, (DHCentre - DHError)*10**5),
                            width=(30.0 - 0.1),
                            height=2*DHError*10**5,
                            alpha=0.1,
                            color='k'))
    sc = ax2.scatter(data['mass'], data['D/H'] * 10**5, 
               lw=0.0,
               marker='.',
               s=20,
               alpha=0.8,
               c=data['OmegaB'],
               cmap='PiYG')
    ax2.set_xlabel(r'$m_{\chi} \, \mathrm{[MeV]}$')
    ax2.set_ylabel(r'$\mathrm{D}/\mathrm{H} \times 10^5$')
    ax2.set_xscale('log')
    ax2.set_xlim(0.1, np.max(data['mass']))
    plt.colorbar(sc, ax=ax2, label=r'$\Omega_{\mathrm{b}} h^2$')
    plt.suptitle('Dependence on Dark Matter Mass')
    plt.savefig(scenario + '/abundances.pdf')

def plot_chisq_distribution(data, scenario):
    """
    Plots distribution of chisq values.
    """
    plt.figure(figsize=(5,5))
    bins = 40
    alpha = 1.0
    plt.hist(chisq(data['Yp'], data['D/H'], data['OmegaB'], data['Neff'], type='BBN'),
             bins=bins, 
             density=True,
             histtype='step',
             alpha=alpha,
             lw=1.7,
             label='BBN')
    plt.hist(chisq(data['Yp'], data['D/H'], data['OmegaB'], data['Neff'], type='CMB'),
             bins=bins, 
             density=True,
             histtype='step',
             alpha=alpha,
             lw=1.7,
             label='CMB')
    plt.hist(chisq(data['Yp'], data['D/H'], data['OmegaB'], data['Neff'], type='BBN+CMB'),
             bins=bins, 
             density=True,
             histtype='step',
             alpha=alpha,
             lw=1.7,
             label='BBN+CMB')
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('Density')
    plt.legend()
    plt.yscale('log')
    plt.title('Distribution of $\chi^2$ Values', fontsize=16)
    plt.savefig(scenario + '/chisq_distributions.pdf')

def plot_mchi_omegab_contours(data, scenario, type):
    """
    Simple plotting for mchi and omegab contours (all confidence levels)
    """
    mass_grid, omegab_grid = get_mass_omegab_grid(data)
    chisq_grid = get_chisq_grid(data, type=type)
    
    confidence_levels = [2.30, 6.18, 11.83, 19.33, 28.74]
    plt.figure(figsize=(5, 5))
    plt.suptitle(type)
    ax = plt.subplot(1,1,1)
    ct = ax.contour(mass_grid, omegab_grid, chisq_grid,
                    levels=confidence_levels,
                    cmap='PiYG')
    ct.levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    ax.set_ylabel(r'$\Omega_b h^2$')
    ax.set_xlim(0.1,)
    ax.set_ylim(0.020, 0.025)
    ax.clabel(ct, ct.levels, inline=True, fontsize=16)
    if type == 'BBN':
        type_str = 'bbn'
    elif type == 'CMB':
        type_str = 'cmb'
    elif type == 'BBN+CMB':
        type_str = 'bbnandcmb'
    plt.savefig(scenario + '/' + type_str + '_mchiomegab.pdf')

def plot_joint_mchi_omegab(data, scenario):
    """
    Produces omegab and mchi contour plot.
    """
    confidence_levels = [0.0, 6.18, 28.74]
    descriptions = {'EE_Neutral_Scalar': 'Electrophilic Neutral Scalar',
                    'EE_Complex_Scalar': 'Electrophilic Complex Scalar',
                    'EE_Maj': 'Electrophilic Majorana Fermion',
                    'EE_Maj_New': r'$\mathrm{Electrophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$',
                    'EE_Dirac': 'Electrophilic Dirac Fermion',
                    'EE_Zp': 'Electrophilic Vector Boson',
                    'Nu_Neutral_Scalar': 'Neutrinophilic Neutral Scalar',
                    'Nu_Complex_Scalar': 'Neutrinophilic Complex Scalar',
                    'Nu_Maj': 'Neutrinophilic Majorana Fermion',
                    'Nu_Maj_New': r'$\mathrm{Neutrinophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$',
                    'Nu_Dirac': 'Neutrinophilic Dirac Fermion',
                    'Nu_Zp': 'Neutrinophilic Vector Boson'}

    save_names = {'EE_Neutral_Scalar': 'EE_Neutral_Scalar',
                    'EE_Complex_Scalar': 'EE_Complex_Scalar',
                    'EE_Maj': 'EE_Maj',
                    'EE_Maj_New': 'EE_Maj',
                    'EE_Dirac': 'EE_Dirac',
                    'EE_Zp': 'EE_Zp',
                    'Nu_Neutral_Scalar': 'Nu_Neutral_Scalar',
                    'Nu_Complex_Scalar': 'Nu_Complex_Scalar',
                    'Nu_Maj': 'Nu_Maj',
                    'Nu_Maj_New': 'Nu_Maj',
                    'Nu_Dirac': 'Nu_Dirac',
                    'Nu_Zp': 'Nu_Zp'}

    locs = {'EE_Maj_New': 'lower right',
            'Nu_Maj_New': 'lower left'}

    markers = {'EE_Maj_New': False,
            'Nu_Maj_New': True}

    if 'EE' in scenario:
        zorders = [-1, -1, -1, -1, 0, 1, 2, 3]
    else:
        zorders = [-1, -1, -1, -1, 2, 3, 0, 1]

    MASS, OMEGAB = get_mass_omegab_grid(data)
    CHISQBBN = get_chisq_grid(data, type='BBN') 
    CHISQBBN = CHISQBBN - np.min(CHISQBBN)
    CHISQBBNOM = get_chisq_grid(data, type='BBN+Omegab') 
    CHISQBBNOM = CHISQBBNOM - np.min(CHISQBBNOM)
    CHISQCMB = get_chisq_grid(data, type='CMB') 
    CHISQCMB = CHISQCMB - np.min(CHISQCMB)
    CHISQBBNandCMB = get_chisq_grid(data, type='BBN+CMB') 
    CHISQBBNandCMB = CHISQBBNandCMB - np.min(CHISQBBNandCMB)

    plt.figure(figsize=(6,5))
    #plt.suptitle('Combining Measurements')
    ax = plt.subplot(1,1,1)
    ax.set_xlim(0.1,11)
    if 'nu' in scenario.lower():
        ax.set_ylim(0.021, 0.024)
        ax.set_yticks([0.021, 0.022, 0.023, 0.024])
        #ax.text(0.4, 0.0212, descriptions[scenario], color='k', fontsize=12)
    else:
        ax.set_ylim(0.0196, 0.023)
        ax.set_yticks([0.020, 0.021, 0.022, 0.023])
        #ax.text(0.4, 0.0192, descriptions[scenario], color='k', fontsize=12)
    plt.title(descriptions[scenario], fontsize=22)
    stddev = np.array([0., 0.682689492137086, 0.954499736103642])
    confidence_levels = []
    for j in range(len(stddev)):
        confidence_levels.append(chi2.ppf(stddev[j], 2))

    vmin = 0.0
    vmax = 7.0
    ct = ax.contourf(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Blues_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[0])
    ct = ax.contour(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#3F7BB6'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[1])
    ct = ax.contourf(MASS, OMEGAB, CHISQBBNOM,
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Purples_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[2])
    ct = ax.contour(MASS, OMEGAB, CHISQBBNOM, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['indigo'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[3])
    ct = ax.contourf(MASS, OMEGAB, CHISQCMB,
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Reds_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[4])
    ct = ax.contour(MASS, OMEGAB, CHISQCMB, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#BF4145'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[5])
    ct = ax.contourf(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Greens_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[6])
    ct = ax.contour(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#306B37'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[7])
    proxy = [plt.Rectangle((0,0),1,1,fc='#3F7BB6',alpha=0.8),
             plt.Rectangle((0,0),1,1,fc='#BF4145',alpha=0.8), 
             plt.Rectangle((0,0),1,1,fc='indigo',alpha=0.8),
             plt.Rectangle((0,0),1,1,fc='#306B37',alpha=0.8)]
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$', fontsize=22)
    ax.set_ylabel(r'$\Omega_{\mathrm{b}} h^2$', fontsize=22)
    ax.set_xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    ax.set_xlim((0.0, 14.0))
    ax.xaxis.set_tick_params(labelsize=20, zorder=8)
    ax.yaxis.set_tick_params(labelsize=20, zorder=8)


    ax.legend(proxy, [r"BBN", r"Planck", r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$', r"BBN+Planck",], 
        fontsize=14, 
        loc=locs[scenario],
        markerfirst=markers[scenario])
    plt.savefig(save_names[scenario] + '/{}_exclusion.pdf'.format(save_names[scenario]))

def plot_cts_and_deltachi(data, new_data, scenario):
    """
    IMPORTANT: Produces the joint contour and delta chisq plots in one figure.
    """
    confidence_levels = [0.0, 6.18, 28.74]
    descriptions = {'EE_Neutral_Scalar': 'Electrophilic Neutral Scalar',
                    'EE_Complex_Scalar': 'Electrophilic Complex Scalar',
                    'EE_Maj': 'Electrophilic Majorana Fermion',
                    'EE_Maj_New': r'$\mathrm{Electrophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$',
                    'EE_Dirac': 'Electrophilic Dirac Fermion',
                    'EE_Zp': 'Electrophilic Vector Boson',
                    'Nu_Neutral_Scalar': 'Neutrinophilic Neutral Scalar',
                    'Nu_Complex_Scalar': 'Neutrinophilic Complex Scalar',
                    'Nu_Maj': 'Neutrinophilic Majorana Fermion',
                    'Nu_Maj_New': r'$\mathrm{Neutrinophilic}\,\mathrm{Majorana}\,\mathrm{Fermion}$',
                    'Nu_Dirac': 'Neutrinophilic Dirac Fermion',
                    'Nu_Zp': 'Neutrinophilic Vector Boson'}

    save_names = {'EE_Neutral_Scalar': 'EE_Neutral_Scalar',
                    'EE_Complex_Scalar': 'EE_Complex_Scalar',
                    'EE_Maj': 'EE_Maj',
                    'EE_Maj_New': 'EE_Maj',
                    'EE_Dirac': 'EE_Dirac',
                    'EE_Zp': 'EE_Zp',
                    'Nu_Neutral_Scalar': 'Nu_Neutral_Scalar',
                    'Nu_Complex_Scalar': 'Nu_Complex_Scalar',
                    'Nu_Maj': 'Nu_Maj',
                    'Nu_Maj_New': 'Nu_Maj',
                    'Nu_Dirac': 'Nu_Dirac',
                    'Nu_Zp': 'Nu_Zp'}

    locs = {'EE_Maj_New': 'lower right',
            'Nu_Maj_New': 'lower left'}

    markers = {'EE_Maj_New': False,
            'Nu_Maj_New': True}

    if 'EE' in scenario:
        zorders = [-1, -1, -1, -1, 0, 1, 2, 3]
    else:
        zorders = [-1, -1, -1, -1, 2, 3, 0, 1]

    MASS, OMEGAB = get_mass_omegab_grid(data)
    CHISQBBN = get_chisq_grid(data, type='BBN') 
    CHISQBBN = CHISQBBN - np.min(CHISQBBN)
    CHISQBBNOM = get_chisq_grid(data, type='BBN+Omegab') 
    CHISQBBNOM = CHISQBBNOM - np.min(CHISQBBNOM)
    CHISQCMB = get_chisq_grid(data, type='CMB') 
    CHISQCMB = CHISQCMB - np.min(CHISQCMB)
    CHISQBBNandCMB = get_chisq_grid(data, type='BBN+CMB') 
    CHISQBBNandCMB = CHISQBBNandCMB - np.min(CHISQBBNandCMB)

    plt.figure(figsize=(6,12))
    #plt.suptitle('Combining Measurements')
    ax = plt.subplot(2,1,1)
    ax.set_xlim(0.1,11)
    if 'nu' in scenario.lower():
        ax.set_ylim(0.021, 0.024)
        ax.set_yticks([0.021, 0.022, 0.023, 0.024])
        #ax.text(0.4, 0.0212, descriptions[scenario], color='k', fontsize=12)
    else:
        ax.set_ylim(0.0196, 0.023)
        ax.set_yticks([0.020, 0.021, 0.022, 0.023])
        #ax.text(0.4, 0.0192, descriptions[scenario], color='k', fontsize=12)
    plt.title(descriptions[scenario], fontsize=22)
    stddev = np.array([0., 0.682689492137086, 0.954499736103642])
    confidence_levels = []
    for j in range(len(stddev)):
        confidence_levels.append(chi2.ppf(stddev[j], 2))

    vmin = 0.0
    vmax = 7.0
    ct = ax.contourf(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Blues_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[0])
    ct = ax.contour(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#3F7BB6'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[1])
    ct = ax.contourf(MASS, OMEGAB, CHISQBBNOM,
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Purples_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[2])
    ct = ax.contour(MASS, OMEGAB, CHISQBBNOM, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['indigo'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[3])
    ct = ax.contourf(MASS, OMEGAB, CHISQCMB,
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Reds_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[4])
    ct = ax.contour(MASS, OMEGAB, CHISQCMB, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#BF4145'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[5])
    ct = ax.contourf(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Greens_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[6])
    ct = ax.contour(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#306B37'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest',
                     zorder=zorders[7])
    proxy = [plt.Rectangle((0,0),1,1,fc='#3F7BB6',alpha=0.8),
             plt.Rectangle((0,0),1,1,fc='#BF4145',alpha=0.8), 
             plt.Rectangle((0,0),1,1,fc='indigo',alpha=0.8),
             plt.Rectangle((0,0),1,1,fc='#306B37',alpha=0.8)]
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$', fontsize=22)
    ax.set_ylabel(r'$\Omega_{\mathrm{b}} h^2$', fontsize=22)
    ax.set_xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    ax.set_xlim((0.0, 14.0))
    ax.xaxis.set_tick_params(labelsize=20, zorder=8)
    ax.yaxis.set_tick_params(labelsize=20, zorder=8)


    ax.legend(proxy, [r"BBN", r"Planck", r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$', r"BBN+Planck",], 
        fontsize=14, 
        loc=locs[scenario],
        markerfirst=markers[scenario])

    ax = plt.subplot(2, 1, 2)
    scenario = scenario.replace("_New","")
    data = new_data

    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN')
    bbnominterpfn = get_chisq_marg_interpolation(data, type='BBN+Omegab')
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB')
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB')
    masses = get_masses(data)
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
    markerfirstdict = {'EE_Neutral_Scalar': False,
                        'EE_Complex_Scalar': False,
                        'EE_Maj': False,
                        'EE_Dirac': False,
                        'EE_Zp': False,
                        'Nu_Neutral_Scalar': False,
                        'Nu_Complex_Scalar': False,
                        'Nu_Maj': True,
                        'Nu_Dirac': True,
                        'Nu_Zp': True}
    locs = {'EE_Neutral_Scalar': 'upper right',
                    'EE_Complex_Scalar': 'upper right',
                    'EE_Maj': 'upper right',
                    'EE_Dirac': 'upper right',
                    'EE_Zp': 'upper right',
                    'Nu_Neutral_Scalar': 'upper right',
                    'Nu_Complex_Scalar': 'upper right',
                    'Nu_Maj': None,
                    'Nu_Dirac': 'upper left',
                    'Nu_Zp': 'upper left'}

    MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
    col = 'k'

    ax.plot(masses, np.sqrt(np.abs(bbnandcmbinterpfn(masses))),lw=2.8,label='BBN+Planck',c='#306B37',ls=(0, (1, 1)))
    ax.plot(masses, np.sqrt(np.abs(bbnominterpfn(masses))),lw=2.8,label=r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$',c='purple',ls=(0, (3, 1, 1, 1)))
    ax.plot(masses, np.sqrt(np.abs(cmbinterpfn(masses))),lw=2.5,label='Planck',c='#BF4145',ls=(0, (5, 2)))
    ax.plot(masses, np.sqrt(np.abs(bbninterpfn(masses))),lw=2.5,label='BBN',c='#3F7BB6',ls='-')
    
    
    
    confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    ct = ax.contour(MC, CSQ, CSQ, 
                    levels=confidence_levels,
                    linewidths=1.0,
                    alpha=0.4,
                    colors=['k'])
    levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    for idx, level in enumerate(levels):
        plt.text(0.112, (idx + 1) + 0.075, level, color='k')
    #plt.text(0.115, 0.22, descriptions[scenario], color='k', fontsize=11)
    #plt.title(descriptions[scenario], color='k', fontsize=20)
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$', fontsize=22)
    ax.set_ylabel(r'$\sqrt{\Delta \chi^2}$', labelpad=15, fontsize=22)
    ax.set_xscale('log')
    ax.set_xticks([0.1, 1.0, 10.0, 30.0])
    ax.set_xlim(0.1,30.0)
    ax.set_ylim(0, 7.0)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if type(locs[scenario]) != type(None):
        ax.legend(fontsize=13, markerfirst=markerfirstdict[scenario], handlelength = 3, loc=locs[scenario])
    plt.savefig(save_names[scenario] + '/{}_exclusion_and_deltachi.pdf'.format(save_names[scenario]))

def plot_deltachisq(data, scenario, zoom=False):
    """
    Plots simple deltachisq given mchi, omegab data
    """
    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN')
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB')
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB')
    masses = get_masses(data)

    MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
    col = 'k'

    plt.figure(figsize=(8,6))
    plt.plot(masses, cmbinterpfn(masses),lw=1.7,label='CMB',c='#BF4145',ls='-')
    plt.plot(masses, bbnandcmbinterpfn(masses),lw=1.7,label='BBN+CMB',c='#306B37',ls='-')
    plt.plot(masses, bbninterpfn(masses),lw=1.7,label='BBN',c='#3F7BB6',ls='-')
    confidence_levels = [1.0, 4.0, 9.0, 16.0, 25.0]
    ct = plt.gca().contour(MC, CSQ, CSQ, 
                    levels=confidence_levels,
                    linewidths=1.0,
                    alpha=0.4,
                    colors=['k'])
    levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    for idx, level in enumerate(levels):
        plt.text(0.105, (idx + 1)**2 + 0.3, level, color='k')
    plt.xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    plt.ylabel(r'$\Delta \tilde{\chi}^2$')
    plt.xscale('log')
    plt.gca().set_xlim(0.1,10)
    if zoom:
        plt.gca().set_ylim(0, 40)
    else:
        plt.gca().set_ylim(0, 80)
    plt.legend(fontsize=14, markerfirst=False)
    plt.tick_params(axis='x', which='minor', size=4)
    if zoom:
        plt.savefig(scenario + '/chisqmarg_zoom.pdf')
    else:
        plt.savefig(scenario + '/chisqmarg_full.pdf')

def plot_sqrtdeltachisq(data, scenario):
    """
    Plots the sqrt delta chisq curve with the smaller figure size for appendix
    """
    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN')
    bbnominterpfn = get_chisq_marg_interpolation(data, type='BBN+Omegab')
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB')
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB')
    masses = get_masses(data)
    descriptions = {'EE_Neutral_Scalar': 'Neutral Scalar',
                    'EE_Complex_Scalar': 'Complex Scalar',
                    'EE_Maj': 'Majorana Fermion',
                    'EE_Dirac': 'Dirac Fermion',
                    'EE_Zp': 'Vector Boson',
                    'Nu_Neutral_Scalar': 'Neutral Scalar',
                    'Nu_Complex_Scalar': 'Complex Scalar',
                    'Nu_Maj': 'Majorana Fermion',
                    'Nu_Dirac': 'Dirac Fermion',
                    'Nu_Zp': 'Vector Boson'}
    markerfirstdict = {'EE_Neutral_Scalar': False,
                        'EE_Complex_Scalar': False,
                        'EE_Maj': False,
                        'EE_Dirac': False,
                        'EE_Zp': False,
                        'Nu_Neutral_Scalar': False,
                        'Nu_Complex_Scalar': False,
                        'Nu_Maj': True,
                        'Nu_Dirac': True,
                        'Nu_Zp': True}
    locs = {'EE_Neutral_Scalar': 'upper right',
                    'EE_Complex_Scalar': None,
                    'EE_Maj': None,
                    'EE_Dirac': None,
                    'EE_Zp': None,
                    'Nu_Neutral_Scalar': None,
                    'Nu_Complex_Scalar': None,
                    'Nu_Maj': None,
                    'Nu_Dirac': None,
                    'Nu_Zp': None}

    MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
    col = 'k'

    plt.figure(figsize=(6,4))
    plt.plot(masses, np.sqrt(np.abs(bbnandcmbinterpfn(masses))),lw=2.8,label='BBN+Planck',c='#306B37',ls=(0, (1, 1)))
    plt.plot(masses, np.sqrt(np.abs(bbnominterpfn(masses))),lw=2.8,label=r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$',c='purple',ls=(0, (3, 1, 1, 1)))
    plt.plot(masses, np.sqrt(np.abs(cmbinterpfn(masses))),lw=2.5,label='Planck',c='#BF4145',ls=(0, (5, 2)))
    plt.plot(masses, np.sqrt(np.abs(bbninterpfn(masses))),lw=2.5,label='BBN',c='#3F7BB6',ls='-')
    confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    ct = plt.gca().contour(MC, CSQ, CSQ, 
                    levels=confidence_levels,
                    linewidths=1.0,
                    alpha=0.4,
                    colors=['k'])
    levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    for idx, level in enumerate(levels):
        plt.text(0.112, (idx + 1) + 0.075, level, color='k')
    plt.text(0.12, 0.24, descriptions[scenario], color='k', fontsize=16)
    if scenario in ['Nu_Zp', 'EE_Zp']:    
        plt.xlabel(r'$m_\chi \, \mathrm{[MeV]}$', fontsize=22)
    plt.ylabel(r'$\sqrt{\Delta \chi^2}$', fontsize=22)
    plt.xscale('log')
    plt.xticks([0.1, 1.0, 10.0, 30.0])
    plt.gca().set_xlim(0.1,30.0)
    plt.gca().set_ylim(0, 7.0)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().xaxis.set_tick_params(labelsize=20)
    plt.gca().yaxis.set_tick_params(labelsize=20)
    if type(locs[scenario]) != type(None):
        plt.legend(fontsize=14, markerfirst=markerfirstdict[scenario], ncol=1, handlelength = 3, loc=locs[scenario])
    # if scenario == 'EE_Neutral_Scalar':
    #     plt.title(r'$\mathrm{Electrophilic}$', color='k', fontsize=22)
    # if scenario == 'Nu_Neutral_Scalar':
    #     plt.title(r'$\mathrm{Neutrinophilic}$', color='k', fontsize=22)
    plt.savefig(scenario + '/{}_sqrtchisq.pdf'.format(scenario))
    # plt.savefig('DHErrorCheck/{}-large-error.pdf'.format(scenario))

def plot_sqrtdeltachisqmain(data, scenario):
    """
    Plots larger sqrt deltachisq plot. Similar to plot_sqrtdeltachisq.
    """
    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN')
    bbnominterpfn = get_chisq_marg_interpolation(data, type='BBN+Omegab')
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB')
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB')
    masses = get_masses(data)
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
    markerfirstdict = {'EE_Neutral_Scalar': False,
                        'EE_Complex_Scalar': False,
                        'EE_Maj': False,
                        'EE_Dirac': False,
                        'EE_Zp': False,
                        'Nu_Neutral_Scalar': False,
                        'Nu_Complex_Scalar': False,
                        'Nu_Maj': True,
                        'Nu_Dirac': True,
                        'Nu_Zp': True}
    locs = {'EE_Neutral_Scalar': 'upper right',
                    'EE_Complex_Scalar': 'upper right',
                    'EE_Maj': 'upper right',
                    'EE_Dirac': 'upper right',
                    'EE_Zp': 'upper right',
                    'Nu_Neutral_Scalar': 'upper right',
                    'Nu_Complex_Scalar': 'upper right',
                    'Nu_Maj': None,
                    'Nu_Dirac': 'upper left',
                    'Nu_Zp': 'upper left'}

    MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
    col = 'k'

    plt.figure(figsize=(6,5))
    plt.plot(masses, np.sqrt(np.abs(bbnandcmbinterpfn(masses))),lw=2.8,label='BBN+Planck',c='#306B37',ls=(0, (1, 1)))
    plt.plot(masses, np.sqrt(np.abs(bbnominterpfn(masses))),lw=2.8,label=r'BBN+$\boldsymbol{\Omega_{\mathbf{b}}h^2}$',c='purple',ls=(0, (3, 1, 1, 1)))
    plt.plot(masses, np.sqrt(np.abs(cmbinterpfn(masses))),lw=2.5,label='Planck',c='#BF4145',ls=(0, (5, 2)))
    plt.plot(masses, np.sqrt(np.abs(bbninterpfn(masses))),lw=2.5,label='BBN',c='#3F7BB6',ls='-')
    
    
    
    confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    ct = plt.gca().contour(MC, CSQ, CSQ, 
                    levels=confidence_levels,
                    linewidths=1.0,
                    alpha=0.4,
                    colors=['k'])
    levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    for idx, level in enumerate(levels):
        plt.text(0.105, (idx + 1) + 0.075, level, color='k')
    #plt.text(0.115, 0.22, descriptions[scenario], color='k', fontsize=11)
    plt.title(descriptions[scenario], color='k', fontsize=20)
    plt.xlabel(r'$m_\chi \, \mathrm{[MeV]}$', fontsize=22)
    plt.ylabel(r'$\sqrt{\Delta \chi^2}$', labelpad=15, fontsize=22)
    plt.xscale('log')
    plt.xticks([0.1, 1.0, 10.0, 30.0])
    plt.gca().set_xlim(0.1,30.0)
    plt.gca().set_ylim(0, 7.0)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().xaxis.set_tick_params(labelsize=20)
    plt.gca().yaxis.set_tick_params(labelsize=20)
    if type(locs[scenario]) != type(None):
        plt.legend(fontsize=13, markerfirst=markerfirstdict[scenario], handlelength = 3, loc=locs[scenario])
    plt.savefig(scenario + '/{}_sqrtchisq_main.pdf'.format(scenario))

def plot_joint_mchi_omegab_forecast(data, scenario):
    """
    Plots forecasted mchi omegab contours for BBN, CMB and BBN+CMB. For CMB forecasts, cmbforecasts.py should be used.
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

    MASS, OMEGAB = get_mass_omegab_grid(data)
    CHISQBBN = get_chisq_grid(data, type='BBN', forecast=True) 
    CHISQBBN = CHISQBBN - np.min(CHISQBBN)
    CHISQCMB = get_chisq_grid(data, type='CMB', forecast=True) 
    CHISQCMB = CHISQCMB - np.min(CHISQCMB)
    CHISQBBNandCMB = get_chisq_grid(data, type='BBN+CMB', forecast=True) 
    CHISQBBNandCMB = CHISQBBNandCMB - np.min(CHISQBBNandCMB)

    plt.figure(figsize=(7,6))
    #plt.suptitle('Combining Measurements')
    ax = plt.subplot(1,1,1)
    ax.set_xlim(0.1,11)
    if 'nu' in scenario.lower():
        ax.set_ylim(0.021, 0.025)
        ax.set_yticks([0.021, 0.022, 0.023, 0.024, 0.025])
        ax.text(0.4, 0.0212, descriptions[scenario], color='k', fontsize=12)
    else:
        ax.set_ylim(0.019, 0.023)
        ax.set_yticks([0.019, 0.020, 0.021, 0.022, 0.023])
        ax.text(0.4, 0.0192, descriptions[scenario], color='k', fontsize=12)
    stddev = np.array([0., 0.682689492137086, 0.954499736103642])
    confidence_levels = []
    for j in range(len(stddev)):
        confidence_levels.append(chi2.ppf(stddev[j], 2))

    vmin = 0.0
    vmax = 7.0
    ct = ax.contourf(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Blues_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest')
    ct = ax.contour(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#3F7BB6'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest')
    ct = ax.contourf(MASS, OMEGAB, CHISQCMB,
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Reds_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest')
    ct = ax.contour(MASS, OMEGAB, CHISQCMB, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#BF4145'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest')
    ct = ax.contourf(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     cmap=plt.get_cmap('Greens_r'),
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest')
    ct = ax.contour(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     linewidths=0.5,
                     colors=['#306B37'],
                     vmin=vmin,
                     vmax=vmax,
                     interpolation='nearest')
    proxy = [plt.Rectangle((0,0),1,1,fc='#BF4145',alpha=0.8), 
             plt.Rectangle((0,0),1,1,fc='#306B37',alpha=0.8),
             plt.Rectangle((0,0),1,1,fc='#3F7BB6',alpha=0.8)]
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    ax.set_ylabel(r'$\Omega_{\mathrm{b}} h^2$')
    ax.set_xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    ax.set_xlim((0.0, 12.0))


    ax.legend(proxy, [r"CMB", r"BBN+CMB", r"BBN"], fontsize=14, loc='upper left')
    plt.savefig(scenario + '/exclusion_forecast.pdf')

def plot_sqrtdeltachisq_forecast(data, scenario):
    """
    Plots sqrt delta chisq forecasts.
    """
    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN', forecast=True)
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB', forecast=True)
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB', forecast=True)
    masses = get_masses(data)
    descriptions = {'EE_Neutral_Scalar': 'Electrophilic Neutral Scalar',
                    'EE_Complex_Scalar': 'Electrophilic Complex Scalar',
                    'EE_Maj': 'Electrophilic Majorana Fermion',
                    'EE_Dirac': 'Electrophilic Dirac Fermion',
                    'EE_Zp': 'Electrophilic Vector Boson',
                    'Nu_Neutral_Scalar': 'Neutrinophilic Neutral Scalar',
                    'Nu_Complex_Scalar': 'Neutrinophilic Complex Scalar',
                    'Nu_Maj': 'Neutrinophilic Majorana Fermion',
                    'Nu_Dirac': 'Neutrinophilic Dirac Fermion',
                    'Nu_Zp': 'Neutrinophilic Vector Boson',
                    'EE_Maj_Sigma': 'Electrophilic Majorana Fermion'}
    markerfirstdict = {'EE_Neutral_Scalar': False,
                        'EE_Complex_Scalar': False,
                        'EE_Maj': False,
                        'EE_Dirac': False,
                        'EE_Zp': False,
                        'Nu_Neutral_Scalar': False,
                        'Nu_Complex_Scalar': False,
                        'Nu_Maj': True,
                        'Nu_Dirac': True,
                        'Nu_Zp': True,
                        'EEE_Maj_Sigma': True}
    locs = {'EE_Neutral_Scalar': 'upper right',
                    'EE_Complex_Scalar': None,
                    'EE_Maj': None,
                    'EE_Dirac': None,
                    'EE_Zp': None,
                    'Nu_Neutral_Scalar': None,
                    'Nu_Complex_Scalar': None,
                    'Nu_Maj': None,
                    'Nu_Dirac': None,
                    'Nu_Zp': None,
                    'EE_Maj_Sigma': None}

    MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
    col = 'k'

    plt.figure(figsize=(6,4))
    plt.plot(masses, np.sqrt(cmbinterpfn(masses)),lw=2.5,label='CMB',c='#BF4145',ls='--')
    plt.plot(masses, np.sqrt(bbnandcmbinterpfn(masses)),lw=2.8,label='BBN+CMB',c='#306B37',ls=(0, (1, 1)))
    plt.plot(masses, np.sqrt(bbninterpfn(masses)),lw=2.5,label='BBN',c='#3F7BB6',ls='-')
    confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    ct = plt.gca().contour(MC, CSQ, CSQ, 
                    levels=confidence_levels,
                    linewidths=1.0,
                    alpha=0.4,
                    colors=['k'])
    levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    for idx, level in enumerate(levels):
        plt.text(0.105, (idx + 1) + 0.075, level, color='k')
    plt.text(0.115, 0.22, descriptions[scenario], color='k', fontsize=11)
    plt.xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    plt.ylabel(r'$\sqrt{\Delta \tilde{\chi}^2}$')
    plt.xscale('log')
    plt.gca().set_xlim(0.1,13.9)
    plt.gca().set_ylim(0, 7.0)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    if type(locs[scenario]) != type(None):
        plt.legend(fontsize=14, markerfirst=markerfirstdict[scenario], handlelength = 3, loc=locs[scenario])
    plt.savefig(scenario + '/{}_sqrtchisq_forecast.pdf'.format(scenario))

def get_mass_omegab_grid(data):
    """
    Generates 2d meshgrid for mchi and omegab
    """
    masses = np.unique(data['mass'])
    omegabs = np.unique(data['OmegaB'])
    MASS, OMEGAB = np.meshgrid(masses, omegabs)
    OMEGABDAT = data['OmegaB'].reshape(len(masses), -1).T
    return MASS, OMEGAB

def get_chisq_grid(data, type, forecast=False, errors=None):
    """
    Generates 2d meshgrid for chisq values of a given type (i.e. BBN, CMB etc)
    """
    masses = np.unique(data['mass'])
    omegabs = np.unique(data['OmegaB'])
    MASS, OMEGAB = np.meshgrid(masses, omegabs)
    OMEGABDAT = data['OmegaB'].reshape(len(masses), -1).T
    YP = data['Yp'].reshape(len(masses), -1).T
    DH = data['D/H'].reshape(len(masses), -1).T
    NEFF = data['Neff'].reshape(len(masses), -1).T
    return chisq(YP, DH, OMEGABDAT, NEFF, type, forecast, errors)

def get_masses(data):
    """
    Returns 1d array for masses
    """
    return np.unique(data['mass'])

def chisqBBN(Yp, DoverH, forecast=False, errors=None, Omegab=None):
    """
    Computes BBN chisq with forecasting option
    """
    if forecast:
        YpCentre = 0.247
        YpError = 0.003
        YpErrorTh = 0.00017/5
        DHCentre = 2.439 * 10**(-5)
        DHError = (0.005) * 10**(-5)
        DHErrorTh = 0.007 * 10**(-5)
        YpCentre = 0.247
        YpError = 0.00029286445646252374
        YpErrorTh = 0.0
        DHCentre = 2.439 * 10**(-5)
        DHError = 1.9952623149688786e-08
        DHErrorTh = 0.0
    else:
        YpCentre = 0.245
        # YpCentre = 0.24717
        YpError = 0.003
        YpErrorTh = 0.00017
        DHCentre = 2.569 * 10**(-5)
        # DHCentre = 2.439 * 10**(-5)
        DHError = (0.027) * 10**(-5)
        DHErrorTh = 0.036 * 10**(-5)
        OmegabCentre = 0.02225
        # OmegabCentre = 0.02236
        OmegabError = 0.00022*3


    if type(errors) != type(None):
        if type(Omegab) != type(None):
            return np.power(Yp - YpCentre, 2)/(errors[0]**2) \
         + np.power(DoverH - DHCentre, 2)/(errors[1]**2) + np.power(Omegab - OmegabCentre, 2)/(OmegabError**2)
        else:
            return np.power(Yp - YpCentre, 2)/(errors[0]**2) \
         + np.power(DoverH - DHCentre, 2)/(errors[1]**2)
    else:
        if type(Omegab) != type(None):
            return 0.0
         #    return np.power(Yp - YpCentre, 2)/(YpError**2 + YpErrorTh**2) \
         # + np.power(DoverH - DHCentre, 2)/(DHError**2 + DHErrorTh**2) + np.power(Omegab - OmegabCentre, 2)/(OmegabError**2)
        else:
            return np.power(Yp - YpCentre, 2)/(YpError**2 + YpErrorTh**2) \
         + np.power(DoverH - DHCentre, 2)/(DHError**2 + DHErrorTh**2)

def chisqBBNOM(Yp, DoverH, OmegaB, forecast=False, errors=None):
    """
    Computes chisq grid with a prior on OmegaB
    """
    YpCentre = 0.245
    # YpCentre = 0.24717
    YpError = 0.003
    YpErrorTh = 0.00017
    DHCentre = 2.569 * 10**(-5)
    # DHCentre = 2.439 * 10**(-5)
    DHError = (0.027) * 10**(-5)
    DHErrorTh = 0.036 * 10**(-5)
    OmegabCentre = 0.02225
    # OmegabCentre = 0.02236
    OmegabError = 0.00022*3
    return np.power(OmegaB - OmegabCentre, 2)/(OmegabError**2)

def chisqCMB(OmegaB, Neff, Yp, forecast=False):
    """
    Computes Planck CMB chisq
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

def chisqBBNandCMB(Yp, DoverH, OmegaB, Neff, forecast=False, errors=None):
    """
    Computes BBN+Planck chisq
    """
    return chisqBBN(Yp, DoverH, forecast, errors) + chisqCMB(OmegaB, Neff, Yp, forecast)

def chisq(Yp, DoverH, OmegaB, Neff, type, forecast=False, errors=None):
    """
    Combines above chisq calculations to accept a type e.g. BBN, BBN + CMB etc.
    """
    if type == 'BBN':
        return chisqBBN(Yp, DoverH, forecast, errors)
    if type == 'BBN+Omegab':
        return chisqBBN(Yp, DoverH, forecast, errors) + chisqBBNOM(Yp, DoverH, OmegaB)
    elif type == 'CMB':
        return chisqCMB(OmegaB, Neff, Yp, forecast)
    elif type == 'BBN+CMB':
        return chisqBBNandCMB(Yp, DoverH, OmegaB, Neff, forecast, errors)


def get_chisq_marg_interpolation(data, type, forecast=False, kind='cubic'):
    """
    Returns interpolating function for marginalised chisq
    """
    CHISQ = get_chisq_grid(data, type, forecast)
    masses = get_masses(data)
    chisq_marg = np.empty(len(masses))
    for idx, mass in enumerate(masses):
        chisq_marg[idx] = np.min(CHISQ[:, idx])
    chisq_marg_rescaled = chisq_marg - np.min(chisq_marg)
    # print(np.min(chisq_marg))
    return interp1d(masses, chisq_marg_rescaled, kind=kind, fill_value='extrapolate')

def get_chisq_marg_data(data, type, forecast=False, errors=None):
    """
    Returns 1d array to do interpolation on for marginalised chisq using get_chisq_marg_interpolation
    """
    CHISQ = get_chisq_grid(data, type, forecast, errors)
    masses = get_masses(data)
    chisq_marg = np.empty(len(masses))
    for idx, mass in enumerate(masses):
        chisq_marg[idx] = np.min(CHISQ[:, idx])
    chisq_marg_rescaled = chisq_marg - np.min(chisq_marg)
    return chisq_marg_rescaled

def save_results(data, scenario, save=True):
    """
    Save interpolation results for bounds on the DM mass
    """
    masses = get_masses(data)
    DeltaChisqBBN = get_chisq_marg_data(data, type='BBN')
    DeltaChisqBBNOM = get_chisq_marg_data(data, type='BBN+Omegab')
    DeltaChisqCMB = get_chisq_marg_data(data, type='CMB')
    DeltaChisqBBNCMB = get_chisq_marg_data(data, type='BBN+CMB')
    stddev = 0.954499736103642
    cl = chi2.ppf(stddev, 1)
    BBNInterp = UnivariateSpline(masses, DeltaChisqBBN - cl, s=0)
    BBNOMInterp = UnivariateSpline(masses, DeltaChisqBBNOM - cl, s=0)
    CMBInterp = UnivariateSpline(masses, DeltaChisqCMB - cl, s=0)
    BBNCMBInterp = UnivariateSpline(masses, DeltaChisqBBNCMB - cl, s=0)
    BBNroot = BBNInterp.roots()[0] if any(BBNInterp.roots()) else 0.0
    BBNOMroot = BBNOMInterp.roots()[0] if any(BBNOMInterp.roots()) else 0.0
    CMBroot = CMBInterp.roots()[0] if any(CMBInterp.roots()) else 0.0
    BBNCMBroot = BBNCMBInterp.roots()[0] if any(BBNCMBInterp.roots()) else 0.0


    if save:
        with open(scenario + '/results.txt', 'w') as f:
            print(scenario, file=f)
            print("", file=f)
            print("-----------------------------------------------------------", file=f)
            print(" Conf. Lev. \t BBN \t BBN+Ob \t CMB \t\t BBN + CMB", file=f)
            print("-----------------------------------------------------------", file=f)
            print(" {} sigma \t {:.3f} MeV \t {:.3f} MeV \t {:.3f} MeV \t {:.3f} MeV".format(2, BBNroot, BBNOMroot, CMBroot, BBNCMBroot), file=f)
            print("-----------------------------------------------------------", file=f)
            print("")
            print("NOTE: A value of 0.0 MeV means no bound could be determined.", file=f)
            print("Runtime:", datetime.today().strftime("%d.%m.%Y %H:%M:%S"), file=f)
    else:
        print(scenario)
        print("--------------------------------------------------------------------------")
        print(" Conf. Lev. \t BBN \t\t BBN+Ob \t CMB \t\t BBN + CMB")
        print("--------------------------------------------------------------------------")
        print(" {} sigma \t {:.1f} MeV \t {:.1f} MeV \t {:.1f} MeV \t {:.1f} MeV".format(2, BBNroot, BBNOMroot, CMBroot, BBNCMBroot))
        print("--------------------------------------------------------------------------")
        print("NOTE: A value of 0.0 MeV means no bound could be determined.\n")
