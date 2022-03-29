#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic
from plot_utils import get_col
from scipy.ndimage import gaussian_filter1d
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

ipi_data = pickle.load(open('../results/ipi_data.p', 'rb'))
all_ipis = ipi_data['ipis']
names = ipi_data['names']
trials = ipi_data['trials']
autocors = ipi_data['autocors']
autocor_inds = ipi_data['autocor_inds']

fig = plt.figure(figsize = (16*cm, 3*cm))

gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=1., wspace = 0.5, hspace = 0.20)

ax = fig.add_subplot(gs[0, 0])
#ax.text(-0.35, 1.20, 'A', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')

cols = [[0, 0, 1], [0, 0, 0.7], [0, 0, 0.4], [1, 0, 0], [0.7, 0, 0], [0.4, 0, 0]]

for iname, name in enumerate(names):
    ipis = all_ipis[iname]
    
    ipis_c = gaussian_filter1d(ipis, 200)
    ts = np.linspace(0, 1, len(ipis_c))
    ax.plot(ts, ipis_c, color = cols[iname], ls = '-', lw = 1)
    
ax.axhline(700, ls = '-', color = 'k')
ax.set_xlabel('time (normalized)')
ax.set_ylabel('IPI (ms)')
ax.set_xticks([])
ax.set_yticks([600, 700, 800])
ax.set_xlim(0, 1)

print('mean trials per day:', np.mean(trials))

#### plot autocorrelation ####

ax = fig.add_subplot(gs[0, 1])
#ax.text(-0.20, 1.20, 'B', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')

bins = np.arange(1/200, 10, 1/12)
binned_autocors = np.zeros((len(names), len(bins)-1))
step = 10 #consider 10 trial intervals
for iname, name in enumerate(names):
    ipis = all_ipis[iname]

    
    inds = autocor_inds[iname] #indices for binning
    autocor = autocors[iname] #autocorrelation
    
    autocor_c = gaussian_filter1d(autocor, 1)
    xs = inds / trials[iname] #normalize by average trial count
    ax.plot(xs, autocor_c, color = cols[iname], ls = '--', lw = 1, alpha = 0.7)
    
    binned_autocors[iname, :] = binned_statistic(xs, autocor, 'mean', bins = bins)[0] #binning for taking mean
    

m, s = [f(binned_autocors, axis = 0) for f in [np.nanmean, np.nanstd]]
s /= np.sqrt(np.sum(1-np.isnan(binned_autocors), axis = 0))

ax.plot(bins[:-1], m, color = 'k')
ax.fill_between(bins[:-1], m-s, m+s, color = 'k', alpha = 0.2)
#plt.axvline(1, ls = ':', color = 'k')
ax.axhline(0, color = 'k', ls = ':')
ax.set_xlim(0, 2)
ax.set_ylabel('autocorrelation')
ax.set_xlabel('time (pseudo days)')
ax.set_ylim(-0.02, 0.25)

### save figure ###
plt.text(0.07, 1.25, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.47, 1.25, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)


#plt.savefig('../paper_figs/Sfig_ipi.png', bbox_inches = 'tight')    
plt.savefig('../paper_figs/Sfig_ipi.png', bbox_inches = 'tight', dpi = png_dpi)
plt.savefig('../paper_figs/Sfig_ipi.pdf', bbox_inches = 'tight')
#plt.show()
plt.close()