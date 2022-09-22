#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, ttest_ind
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi, col_stab, col_un
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

fig = plt.figure(figsize = (16*cm, 3*cm))

### now plot similarity in parameter space ###
gs = fig.add_gridspec(1, 3, left=0, right=1.0, bottom=0.0, top=1.0, wspace = 0.25, hspace = 0.20)
data_d = [pickle.load(open('../results/rnn/reps/rep'+str(rep)+'_data_interp.pickled', 'rb')) for rep in range(10)]
data_s = [pickle.load(open('../results/rnn/reps/rep'+str(rep)+'_data_sinterp.pickled', 'rb')) for rep in range(10)]

labs = ['neural', 'behav', 'CCA']
titles = ['initial conditions', 'recurrent W', 'readout W']
bins = np.arange(0.5, 7.5, 1)
bin_dts = 0.5*(bins[1:] + bins[:-1])
for itype, lab in enumerate(labs): #for each parameter type
    ax = fig.add_subplot(gs[0, itype])
    for idata, data in enumerate([data_s, data_d]): #for stable and unstable
        params = [dat[itype] for dat in data] #extract parameters
        
        bins = np.arange(0.5, 7.5, 1)
        bin_dts = 0.5*(bins[1:] + bins[:-1])
        bin_sims = np.zeros((len(params), len(bin_dts)))*np.nan
        for rep, param in enumerate(params): #for each repeat
            sim = []
            dt = []
            for i1, p1 in enumerate(param):
                for i2 in range(i1+1, len(param)):
                    p2 = param[i2]
                    sim.append(pearsonr(p1.flatten(), p2.flatten())[0])
                    dt.append(i2-i1)
            bin_sims[rep, :] = binned_statistic(dt, sim, bins = bins)[0]
        m, s = np.nanmean(bin_sims, axis = 0), np.nanstd(bin_sims, axis = 0)
        s = s
        label = ['stable', 'drifting'][idata] if itype == 0 else None
        c = [col_stab, col_un][idata]
        ax.plot(bin_dts, m, color = c, ls = ['-', '--'][idata], label = label)
        ax.fill_between(bin_dts, m-s, m+s, color = c, alpha = 0.2)
        
    ax.set_xlabel('time difference')
    ax.set_ylim(0, 1.02)
    ax.set_title(titles[itype], fontsize = plt.rcParams['font.size'])
    if itype == 0:
        ax.set_ylabel('correlation', labelpad = -10)
        ax.set_yticks([0, 1])
        ax.legend(frameon = False, handletextpad=0.5, handlelength=1.5)
    else:
        ax.set_yticks([])

plt.text(-0.05, 1.25, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/ext_data_fig1.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close()

