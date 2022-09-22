#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, linregress
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi, get_col
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54
fig = plt.figure(figsize = (16*cm, 11*cm))
gs = fig.add_gridspec(2, 8, left=0, right=1, bottom=0.45, top=1., wspace = 0.35, hspace = 0.60, width_ratios = [1,1,1,1,1,1,0.05,1.5])

### plot all behavior and a histogram of the CIs. Each row is an experiment ###

names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh']
cols = [[0, 0, 1], [0, 0, 0.8], [0, 0, 0.6], [1, 0, 0], [0.8, 0, 0], [0.6, 0, 0]]
cols = [get_col(name) for name in names]

for iwds, wds in enumerate([False, True]):
    if wds:
        data = pickle.load(open('../results/behav_similarity_analyses_wds.p', 'rb'))
    else:
        data = pickle.load(open('../results/behav_similarity_analyses_twotap.p', 'rb'))

    allcorrs = []
    rawcorrs = []
    bins = np.linspace(-1, 1, 1501)
    for iname, name in enumerate(names):
        if wds:
            name += '_wds'
        
        ax = fig.add_subplot(gs[iwds, iname])
        dts, sims = data[name]['dts'], data[name]['sims']
        ax.scatter(dts, sims, color = cols[iname], marker = 'o', s = 1, alpha = 0.05)
        s, i = linregress(dts, sims)[:2]
        xs = np.array([np.amin(dts), np.amax(dts)])
        ax.plot(xs, i+s*xs, ls = '-', color = 'k')#cols[iname])
        ax.set_ylim(0, 1)
        ax.set_xlim(xs[0], xs[1])
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize = 10)
        ax.set_yticks([])
        ax.set_xlabel(r'$\Delta$'+'time (d.)', labelpad = -1)
        if iname == 0:
            ax.set_yticks([0, 1])
            ax.set_ylabel('corr.', labelpad = -10)

        newcorrs = data[name]['corrs']
        h = np.histogram(newcorrs, bins=bins)[0].astype(float)
        c = gaussian_filter1d(h, 5, mode = 'nearest')
        allcorrs.append(c)
        rawcorrs.append(newcorrs)
    
    ax = fig.add_subplot(gs[iwds, -1])
    print('\nwds:', wds)
    for iname in range(len(names)):
        print(names[iname]+': p='+str(np.nanmean(rawcorrs[iname] >= 0.)))
        plt.plot(bins[1:], allcorrs[iname], color = cols[iname], ls = '-', lw = 1)
    ax.set_xlim(-0.7, 0.1)
    ax.axvline(0, ls = '-', color = 'k')
    ax.set_yticks([])
    xs = [-0.5, 0]
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, fontsize = 10)
    ax.set_xlabel('correlation', labelpad = -1)
    ax.set_ylabel('frequency')

### plot performance!!! ###
ipi_data = pickle.load(open('../results/ipi_data.p', 'rb'))
all_ipis = ipi_data['ipis']
names = ipi_data['names']
trials = ipi_data['trials']
autocors = ipi_data['autocors']
autocor_inds = ipi_data['autocor_inds']

gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=0.27, wspace = 0.5, hspace = 0.20)

ax = fig.add_subplot(gs[0, 0])
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

#### plot autocorrelation ####
ax = fig.add_subplot(gs[0, 1])

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
ax.axhline(0, color = 'k', ls = ':')
ax.set_xlim(0, 2)
ax.set_ylabel('autocorrelation')
ax.set_xlabel('time (pseudo days)')
ax.set_ylim(-0.02, 0.25)

plt.text(-0.05, 1.06, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.80, 1.06, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.07, 0.33, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.46, 0.33, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/ext_data_fig9.jpg', bbox_inches = 'tight', dpi = png_dpi)    
plt.close()

