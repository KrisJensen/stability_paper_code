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
from plot_utils import get_col, col_blue, col_green
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

fig = plt.figure(figsize = (16*cm, 3*cm))

gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=1., wspace = 0.25, hspace = 0.20)

### plot for DLS and MC ###
names = [['Hindol', 'Dhanashri', 'Jaunpuri'],['Hamir','Gorakh','Gandhar']]
for igroup, group in enumerate(names):
    
    zs = [[], []] #twotap, wds
    for iname, name in enumerate(group):
        ### twotap
        for itask, taskname in enumerate(['', '_wds']):
        
            rat = pickle.load(open('../data/'+name+taskname+'_data_warped.p', 'rb'))
            for u in rat['units'].values():
                newrast = [d['peth_w_t'] for d in u.values()]
                if len(newrast) > 0:
                    newrast = np.concatenate(newrast, axis = 0)
                    newrast = np.sum(newrast, axis = 0)
                    if np.sum(newrast) > 50: #more than 50 total spikes
                        zs[itask].append(np.amax( np.abs(newrast-np.mean(newrast))/np.std(newrast) ))
                
    ax = fig.add_subplot(gs[0, igroup])
    
    bins = np.linspace(1, 6, 501)
    h1, h2 = [np.histogram(z, bins = bins)[0] for z in zs]
    h1, h2 = [h/np.sum(h)/(bins[1] - bins[0]) for h in [h1, h2]]
    c1, c2 = [gaussian_filter1d(h, 0.15/(bins[1]-bins[0])) for h in [h1, h2]]
    zvals = 0.5*(bins[1:] + bins[:-1])
    ax.plot(zvals, c1, color = 'k', ls = '-')
    ax.plot(zvals, c2, color = col_blue, ls = '-')
    ax.set_xlabel('modulation (z score)', labelpad = -10)
    ax.set_ylabel('frequency (a.u.)')
    ax.set_yticks([])
    ax.set_ylim(0, 1.1*np.amax(c2))
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks([bins[0], bins[-1]])
    
    if igroup == 0:
        ax.set_title('DLS', fontsize = plt.rcParams['font.size'])
        ax.legend(['lever', 'WDS'], frameon=False, fontsize = 10)
    else:
        ax.set_title('motor cortex', fontsize = plt.rcParams['font.size'])
            

### save figure ###
plt.text(0.12, 1.25, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.49, 1.25, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/ext_data_fig10.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close()