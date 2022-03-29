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

gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=1., wspace = 0.5, hspace = 0.20)


### plot model fits for some example neurons ###
data = pickle.load(open('../results/neural_similarity_analyses_twotap.p', 'rb'))

names = ['Hindol']
units = [[727, 269, 168, 138]]
cols = ['g', 'b', 'c', 'm']
ax = fig.add_subplot(gs[0, 0])
#ax.text(-0.20, 1.20, 'A', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
iplot = -1
for iname, name in enumerate(names):
    unums, dts, sims, fits, alpha_inds = [data[name][k] for k in ['unums', 'dts', 'sims', 'fits', 'alpha_inds']]
    for unum in units[iname]:
        iplot += 1
        ind = unums.index(unum) #index for dt, similarity
        alpha_ind = list(alpha_inds).index(ind) #index for alpha, fit
        dt, sim = np.array(dts[ind]), np.array(sims[ind])
        fit = fits[alpha_ind]
        ax.scatter(dt, sim, color = cols[iplot], marker = 'x', s = 20)
        dif = np.amax(dt)-np.amin(dt)
        #dt = np.concatenate([dt, dt+dif+1, dt+2*dif+1])
        simhat = fit.x[0]*np.exp(fit.x[1]*dt)
        ax.plot(dt, simhat, color = cols[iplot], ls = '-')
        #print(alpha_ind, np.mean(np.abs(sim - simhat)), data[name]['types'][unum])
        
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 1])
ax.set_ylabel('similarity', labelpad = -10)
ax.set_xlabel('time difference (days)')

### plot distribution of model errors ###
ax = fig.add_subplot(gs[0, 1])
#ax.text(-0.30, 1.20, 'B', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
bins = np.linspace(0, 0.5, 21)
regions = ['MC', 'DLS']
for ireg, region in enumerate(regions):
    errors = np.array(data[region]['errors'])
    mean_errors = np.array(data[region]['mean_errors'])
    #errors = errors/mean_errors
    ax.hist(errors, color = get_col(region), bins = bins, alpha = 0.3)
    qs = np.quantile(errors, [0.25, 0.5, 0.75])
    for iq, q in enumerate(qs):
        ax.axvline(q, ls = ':', color = get_col(region), label = (region if iq == 0 else None))
    #print(qs)
ax.set_xlabel('mean error')
ax.set_ylabel('frequency')
ax.legend(frameon = False)

plt.text(0.12, 1.20, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.49, 1.20, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

#plt.savefig('../paper_figs/Sfig_example_fits.png', bbox_inches = 'tight')    
plt.savefig('../paper_figs/Sfig_example_fits.png', bbox_inches = 'tight', dpi = png_dpi)
plt.savefig('../paper_figs/Sfig_example_fits.pdf', bbox_inches = 'tight')
#plt.show()
plt.close()