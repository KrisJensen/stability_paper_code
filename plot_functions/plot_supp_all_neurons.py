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

fig = plt.figure(figsize = (12*cm, 12*cm))
gs = fig.add_gridspec(2, 4, left=0, right=1, bottom=0.0, top=1., wspace = 0.1, hspace = 0.52, width_ratios = [1, 0.15, 1, 0.1])

for itype, type_ in enumerate(['twotap', 'wds']):

    if type_ == 'wds':
        data = pickle.load(open('../results/neural_similarity_analyses_wds.p', 'rb'))
        names = ['DLS_wds', 'MC_wds']
    else:
        data = pickle.load(open('../results/neural_similarity_analyses_twotap.p', 'rb'))
        names = ['DLS', 'MC']

    vmin, vmax = 0, 1

    for iname, name in enumerate(names):

        ax = fig.add_subplot(gs[itype, 2*iname])

        unums, dts, sims, rec_times = [data[name][k] for k in ['unums', 'dts', 'sims', 'rec_times']]
        max_dt = 22
        u_dt = np.arange(1, max_dt)
        vals = []
        for ind in np.argsort(-np.array(rec_times)): #from longest to shortest
            dt, sim = np.array(dts[ind]), np.array(sims[ind])
            u_sim = np.array([np.mean(sim[dt == t]) for t in u_dt])

            if len(dt) > 0 and rec_times[ind] >= 2.5:
                nans, x = np.isnan(u_sim[:np.amax(dt)]), lambda z: z.nonzero()[0]
                u_sim[:np.amax(dt)][nans]= np.interp(x(nans), x(~nans), u_sim[:np.amax(dt)][~nans]) #interpolate nans

                vals.append(u_sim)

        sims = np.array(vals)

        #sims = sims[ np.sum(1 - np.isnan(sims), axis = 1) >= 3.5, :]

        im = ax.imshow(sims, cmap = 'viridis', vmin = vmin, vmax = vmax, aspect = 'auto', interpolation = 'nearest')
        ax.set_xlabel('time difference (days)')
        ax.set_title(name.replace('_', ' '))

        if iname == 0:
            ax.set_ylabel('neuron')

    newax = fig.add_subplot(gs[itype, 3])
    newax.set_xticks([])
    newax.set_yticks([])
    cbar = plt.colorbar(im, orientation = 'vertical', ax = newax, fraction = 1.0, shrink = 1, ticks=[vmin, vmax])
    cbar.ax.set_yticklabels([str(vmin), str(vmax)], rotation = 0, va = 'center', ha = 'left')
    cbar.ax.set_ylabel('mean correlation', rotation=270, labelpad = -3)

plt.text(-0.12, 1.05, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.12, 0.45, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/Sfig_all_neurons.png', bbox_inches = 'tight', dpi = png_dpi)
plt.savefig('../paper_figs/Sfig_all_neurons.pdf', bbox_inches = 'tight')
# plt.show()
plt.close()