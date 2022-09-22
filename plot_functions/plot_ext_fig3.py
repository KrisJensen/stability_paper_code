#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
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

fig = plt.figure(figsize = (12*cm, 18*cm))
gs = fig.add_gridspec(5, 4, left=0, right=1, bottom=0.0, top=1., wspace = 0.1, hspace = 0.2, width_ratios = [1, 0.15, 1, 0.1], height_ratios = [1,0.7, 0.3, 1,0.7])
for itype, type_ in enumerate(['twotap', 'wds']):

    if type_ == 'wds':
        data = pickle.load(open('../results/neural_similarity_analyses_wds.p', 'rb'))
        names = ['DLS_wds', 'MC_wds']
    else:
        data = pickle.load(open('../results/neural_similarity_analyses_twotap.p', 'rb'))
        names = ['DLS', 'MC']
    vmin, vmax = 0, 1

    for panel in range(2): # zoom in on >=14 days for second plot
        for iname, name in enumerate(names):
            ax = fig.add_subplot(gs[3*itype+panel, 2*iname])
            unums, dts, sims, rec_times = [data[name][k] for k in ['unums', 'dts', 'sims', 'rec_times']]
            max_dt = 22
            u_dt = np.arange(1, max_dt)
            vals = []

            if panel == 1: #only >= tmax days
                tmax = 14 if itype == 0 else 10
                inds = np.where(rec_times >= tmax)[0]
                unums, dts, sims, rec_times = [[arr[ind] for ind in inds] for arr in [unums, dts, sims, rec_times]]

            for ind in np.argsort(-np.array(rec_times)): #from longest to shortest
                dt, sim = np.array(dts[ind]), np.array(sims[ind])
                u_sim = np.array([np.mean(sim[dt == t]) if np.sum(dt == t) > 0 else np.nan for t in u_dt])

                if len(dt) > 0 and rec_times[ind] >= 2.5:
                    nans, x = np.isnan(u_sim[:np.amax(dt)]), lambda z: z.nonzero()[0]
                    u_sim[:np.amax(dt)][nans]= np.interp(x(nans), x(~nans), u_sim[:np.amax(dt)][~nans]) #interpolate nans
                    vals.append(u_sim)

            sims = np.array(vals)
            im = ax.imshow(sims, cmap = 'viridis', vmin = vmin, vmax = vmax, aspect = 'auto', interpolation = 'nearest')
            if panel == 0:
                ax.set_title(name.replace('_', ' '))
                ax.set_xticks([])
            else:
                ax.set_xlabel('time difference (days)')
            if iname == 0:
                ax.set_ylabel('neuron')

        if itype == 0 and panel == 0:
            newax = fig.add_subplot(gs[itype, 3])
            newax.set_xticks([])
            newax.set_yticks([])
            cbar = plt.colorbar(im, orientation = 'vertical', ax = newax, fraction = 1.0, shrink = 1, ticks=[vmin, vmax])
            cbar.ax.set_yticklabels([str(vmin), str(vmax)], rotation = 0, va = 'center', ha = 'left')
            cbar.ax.set_ylabel('mean correlation', rotation=270, labelpad = -3)

plt.text(-0.12, 1.05, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.12, 0.47, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/ext_data_fig3.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close()