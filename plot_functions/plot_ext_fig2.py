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
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi, global_cmap
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

### plot example z scored kinematics ###
labels = [('paw_R', 0), ('paw_R', 1), ('paw_L', 0), ('paw_L', 1)]
names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh']

fig = plt.figure(figsize = (16*cm, 20*cm))
for iz, zscore in enumerate([False, True]):

    if iz == 0:
        top, bot = 1.0, 0.55
    else:
        top, bot = 0.45, 0.00
    gs = fig.add_gridspec(4, 6, left=0, right=1.0, bottom=bot, top=top, wspace = 0.45, hspace = 0.30)
    
    for iname, name in enumerate(names):
        kin_type = 'vels_w' if zscore else 'kinematics_w' #kinematics_w/vels_w
        rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))

        days = np.sort(list(rat['trials'].keys()))
        inds = np.arange(48, 157)
        for ilab, label in enumerate(labels):
            trajecs = [rat['trials'][day][kin_type][label[0]][..., label[1]] for day in days]

            newdays = [days[j] for j in range(0, len(days)) if len(trajecs[j]) > 0]
            trajec = [np.array(traj)[:, inds] for traj in trajecs if len(traj) > 0]
            plottrajec = np.concatenate(trajec)
            if zscore:
                plottrajec = (plottrajec - np.nanmean(plottrajec, axis=0)) / np.std(plottrajec, axis=0)
                vmin, vmax = -0.75, 0.75
                interpolation = 'antialiased'
            else:
                vmin, vmax = np.nanquantile(plottrajec, 0.01), np.nanquantile(plottrajec, 0.99)
                interpolation = 'none'

            ax = fig.add_subplot(gs[ilab, iname])
            im = ax.imshow(plottrajec, cmap=global_cmap, aspect='auto', vmin=vmin, vmax = vmax, interpolation = interpolation)

            ax.set_yticks([0, len(plottrajec)])
            ax.set_yticklabels([1, newdays[-1]-newdays[0]+1], fontsize = 10)
            ax.tick_params(axis='y', which='major', pad=-0.1)
            
            if ilab == len(labels)-1:
                ax.set_xlabel('time (s)', labelpad=2)
                ax.set_xticks(np.linspace(plottrajec.shape[1]/9, 8/9*plottrajec.shape[1], 2)-0.5)
                ax.set_xticklabels(np.round(np.linspace(0.0, 0.7, 2), 1).astype(str), fontsize = 10)
            else:
                ax.set_xticks([])

            if iname == 0:
                ax.set_ylabel('day', labelpad=-10)

plt.text(-0.05, 1.04, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.05, 0.49, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/ext_data_fig2.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close()

