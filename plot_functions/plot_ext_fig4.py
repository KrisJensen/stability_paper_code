#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 07:45:50 2021
@author: kris
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, linregress
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

max_ts = [5,8,11,17]
fig = plt.figure(figsize = (16*cm, 7*cm))
gs = fig.add_gridspec(2,len(max_ts), left=0, right=1.0, bottom=0.0, top=1.00, wspace = 0.15, hspace = 0.6)

for iwds, wds in enumerate([False, True]):

    if wds:
        dataname = '../results/neural_similarity_analyses_wds.p'
    else:
        dataname = '../results/neural_similarity_analyses_twotap.p'

    r1, l2 = 0.51, 0.61

    #%% add similarity vs time analysis
    from plot_utils import fit_model

    data = pickle.load(open(dataname, 'rb'))

    for imax, max_t in enumerate(max_ts):
        ax = fig.add_subplot(gs[iwds, imax]) #fill the whole thing!

        labels = ['DLS', 'MC']
        for iname, name in enumerate(['DLS', 'MC']):
            if wds:
                name += '_wds'
            dts, rec_times, sims, unums = [data[name][k] for k in ['dts', 'rec_times', 'sims', 'unums']]

            inds = (rec_times >= max_t).nonzero()[0]
            bins = np.arange(0.5, max_t - 0.5 + 0.01, 1)
            long_sims = np.zeros((len(inds), len(bins)-1))
            for iind, ind in enumerate(inds):
                dt, sim = dts[ind], sims[ind]
                long_sims[iind, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]

            nans = np.isnan(long_sims)
            no_nans = np.array([(not any(nans[i, :])) for i in range(nans.shape[0])])
            long_sims = long_sims[no_nans, :]
            inds = inds[no_nans]

            m, s = np.nanmean(long_sims, axis = 0), np.nanstd(long_sims, axis = 0)/np.sqrt(np.sum(1-np.isnan(long_sims), axis = 0))
            xs = (bins[1:] + bins[:-1])/2
            ax.plot(xs, m, color = get_col(name), ls = '-', label = labels[iname])
            ax.fill_between(xs, m-s, m+s, color = get_col(name), alpha = 0.2)

            #get upper bound
            res = data[name]['resamples']
            long_sims = np.zeros((len(inds), len(res[unums[inds[0]]]['dts']), len(bins)-1))
            for iind, ind in enumerate(inds):
                unum = unums[ind]
                for i in range(len(res[unum]['dts'])):
                    dt, sim = np.array(res[unum]['dts'][i]), np.array(res[unum]['sims'][i])
                    long_sims[iind, i, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]
            m, s = np.nanmean(long_sims, axis = (0, 1)), np.nanstd(long_sims, axis = (0, 1))/np.sqrt(np.sum(1-np.isnan(long_sims), axis = (0, 1)))
            ax.plot(xs, m, color = get_col(name), ls = '--', alpha = 0.5)
            ax.fill_between(xs, m-s, m+s, color = get_col(name), alpha = 0.1)

            # get lower bound
            m = np.mean(data[name]['long_shuffled_sims'])
            s = np.std(data[name]['long_shuffled_sims'])/np.sqrt(len(data[name]['long_shuffled_sims']))
            ax.plot([xs[0], xs[-1]], np.ones(2)*m, color = get_col(name), ls = '--', alpha = 0.5)
            ax.fill_between([xs[0], xs[-1]], np.ones(2)*(m-s), np.ones(2)*(m+s), color = get_col(name), alpha = 0.1)
            ax.set_title('>= '+str(int(max_t))+' days', fontsize = font['size'])

        ax.set_xlim(xs[0], xs[-1])
        if wds:
            ax.set_ylim(-0.03, 0.7)
            yticks = [0, 0.7]
        else:
            ax.set_ylim(-0.03, 1)
            yticks = [0, 1]
        ax.set_xticks([xs[0], xs[-1]])
        ax.set_xlabel(r'$\delta t$ (days)', labelpad = -12)
        if imax == 0:
            ax.set_ylabel('correlation', labelpad = -10)
            if iwds == 0: ax.legend(frameon = False)
            ax.set_yticks(yticks)
        else:
            ax.set_yticks([])

plt.text(-0.05, 1.125, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.05, 0.5, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

#%% save fig
plt.savefig('../paper_figs/ext_data_fig4.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close() 