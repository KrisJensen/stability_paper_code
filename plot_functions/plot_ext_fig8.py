#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, ttest_ind
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

dataname = '../results/neural_similarity_analyses_twotap.p'
fig = plt.figure(figsize = (16*cm, 3*cm))
gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=1., wspace = 0.25, hspace = 0.20)

from plot_utils import fit_model
data = pickle.load(open(dataname, 'rb'))
labels = ['DLS', 'MC']
for iname, name in enumerate(['DLS', 'MC']):
    dts, rec_times, sims, unums = [data[name][k] for k in ['dts', 'rec_times', 'sims', 'unums']]
    inds = (rec_times >= 14).nonzero()[0]
    bins = np.arange(0.5, 13.5+0.01, 1)
    long_sims = np.zeros((len(inds), len(bins)-1))
    for iind, ind in enumerate(inds):
        dt, sim = dts[ind], sims[ind]
        long_sims[iind, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]
    nans = np.isnan(long_sims)
    no_nans = np.array([(not any(nans[i, :])) for i in range(nans.shape[0])])
    long_sims = long_sims[no_nans, :]
    m, s = np.nanmean(long_sims, axis = 0), np.nanstd(long_sims, axis = 0)/np.sqrt(np.sum(1-np.isnan(long_sims), axis = 0))
    xs = (bins[1:] + bins[:-1])/2

    ax = fig.add_subplot(gs[0, iname])

    #plot fits to days
    max_dts = np.arange(len(xs)-2)+3
    for i in max_dts:
        v = (i - max_dts[0]) / (max_dts[-1] - max_dts[0])
        cmap = matplotlib.cm.get_cmap('viridis')
        col = cmap(0.5*v + 0.25)

        fit = fit_model(xs[:i], m[:i], baseline = False)
        ys = fit.x[0]*np.exp( fit.x[1]*xs )
        label = str(i)+' days' if i % 2 == 1 else None
        ax.plot(xs, ys, ls = '-', color = col, label = label, lw = 1)

    ax.plot(xs, m, 'k-')
    fit = fit_model(xs, m, baseline = True)
    ys = fit.x[0]*np.exp( fit.x[1]*xs ) + fit.x[2]
    ax.plot(xs, ys, ls = '--', color = get_col('MC'), lw = 2)

    ax.set_xlim(xs[0], xs[-1])
    ax.set_xticks([xs[0], xs[-1]])
    ax.set_xlabel('time difference (days)', labelpad = -10)
    if iname == 0:
        ax.set_ylabel('correlation', labelpad = 0)
        ax.legend(frameon = False, fontsize = 6.5, ncol = 2)
        ax.set_ylim(0.7,0.87)
    else:
        ax.set_ylim(0.50,0.8)

plt.text(0.13, 1.20, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.50, 1.20, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
  
plt.savefig('../paper_figs/ext_data_fig8.jpg', bbox_inches = 'tight', dpi = png_dpi) 
plt.close()