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

fig = plt.figure(figsize = (16*cm, 7*cm))
gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.62, top=1., wspace = 0.5, hspace = 0.20)

### plot model fits for some example neurons ###
data = pickle.load(open('../results/neural_similarity_analyses_twotap.p', 'rb'))

names = ['Hindol']
units = [[727, 269, 168, 138]]
cols = ['g', 'b', 'c', 'm']
ax = fig.add_subplot(gs[0, 0])
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
        simhat = fit.x[0]*np.exp(fit.x[1]*dt)
        ax.plot(dt, simhat, color = cols[iplot], ls = '-')
        
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 1])
ax.set_ylabel('similarity', labelpad = -10)
ax.set_xlabel('time difference (days)')

### plot distribution of model errors ###
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 0.5, 21)
regions = ['MC', 'DLS']
for ireg, region in enumerate(regions):
    errors = np.array(data[region]['errors'])
    mean_errors = np.array(data[region]['mean_errors'])
    ax.hist(errors, color = get_col(region), bins = bins, alpha = 0.3)
    qs = np.quantile(errors, [0.25, 0.5, 0.75])
    for iq, q in enumerate(qs):
        ax.axvline(q, ls = ':', color = get_col(region), label = (region if iq == 0 else None))
ax.set_xlabel('mean error')
ax.set_ylabel('frequency')
ax.legend(frameon = False)

### plot all stability indices ###
gs = fig.add_gridspec(1,4, left=0, right=1.0, bottom=0.0, top=0.38, wspace = 0.25, hspace = 0.1)
for iwds, wds in enumerate([False, True]): #repeat for WDS
    if wds:
        dataname = '../results/neural_similarity_analyses_wds.p'
    else:
        dataname = '../results/neural_similarity_analyses_twotap.p'
        
    #%% add similarity vs time analysis
    data = pickle.load(open(dataname, 'rb'))

    for ireg, region in enumerate(['DLS', 'MC']):
            if wds: region += '_wds'

            x, y, s = [data[region][k] for k in ['bin_x_dur', 'bin_y_dur', 'bin_s_dur']] #binned data
            res, yhat = data[region]['fit_dur'], data[region]['bin_yhat_dur']#model fit
            alphas, dts, rec_times, inds = [data[region][k] for k in ['alphas', 'dts', 'rec_times', 'alpha_inds']]
            rec_times = np.array(rec_times)[np.array(inds)]
            ax = fig.add_subplot(gs[0, 2*iwds + ireg])
            ax.scatter(rec_times, alphas, color = 'k', marker = 'o', s = 10, alpha = 0.5)
            boot_vals = data[region]['fit_boot_vals_dur']
            q1, q2, q3 = np.nanquantile(boot_vals, [0.25, 0.50, 0.75], axis = 0)
            xs = np.linspace(np.amin(rec_times), np.amax(rec_times), 101)
            ys = -np.abs(res[0]) - np.abs(res[1])*np.exp(-np.abs(res[2])*xs)
            ax.plot(xs, ys, color = get_col(region), ls = '-')

            ax.axhline(0, color = 'k', lw = 1)
            ymin, ymax = -0.61, 0.61
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(r'$\delta t$ (days)', labelpad = -2)
            if ireg== 0 and iwds == 0:
                ax.set_ylabel('stability index', labelpad = -0)
            else:
                ax.set_yticks([])

plt.text(0.12, 1.08, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.49, 1.08, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.1, 0.45, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.49, 0.45, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

plt.savefig('../paper_figs/ext_data_fig6.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close()