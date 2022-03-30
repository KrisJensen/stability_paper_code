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

fig = plt.figure(figsize = (16*cm, 3*cm))
gs = fig.add_gridspec(1,4, left=0, right=1.0, bottom=0.0, top=1.00, wspace = 0.25, hspace = 0.1)

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
            #ax.plot(x, q2, get_col(region)+'--')
            #ax.fill_between(x, q1, q3, color = get_col(region), alpha = 0.2)

            xs = np.linspace(np.amin(rec_times), np.amax(rec_times), 101)
            ys = -np.abs(res[0]) - np.abs(res[1])*np.exp(-np.abs(res[2])*xs)
            ax.plot(xs, ys, get_col(region)+'-')

            ax.axhline(0, color = 'k', lw = 1)

            ymin, ymax = -0.5, 0.5
            ax.set_ylim(ymin, ymax)
            #ax.set_xticks([xs[0], xs[-1]])
            ax.set_xlabel(r'$\delta t$ (days)', labelpad = -2)
            if ireg== 0 and iwds == 0:
                ax.set_ylabel('stability index', labelpad = -0)
                #ax.legend(frameon = False)
                #ax.set_yticks([0, 0.6]) if wds else ax.set_yticks([0, 1])
            else:
                ax.set_yticks([])


    plt.text(-0.10, 1.25, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    plt.text(0.48, 1.25, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    # plt.text(0.57, 1.07, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    #plt.text(0.81, 1.07, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    # plt.text(0.57, 0.63, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    # plt.text(0.81, 0.63, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    # plt.text(0.57, 0.28, 'F',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
        
    #%% save fig

    # if wds:
    #     plt.savefig('../paper_figs/Sfig_4G_wds.png', bbox_inches = 'tight', dpi = 100)
    # else:
    #     plt.savefig('../paper_figs/Sfig_4G.png', bbox_inches = 'tight', dpi = 100)
    # # plt.show()
    # plt.close()

plt.savefig('../paper_figs/Sfig_4G.png', bbox_inches = 'tight', dpi = png_dpi)
plt.savefig('../paper_figs/Sfig_4G.pdf', bbox_inches = 'tight')
plt.close()