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
from plot_utils import panel_font, png_dpi, get_col, col_stab, col_un, global_cmap
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

fig = plt.figure(figsize = (16*cm, 9*cm))

gs = fig.add_gridspec(1, 4, left=0, right=0.50, bottom=0.59, top=1., wspace = 0.1, hspace = 0.20, width_ratios = [1, 0.05, 1, 0.07])

### plot example z scored kinematics ###
labels = [('paw_L', 0), ('paw_L', 1)]
name = 'Dhanashri'
kin_type = 'vels_w' #kinematics_w/vels_w
rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))

d0 = 0
d1 = d0+44

days = np.sort(list(rat['trials'].keys()))[d0:d1]
inds = np.arange(48, 157)
for ilab, label in enumerate(labels):
    trajecs = [rat['trials'][day][kin_type][label[0]][..., label[1]] for day in days]

    newdays = [days[j] for j in range(0, len(days)) if len(trajecs[j]) > 0]
    trajec = [np.array(traj)[:, inds] for traj in trajecs if len(traj) > 0]
    plottrajec = np.concatenate(trajec)
    plottrajec = (plottrajec - np.nanmean(plottrajec, axis=0)) / np.std(plottrajec, axis=0)

    ax = fig.add_subplot(gs[0, 2*ilab])
    vmin, vmax = -1, 1
    vmin, vmax = -0.75, 0.75

    im = ax.imshow(plottrajec, cmap=global_cmap, aspect='auto', vmin=vmin, vmax = vmax, interpolation = 'antialiased')
    ax.set_xticks(np.linspace(plottrajec.shape[1]/9, plottrajec.shape[1], 5)-0.5)
    ax.set_xticklabels(np.round(np.linspace(0.0, 0.8, 5), 1).astype(str))
    ax.set_yticks([0, len(plottrajec)])
    ax.set_yticklabels([1, newdays[-1]-newdays[0]+1])
    ax.set_xlabel('time (s)', labelpad=5)
    ax.set_ylabel('day (expert rat)', labelpad=-15)
    
    if label[1] == 1:
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.set_title('vertical', fontsize = plt.rcParams['font.size'])
    else:
        ax.set_title('horizontal', fontsize = plt.rcParams['font.size'])

newax = fig.add_subplot(gs[0, 3])
newax.set_xticks([])
newax.set_yticks([])
cbar = plt.colorbar(im, orientation = 'vertical', ax = newax, fraction = 1.0, shrink = 1, ticks=[vmin, vmax])
cbar.ax.set_yticklabels([str(vmin), str(vmax)], rotation = 0, va = 'center', ha = 'left')

### plot similarity vs time ###

gs = fig.add_gridspec(1, 1, left=0.64, right=1.0, bottom=0.58, top=1., wspace = 0.15, hspace = 0.20)

data = pickle.load(open('../results/behav_similarity_analyses_twotap.p', 'rb'))
dts, sims, day1s = [data[name][k] for k in ['dts', 'sims', 'day1s']]

bins = np.arange(0.5, np.amax(dts)+1, 1)
m = binned_statistic(dts, sims, statistic = 'mean', bins = bins)[0]
s = binned_statistic(dts, sims, statistic = 'std', bins = bins)[0]
xs = (bins[1:]+bins[:-1]) / 2

ax = fig.add_subplot(gs[0, 0])
ax.plot(xs, m, 'k-')
ax.fill_between(xs, m-s, m+s, color = 'k', alpha = 0.2)
ax.set_xlabel('time difference (days)')
ax.set_xticks([1, 10, 20, 30])
ax.set_xlim(xs[0], 30)
ax.set_ylim(0.5, 1)
ax.set_yticks([0.5, 1])
ax.set_ylabel('velocity corr.', labelpad = -10)


### plot neural vs behavioral correlation ###
gs = fig.add_gridspec(1, 6, left=0.03, right=1.0, bottom=0.00, top=0.34, wspace = 0.25, hspace = 0.15, width_ratios = [0.90, 0.02, 0.90, 0.01, 1, 1])

#### plot example similarity ####
res = pickle.load(open('../results/neural_behav_corr_DLS.p', 'rb'))
res_mc = pickle.load(open('../results/neural_behav_corr_MC.p', 'rb'))

all_sims_n, all_sims_b = res['all_sims_n'], res['all_sims_b']
all_corrs = [pearsonr(all_sims_n[i], all_sims_b[i])[0] for i in range(len(all_sims_n))]

all_sims_n_mc, all_sims_b_mc = res_mc['all_sims_n'], res_mc['all_sims_b']
all_corrs_mc = [pearsonr(all_sims_n_mc[i], all_sims_b_mc[i])[0] for i in range(len(all_sims_n_mc))]

Ls = np.array([len(sim) for sim in all_sims_n])
ind = np.argsort(-Ls)[0] #4
ex_sim_n, ex_sim_b = all_sims_b[ind], all_sims_n[ind]
print('example neural-behavioral correlation:', pearsonr(ex_sim_n, ex_sim_b))

ax = fig.add_subplot(gs[0, 0])
ax.scatter(ex_sim_n, ex_sim_b, s = 30, c = 'k')
ax.set_xlabel('kinematic similarity')
ax.set_yticks([0.9, 1.0])
ax.set_xticks([0.8, 0.9, 1.0])
ax.set_xlim(0.75, 1.0)
ax.set_ylabel('neural similarity', labelpad = 0)


### plot full distribution

boot = [np.mean(np.random.choice(all_corrs, len(all_corrs), replace = True)) for _ in range(1000)]
ax = fig.add_subplot(gs[0, 2])
h, _, _ = ax.hist(all_corrs, bins = np.linspace(-1, 1, 11), color = get_col('DLS'), alpha = 0.4)
maxval = np.nanmax(h)*1.05
ax.axvline(np.mean(all_corrs), color = get_col('DLS'), label = 'DLS')

h_mc, _, _ = ax.hist(all_corrs_mc, bins = np.linspace(-1, 1, 11), color = get_col('MC'), alpha = 0.4)
ax.axvline(np.mean(all_corrs_mc), color = get_col('MC'), label = 'MC')

ax.set_ylim(0, maxval)
ax.set_xlabel('correlation')
ax.set_ylabel('frequency', labelpad = -10)
ax.set_yticks([0, 20])
ax.legend(frameon = False, fontsize = 10, handlelength = 1.2, handletextpad = 0.4, loc = 'upper left', borderpad = 0.2)

#### plot mean vs. synthetic/ctrl distributions ####

for ireg, region in enumerate(['DLS', 'MC']):
    res = pickle.load(open('../results/neural_behav_corr_syn_'+region+'.p', 'rb'))
    means, means_syn, all_corrs = res['means'], res['means_syn'], res['all_corrs']
    
    minval, maxval = [func(np.concatenate([means, means_syn])) for func in [np.amin, np.amax]]
    bins = np.linspace(minval, maxval, 50)
    
    col_un = [0,0,0]
    col_stab = [0.5, 0.5, 0.5]

    ax = fig.add_subplot(gs[0, ireg+4])
    ax.hist(means_syn, bins = bins, color = col_stab, alpha = 0.7)
    ax.hist(means, bins = bins, color = col_un, alpha = 0.7)
        
    ax.axvline(np.mean(all_corrs), color = get_col(region), lw = 2)
    ax.axvline(np.mean(means), color = col_un, lw = 2, ls ='--')
    ax.axvline(np.mean(means_syn), color = col_stab, lw = 2, ls = '--')
    ax.set_xlabel('mean correlation')
    
    if ireg == 0:
        ax.plot([], [], ls = '-', color = get_col('MC'))
        leg = ['DLS', 'ctrl', 'syn', 'MC']
        ax.legend(leg, frameon = False, ncol = len(leg), bbox_to_anchor = (-0.05, 1.30), loc = 'upper left',
                   handlelength = 1.2, handletextpad = 0.4, columnspacing = 1.0)
        ax.set_ylabel('frequency')
    ax.set_yticks([])

    print(region+' mean corr = '+str(np.round(np.mean(all_corrs), 3)))
    print(region+' p ctrl = '+str(np.mean(np.array(means) >= np.mean(all_corrs))))
    print(region+' mean synthetic = '+str(np.round(np.mean(means_syn), 3)))
    print(region+' p synthetic = '+str(np.mean(np.array(means_syn) <= np.mean(all_corrs))))

    
plt.text(-0.06, 1.07, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.57, 1.07, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.06, 0.42, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.25, 0.42, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.50, 0.42, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
     
plt.savefig('../paper_figs/main_fig5.pdf', bbox_inches = 'tight')
plt.close()

