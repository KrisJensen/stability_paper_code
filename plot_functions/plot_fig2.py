#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, ttest_ind
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 11}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi, col_un, col_stab, global_cmap
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

fig = plt.figure(figsize = (16*cm, 8.5*cm))
sub = 1


###tf 2.7

### best: [18, 17, 23, 26]
### good ones: [9, 12, 13, 15, 17, 18]

nexs = [15,17,18]

nexs = [47,23,15]

### plot behavioral responses ###


data_iid = pickle.load(open('../results/rnn/iid_analyses.p', 'rb'))


gs = fig.add_gridspec(1, 1, left=0.00, right=0.28, bottom=0.67, top=1., wspace = 0.25, hspace = 0.20)
y  = data_iid['y']
inds = np.arange(12, 238)
vel = np.mean(y[..., inds], axis = 0)
std = 1*np.std(y[..., inds], axis = 0)
ts = np.arange(len(inds)) / (len(inds)-1)

ax = fig.add_subplot(gs[0, 0])
for i in range(3):
    ax.plot(ts, vel[i, :], lw = 1, color = 'k')
    ax.fill_between(ts, vel[i, :] - std[i, :], vel[i, :]+std[i, :], alpha = 0.2, color = 'k')
ax.set_xlabel('trial time (a.u.)')
ax.set_ylabel('output (a.u.)')
ax.set_yticks([])


### plot 3 example rasters and PETHs ####
gs = fig.add_gridspec(3, 2, left=0.34, right=0.73, bottom=0.55, top=1., wspace = 0.25, hspace = 0.20)
ex_spikes, ex_peths = data_iid['ex_spikes'], data_iid['ex_peth']
tmax = ex_spikes.shape[-1]-1

cols = ['k', np.array([100, 55, 0])/256, np.array([0, 100, 150])/256]
cols = [[0,0,0], [0.4, 0.4, 0.4], [0.7, 0.7, 0.7]]
for i_n, n in enumerate(nexs):
    ax = fig.add_subplot(gs[i_n, 0])
    for t in range(0, 100, sub):
        rast = np.nonzero(ex_spikes[t, n, :])[0][::sub]
        ax.scatter(rast, -np.ones(len(rast))*t, color = cols[i_n], s = 2, alpha = 1,  marker='o', linewidths=0.)
    if i_n == 0:
        ax.set_title('simulated units', fontsize = plt.rcParams['font.size'])
    elif i_n == 2:
        ax.set_xlabel('trial time')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('trial', labelpad = -0.5)
    ax.set_xlim(0, tmax)
    
ax = fig.add_subplot(gs[:, 1])
ts = np.arange(ex_peths.shape[-1])
for i_n, n in enumerate(nexs):
    ax.plot(ts, ex_peths[n, :], color = cols[i_n], ls = '-')
ax.set_title('simulated PETHs', fontsize = plt.rcParams['font.size'])
ax.set_xticks([])
ax.set_xlabel('trial time')
ax.set_ylabel('activity', labelpad = -0.5)
ax.set_yticks([])
ax.set_xlim(0, tmax)

### plot full heatmap of rasters
gs = fig.add_gridspec(1, 1, left=0.81, right=1.00, bottom=0.53, top=1., wspace = 0.25, hspace = 0.20)

plot_peths = np.array([ex_spikes[0::2, ...], ex_spikes[1::2, ...]])
plot_peths = np.sum(plot_peths, axis = 1) #across trials; 2xNxt
plot_peths = plot_peths[:, np.sum(plot_peths, axis = (0,-1)) >= 100, :]

ex_zs = np.array([np.mean(plot_peths[..., 5*i:5*(i+1)], axis = -1) for i in range(45)]) #45x2xN
ex_zs = np.transpose(ex_zs, (1,2,0)) #2xNx45
ex_zs = (ex_zs - np.mean(ex_zs, axis = -1)[..., None]) / np.std(ex_zs, axis = -1)[..., None]
plot_zs = ex_zs[0, np.argsort(np.argmax(ex_zs[1, ...], axis = 1)), :] #sort according to held-out trials

ax = fig.add_subplot(gs[0, 0])

ax.imshow(plot_zs, cmap = global_cmap, aspect = 'auto', interpolation = 'none', vmin = -2.5, vmax = 2.5)
ax.set_xlabel('trial time')
ax.set_xticks([])
ax.set_yticks([1, plot_zs.shape[0]])
ax.set_yticklabels([plot_zs.shape[0], 1])
ax.set_ylabel('simulated unit', labelpad = -15)

### plot example rasters (2 stable 2 drifting)###
data_s, data_d = [pickle.load(open('../results/rnn/interp_analyses_'+driftstr+'.p', 'rb')) for driftstr in ['stable', 'drift']]

shift = -0.25
gs = fig.add_gridspec(1, 5, left=0.25+shift, right=0.55+shift, bottom=-0.02, top=0.36, wspace = 0.15, hspace = 0.20, width_ratios = [1,1,0.2,1,1])

iplot = -2
for idat, data in enumerate([data_s, data_d]):
    ex_spikes = data[0]['example_spikes']
    ninterp = ex_spikes.shape[0]
    tmax = ex_spikes.shape[-1]-1
    iplot += 1
    for i_n, n in enumerate(nexs[:2]):
        iplot += 1
        ax = fig.add_subplot(gs[0, iplot])
        yval = 1
        for iday in range(ninterp):
            frac = iday/(ninterp-1)
            if idat == 0:
                col = frac * np.array([0, 1, 1]) + (1-frac)*np.array([0, 0.5, 0.5])
                col = np.array(col_stab) * (0.5*(1-frac)+0.5)
            else:
                col = frac * np.array([1, 0, 1]) + (1-frac)*np.array([0.5, 0, 0.5])
                col = np.array(col_un) * (0.5*(1-frac)+0.5)

            for itrial in range(0, 100, sub):
                yval -= 1
                rast = ex_spikes[iday, itrial, n, :]
                rast = np.nonzero(rast)[0][::sub]
                ax.scatter(rast, np.ones(len(rast))*yval, color = col, s = 2, alpha = 1,  marker='o', linewidths=0.)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, tmax)
        if iplot == 0:
            ax.set_ylabel('time (days)', labelpad = -10)
            
            yday = yval/ninterp
            ax.set_yticks([yday/2, yval - yday/2])
            ax.set_yticklabels(['1', '7'])
        ax.set_ylim(yval, 0)
        
        ax.set_title(['unit A', 'unit B'][i_n], fontsize = plt.rcParams['font.size'], pad = 0.5*5)

shift += 0.2
plt.text(0.12+shift, 0.43, 'stable',ha='center', va='bottom',transform=fig.transFigure)
plt.text(0.2825+shift, 0.43, 'drifting',ha='center', va='bottom',transform=fig.transFigure)
con = ConnectionPatch(xyA=(0.055+shift, 0.42), xyB=(0.185+shift, 0.42), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)
con = ConnectionPatch(xyA=(0.2175+shift, 0.42), xyB=(0.3475+shift, 0.42), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)

### plot correlation of activity/behavioral/latent space ###
gs = fig.add_gridspec(1, 3, left=0.60-0.23, right=1.00, bottom=0.05, top=0.35, wspace = 0.15, hspace = 0.20)
data_s, data_d = [pickle.load(open('../results/rnn/interp_analyses_rep_'+driftstr+'.p', 'rb')) for driftstr in ['stable', 'drift']]

labs = ['behav', 'neural', 'CCA']
titles = ['behavior', 'PETH', 'latent']
bins = np.arange(0.5, 7.5, 1)
bin_dts = 0.5*(bins[1:] + bins[:-1])
for itype, lab in enumerate(labs):
    ax = fig.add_subplot(gs[0, itype])
    for idata, data in enumerate([data_s, data_d]):
        sims = [data[key][lab+'_sims'] for key in data.keys()]
        dts = [data[key][lab+'_dts'] for key in data.keys()]
        bin_sims = np.zeros((len(sims), len(sims[0]), len(bin_dts)))*np.nan
        for rep in range(len(sims)): #for each repeat
            for n in range(len(sims[rep])): #for each pair of networks
                dt, sim = dts[rep][n], sims[rep][n]
                if len(dt) >= 1:
                    bin_sims[rep, n, :] = binned_statistic(dt, sim, bins = bins)[0]
        bin_sims = np.nanmean(bin_sims, axis = 1) #average over 'output units' (neurons/output dims/neuron groups)
        m, s = np.nanmean(bin_sims, axis = 0), np.nanstd(bin_sims, axis = 0)
        s = s

        label = (['stable', 'drifting'][idata] if itype == 0 else None)
        c = [col_stab, col_un][idata]
        ax.plot(bin_dts, m, color = c, ls = ['-', '--'][idata], label = label)
        ax.fill_between(bin_dts, m-s, m+s, color = c, alpha = 0.2)
        
    ax.set_xlabel(r'$\Delta$'+'time (d.)', labelpad = -2)#, fontsize = 10)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1,3,5])
    ax.set_xlim(0, 6)
    ax.set_title(titles[itype], fontsize = plt.rcParams['font.size'])
    if itype == 0:
        ax.set_ylabel('similarity', labelpad = -10)
        ax.set_yticks([0, 1])
    else:
        ax.set_yticks([])
    if itype == 0:
        ax.legend(frameon = False, handletextpad=0.5, handlelength=1.2, loc = 'lower center')

        
plt.text(-0.03, 1.09, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.30, 1.09, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.76, 1.09, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.03, 0.49, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.34, 0.44, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
        
plt.savefig('../paper_figs/main_fig2.pdf', bbox_inches = 'tight')
plt.close()

