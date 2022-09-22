#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:50:21 2021

@author: kris
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, linregress, ttest_ind
from plot_utils import get_col, fit_model, col_stab, col_un
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

fig = plt.figure(figsize = (16*cm, 10*cm))
dataname = '../results/neural_similarity_analyses_wds.p'
r1, l2 = 0.62, 0.73

#%% plot example kinematics
gs = fig.add_gridspec(1, 5, left=0.01, right=r1+0.05, bottom=0.69, top=1., wspace = 0.0, hspace = 0.20, width_ratios = [1, 0.32, 1, 0.45, 1])

labels = [('paw_L', 0), ('paw_L', 1)]
name = 'Dhanashri_wds'
kin_type = 'kinematics_w'
rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))

d0 = 0
d1 = len(rat['trials'].keys())

days = np.sort(list(rat['trials'].keys()))[d0:d1]
trajecs = [rat['trials'][day][kin_type]['acc'][..., 2] for day in days]

newdays = [days[j] for j in range(0, len(days)) if len(trajecs[j]) > 0]
trajec = [np.array(traj)[:, :] for traj in trajecs if len(traj) > 0]
plottrajec = np.concatenate(trajec)
vmin, vmax = np.nanquantile(plottrajec, 0.01), np.nanquantile(plottrajec, 0.99)
plottrajec -= vmin
plottrajec *= 100/(vmax-vmin)

T = plottrajec.shape[1]
ts = np.linspace(-0.2, 0.5, T)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(plottrajec, cmap=global_cmap, aspect='auto', vmin=0, vmax=100)
ax.set_yticks([0, len(plottrajec)])
ax.set_yticklabels([1, newdays[-1]-newdays[0]+1])
ax.set_xlabel('time (s)', labelpad = -0, fontsize = 10)
ax.set_xticks([-0.5, T/7*2-0.5, T/7*4.5-0.5, T-0.5])
ax.set_xticklabels(['-0.2', '0.0', '0.25', '0.5'], fontsize = 10)
ax.set_ylabel('day', labelpad=-15)

### compute and print warping coefficients ###
all_warps = []
for name in ['Hamir', 'Gorakh', 'Gandhar', 'Hindol', 'Dhanashri', 'Jaunpuri']:
    rat = pickle.load(open('../data/'+name+'_wds_data_warped.p', 'rb'))
    warps = [rat['trials'][day]['periods']/rat['targetint'] for day in rat['trials'].keys()]
    all_warps.append(np.concatenate(warps))
all_warps = np.concatenate(all_warps)
print('WDS warping coefficients:', np.mean(all_warps), np.std(all_warps), '\n')

#### plot mean #####
accs = []
rat = pickle.load(open('../data/Dhanashri_wds_data_warped.p', 'rb'))
for day in rat['trials'].keys():
    newaccs = rat['trials'][day]['kinematics_w']['acc'][..., 2]
    accs.append(np.mean(newaccs, axis = 0))
accs = np.array(accs)
m, s = np.mean(accs, axis = 0), np.std(accs, axis = 0)
ts = np.linspace(-0.2, 0.5, len(m))

ax = fig.add_subplot(gs[0, 2])
ax.plot(ts, m, 'k-')
ax.fill_between(ts, m-s, m+s, color = 'k', alpha = 0.2)
ax.set_xticks([-0.2, 0.0, 0.25, 0.5])
ax.set_xticklabels(['-0.2', '0.0', '0.25', '0.5'], fontsize = 10)
ax.set_xlim(-0.2, 0.5)
ax.set_xlabel('time (s)', labelpad = -0, fontsize = 10)
ax.set_ylabel('acceleration (a.u.)')
ax.set_yticks([])

#%% plot similarity vs time for >14 days
data = pickle.load(open(dataname, 'rb'))
ax = fig.add_subplot(gs[0, 4])
labels = ['DLS', 'MC']
for iname, name in enumerate(['DLS_wds', 'MC_wds']):
    dts, rec_times, sims, unums = [data[name][k] for k in ['dts', 'rec_times', 'sims', 'unums']]
    
    tmax = 10
    inds = (rec_times >= tmax).nonzero()[0]
    bins = np.arange(0.5, tmax-0.5+0.01, 1)
    long_sims = np.zeros((len(inds), len(bins)-1))
    for iind, ind in enumerate(inds):
        dt, sim = dts[ind], sims[ind]
        long_sims[iind, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]
        
    nans = np.isnan(long_sims)
    no_nans = np.array([(not any(nans[i, :])) for i in range(nans.shape[0])])
    long_sims = long_sims[no_nans, :]
    inds = inds[no_nans]

    m, s = np.nanmean(long_sims, axis = 0), np.nanstd(long_sims, axis = 0)/np.sqrt(np.sum(1-np.isnan(long_sims), axis = 0))
    print(name+' number of units with >= '+str(tmax)+' recording days:', long_sims.shape[0])
    
    xs = (bins[1:] + bins[:-1])/2
    ax.plot(xs, m, color = get_col(name), ls = '-', label = labels[iname])
    ax.fill_between(xs, m-s, m+s, color = get_col(name), alpha = 0.2)

    ## plot exponential fit
    fit = fit_model(xs, m, baseline = True)
    ys = fit.x[0] * np.exp( fit.x[1]*xs ) + fit.x[2]
    print('asymptotic baseline:', fit.x[2])

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

ax.set_xlim(xs[0], xs[-1])
ymin, ymax = 0. ,0.6
ax.set_ylim(ymin, ymax)
ax.set_yticks([ymin, ymax])
ax.set_xticks([xs[0], xs[-1]])
ax.set_xlabel(r'$\Delta$'+'time (d.)', labelpad = -12)
ax.set_ylabel('correlation', labelpad = -10)
ax.legend(frameon = False, loc = 'lower center')


#%% plot example rasters

names = ['Hindol', 'Dhanashri', 'Hamir']

units = [[132], [427], [256, 811]]

gs = fig.add_gridspec(2, 4, left=0, right=r1, bottom=0.0, top=0.47, wspace = 0.5, hspace = 0.1)
iplot = -1
tmin, tmax = -0.2, 0.5
subsamps = [1, 1, 1, 1]
ss = np.ones(4)*0.15
alphas = [1,1,1,1]
    
for iname, name in enumerate(names):
    rat = pickle.load(open('../data/'+name+'_wds_data_warped.p', 'rb'))
    for iu, unum in enumerate(units[iname]):
        iplot += 1
        
        u = rat['units'][unum]
        rasts = [u[day]['raster_w'] for day in np.sort(list(u.keys()))]
        cols = [[0, 0.1+i/len(rasts)*0.8, 0.9-i/len(rasts)*0.8] for i in range(len(rasts))]

        cmap = matplotlib.cm.get_cmap('viridis')
        cols = [cmap(0.5*i/len(rasts)+0.25) for i in range(len(rasts))]
        cols = [np.array(get_col(name)) * (0.7*(1 - i/len(rasts))+0.4) for i in range(len(rasts))]

        n = 0
        ns_day = []
        ax = fig.add_subplot(gs[0, iplot])
        for irast, rasters in enumerate(rasts):
            if len(rasters) > 1:
                for rast in rasters[::subsamps[iplot]]:
                    if len(rast) > 2:
                        rast = np.array(rast)[::subsamps[iplot]]
                        ax.scatter(rast, [n for i in range(len(rast))], s=ss[iname], color=cols[irast], alpha = alphas[iname],  marker='o', linewidths=0.)
                        n -= 1
    
                if irast < (len(rasts) - 1):
                    ns_day.append(n)
                    n -= 1
    
        if iname in [0, 1, 2, 3]:
            ax.text(-0.22, (0+ns_day[0])/2, 'day 1', horizontalalignment='right', verticalalignment='center', fontsize = 6)
        ax.text(-0.22, (n+ns_day[-1])/2, 'day '+str(len(ns_day)+1), horizontalalignment='right', verticalalignment='center', fontsize = 6)
    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(n, 0)

        # plot example PETHs in panel c
        ax = fig.add_subplot(gs[1, iplot])
        hs = [u[day]['peth_w'] for day in np.sort(list(u.keys()))]
        hs = [h/np.sum(h) for h in hs] #max normalize for visualization
        ts = np.linspace(tmin, tmax, len(hs[0]))
        
        if iplot in [0, 2]:
            inds = [0,round(len(hs)/2-0.5), len(hs)-1]
        else:
            inds = [1,round(len(hs)/2-0.5), len(hs)-2]
        for ind in inds:
            h = hs[ind]
            ax.plot(ts, h, c=cols[ind])
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(0, np.amax(np.concatenate([hs[ind] for ind in inds]))*1.1)
        
        tticks = np.arange(5)*rat['targetint']/0.7
        labs = ['peak 1  ', '', '', '', '    peak 5']
        for itick, tick in enumerate(tticks):
            maxval = np.amax([np.amax(h) for h in hs])
            ax.axvline(tick, color = 'k', ls = '-', lw = 0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('time', fontsize = 11)
        if iplot == 0:
            ax.set_ylabel('firing rate', fontsize = 11)
            
plt.text(0.14, 0.50, 'DLS',ha='center', va='bottom',transform=fig.transFigure)
plt.text(0.48, 0.50, 'motor cortex',ha='center', va='bottom',transform=fig.transFigure)
con = ConnectionPatch(xyA=(0.000, 0.495), xyB=(0.28, 0.495), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)
con = ConnectionPatch(xyA=(0.34, 0.495), xyB=(0.62, 0.495), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)

#%% print stability indices
gs = fig.add_gridspec(2,1, left=l2+0.03, right=1.0, bottom=0.35, top=1.00, wspace = 0.5, hspace = 0.5)

labels = ['DLS', 'MC']
data = pickle.load(open(dataname, 'rb'))
for igroup in range(2):
    gname = labels[igroup]+'_wds'
    perm_alphas = pickle.load(open('../results/'+gname+'_shuffle_alphas_wds.p', 'rb'))
    alphas = data[gname]['alphas']
    med_alpha = np.median(alphas)
    print(gname+' n='+str(len(alphas))+', alpha_med='+str(np.round(med_alpha, 3))+', tau='+str(np.round(-1/med_alpha, 1))+', p='+str(np.mean(perm_alphas <= med_alpha))+'\n')


#%% plot extrapolation analysis
data = pickle.load(open(dataname, 'rb'))
ax = fig.add_subplot(gs[0,0])

for ireg, region in enumerate(['DLS_wds', 'MC_wds']):
    x, y, s = [data[region][k] for k in ['bin_x_dur', 'bin_y_dur', 'bin_s_dur']] #binned data
    res, yhat = data[region]['fit_dur'], data[region]['bin_yhat_dur']#model fit
    
    boot_vals, boot_x = data[region]['fit_boot_vals_dur'], data[region]['boot_x']
    boot_xinds = np.where((boot_x >= x[0]-0.5) & (boot_x <= 35))[0]
    boot_vals, boot_x = boot_vals[..., boot_xinds], boot_x[boot_xinds]
    
    q1, q2, q3 = np.nanquantile(boot_vals, [0.25, 0.50, 0.75], axis = 0)
    boot_y = -np.abs(res[0]) - np.abs(res[1])*np.exp(-np.abs(res[2])*boot_x)
    ax.plot(boot_x, boot_y, color = get_col(region), ls = '--')
    ax.fill_between(boot_x, q1, q3, color = get_col(region), alpha = 0.2)
    ax.scatter(x, y, marker='o', color = get_col(region), s = 15)
    
    alphas, rec_times, alpha_inds = [np.array(data[region][k]) for k in ['alphas', 'rec_times', 'alpha_inds']]
    rec_times = rec_times[alpha_inds] #recording times for the neurons we have alphas for
    Nshuff = 10000
    rhos = np.zeros(Nshuff)
    for n in range(Nshuff):
        inds = np.random.choice(len(alphas), len(alphas), replace = True)
        rhos[n] = pearsonr(rec_times[inds], alphas[inds])[0]
    print(region, 'rho:', pearsonr(rec_times, alphas)[0], 'bootstrapped CI:', np.quantile(rhos, [0.025, 0.975]), '  p =', np.mean(rhos <= 0))
    
ax.axhline(0, ls = '--', color = 'k')
ax.set_xlabel('recording time', labelpad = -10)
ax.set_xticks([4, 30])
ax.set_xlim([4, 30])
ax.set_ylabel(r'$\alpha$', labelpad = -20)
ax.set_yticks([-0.05, 0.0])
ax.set_ylim(-0.06, 0.005)

#%% plot full distribution of neural-behavioral correlations

res = pickle.load(open('../results/neural_behav_corr_DLS_wds.p', 'rb'))
all_sims_n, all_sims_b = res['all_sims_n'], res['all_sims_b']
all_corrs = np.array([pearsonr(all_sims_n[i], all_sims_b[i])[0] for i in range(len(all_sims_n))])
boot = [np.mean(np.random.choice(all_corrs, len(all_corrs), replace = True)) for _ in range(1000)]

res_mc = pickle.load(open('../results/neural_behav_corr_MC_wds.p', 'rb'))
all_sims_n_mc, all_sims_b_mc = res_mc['all_sims_n'], res_mc['all_sims_b']
all_corrs_mc = np.array([pearsonr(all_sims_n_mc[i], all_sims_b_mc[i])[0] for i in range(len(all_sims_n_mc))])

ax = fig.add_subplot(gs[1,0])
h, _, _ = ax.hist(all_corrs, bins = np.linspace(-1, 1, 15), color = get_col('DLS'), alpha = 0.5)
maxval = np.nanmax(h)*1.05
ax.set_xlabel('correlation', labelpad = -2)
ax.axvline(np.mean(all_corrs), color = get_col('DLS'))

h, _, _ = ax.hist(all_corrs_mc, bins = np.linspace(-1, 1, 15), color = get_col('MC'), alpha = 0.5)
ax.axvline(np.mean(all_corrs_mc), color = get_col('MC'))

ax.set_ylim(0, maxval)
ax.set_ylabel('frequency', labelpad = -10)
ax.set_yticks([0, 25])

#%% plot neural vs behav analysis
gs = fig.add_gridspec(1,2, left=l2-0.03, right=1.05, bottom=0.0, top=0.20, wspace = 0.1, hspace = 0.1)

for ireg, region in enumerate(['DLS_wds', 'MC_wds']):
    res = pickle.load(open('../results/neural_behav_corr_syn_'+region+'.p', 'rb'))
    means, means_syn, all_corrs = res['means'], res['means_syn'], res['all_corrs']
    
    minval, maxval = [func(np.concatenate([means, means_syn])) for func in [np.amin, np.amax]]
    bins = np.linspace(minval, maxval, 50)

    col_un = [0,0,0]
    col_stab = [0.5, 0.5, 0.5]
    
    ax = fig.add_subplot(gs[0, ireg])
    ax.hist(means_syn, bins = bins, color = col_stab, alpha = 0.7)
    ax.hist(means, bins = bins, color = col_un, alpha = 0.7)
    
    if ireg == 1:
        ax.plot([], [], ls = '-', color = get_col('DLS'))
    ax.axvline(np.mean(means), color = col_un, lw = 2, ls ='--')
    ax.axvline(np.mean(means_syn), color = col_stab, lw = 2, ls = '--')
    ax.axvline(np.mean(all_corrs), color = get_col(region), lw = 2)
    
    if ireg == 0:
        ax.set_xlabel('                   mean correlation')
        ax.set_ylabel('frequency')
        ax.set_xticks([-0.1, 0.15])
    else:
        leg = ['DLS', 'ctrl', 'syn', 'MC']
        ax.legend(leg, frameon = False, ncol = len(leg), bbox_to_anchor = (-1.4, 1.40), loc = 'upper left',
                   handlelength = 1.0, handletextpad = 0.3, columnspacing = 0.9)
        ax.set_xticks([-0.1, 0.2])
    ax.set_yticks([])

    print('\n'+region+' mean corr = '+str(np.round(np.mean(all_corrs), 3)))
    print(region+' p ctrl = '+str(np.mean(np.array(means) >= np.mean(all_corrs))))
    print(region+' mean synthetic = '+str(np.round(np.mean(means_syn), 3)))
    print(region+' p synthetic = '+str(np.mean(np.array(means_syn) <= np.mean(all_corrs))))

plt.text(-0.05, 1.07, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.05, 0.56, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.44, 1.07, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.71, 1.07, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.71, 0.68, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.65, 0.3, 'F',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

#%% save/show figures

plt.savefig('../paper_figs/main_fig6.pdf', bbox_inches = 'tight')
plt.close()



