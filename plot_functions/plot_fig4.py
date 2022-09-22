#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 07:45:50 2021

@author: kris
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, linregress, ttest_ind
from plot_utils import get_col
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
np.random.seed(8422209)

fig = plt.figure(figsize = (16*cm, 12*cm))
dataname = '../results/neural_similarity_analyses_twotap.p'

r1, l2 = 0.51, 0.61

#%% plot activity heatmaps in panel a

gs = fig.add_gridspec(1, 4, left=0, right=r1-0.01, bottom=0.65, top=1., wspace = 0.1, hspace = 0.20, width_ratios = [1, 0.15, 1, 0.07])
xs = [-0.1, 0.8]
titles = ['DLS', 'motor cortex']
for i, name in enumerate(['Hindol', 'Hamir']):
    rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))
    
    rast_sort = []
    raster = []
    peaks = []
    for unum, u in rat['units'].items():
        if len(u.keys()) >= 1:
            newrast = np.concatenate([u[day]['peth_w_t'] for day in u.keys()], axis = 0)
            if np.sum(newrast) >= 100:
                sort_rast = np.mean(newrast[1::2, :], axis = 0)
                peaks.append(np.argmax(sort_rast))
                raster.append(np.mean(newrast[0::2, :], axis = 0))
            
    raster = np.array(raster)[np.argsort(peaks), :]
    grid = (raster - np.mean(raster, axis = 1, keepdims = True))/np.std(raster, axis = 1, keepdims = True)
    
    vmin, vmax = -2, 4
    ax = fig.add_subplot(gs[0, 2*i])
    im = ax.imshow(grid, cmap=global_cmap, vmin=vmin, vmax = vmax, aspect='auto', interpolation = 'none')
    ax.set_yticks([-0.5, grid.shape[0]-0.5])
    ax.set_yticklabels([grid.shape[0], 1])
    ax.set_xticks(np.linspace(-0.5, np.shape(grid)[1]-0.5, len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlabel('time (s)', labelpad=-10)
    ax.tick_params(axis='y', which='major', pad=-0.5)
    ax.set_title(titles[i], pad = 9)
    
    N = grid.shape[0]
    xval = grid.shape[1]/9 - 0.5
    ax.annotate("", xy=(xval, 0.020*N-0.5), xytext=(xval, -0.060*N-0.5), arrowprops=dict(arrowstyle="-|>", fc = 'k'))
    xval = grid.shape[1]/9*8 - 0.5
    ax.annotate("", xy=(xval, 0.020*N-0.5), xytext=(xval, -0.060*N-0.5), arrowprops=dict(arrowstyle="-|>", fc = 'k'))
    
    if i == 0:
        ax.set_ylabel('unit', labelpad=-20)
        
newax = fig.add_subplot(gs[0, 3])
newax.set_xticks([])
newax.set_yticks([])
cbar = plt.colorbar(im, orientation = 'vertical', ax = newax, fraction = 1.0, shrink = 1, ticks=[vmin, vmax])
cbar.ax.set_yticklabels([str(vmin), str(vmax)], rotation = 0, va = 'center', ha = 'left')


# %% plot example rasters in panel b
names = ['Hindol', 'Hamir']
units = [[378, 569], [790, 817]]

gs = fig.add_gridspec(2, 4, left=0, right=r1, bottom=0.0, top=0.44, wspace = 0.5, hspace = 0.1)
iplot = -1
tmin, tmax = -0.1, 0.8
subsamps = [1, 1, 1, 1]
ss = np.ones(4)*0.15
alphas = [1,1,1,1]

for iname, name in enumerate(names):
    rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))
    for iu, unum in enumerate(units[iname]):
        iplot += 1
        
        u = rat['units'][unum]
        days = np.sort(list(u.keys()))
        days = [day for day in days if np.sum(u[day]['peth_w_t']) >= 10] #at least 10 spikes
        rasts = [u[day]['raster_w'] for day in days]

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
                    ax.axhline(n, color = 'k', ls = '-', linewidth=0.5)
                    ns_day.append(n)
                    n -= 1
    
        if iplot in [0]:
            ax.text(-0.12, (ns_day[0]+ns_day[1])/2, 'day 2', horizontalalignment='right', verticalalignment='center', fontsize = 6)
        if iname in [0, 1, 2, 3]:
            ax.text(-0.12, (0+ns_day[0])/2, 'day 1', horizontalalignment='right', verticalalignment='center', fontsize = 6)
        ax.text(-0.12, (n+ns_day[-1])/2, 'day '+str(len(ns_day)+1), horizontalalignment='right', verticalalignment='center', fontsize = 6)
    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(n, 0)
    
        # plot example PETHs in panel c
        ax = fig.add_subplot(gs[1, iplot])
        hs = [u[day]['peth_w'] for day in np.sort(list(u.keys()))]
        hs = [h/np.sum(h) for h in hs] #max normalize for visualization
        xs = np.arange(len(hs[0]))
        
        inds = [0,round(len(hs)/2-0.5), len(hs)-1]
        for ind in inds:
            h = hs[ind]
            ax.plot(xs, h, c=cols[ind])
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0, np.amax(np.concatenate([hs[ind] for ind in inds]))*1.1)
        for itick, tick in enumerate([5, 40]):
            maxval = np.amax([np.amax(h) for h in hs])
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('time', fontsize = 11)
        if iplot == 0:
            ax.set_ylabel('firing rate', fontsize = 11)

### add labels
plt.text(0.12, 0.46, 'DLS',ha='center', va='bottom',transform=fig.transFigure)
plt.text(0.39, 0.46, 'motor cortex',ha='center', va='bottom',transform=fig.transFigure)
con = ConnectionPatch(xyA=(0.01, 0.45), xyB=(0.23, 0.45), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)
con = ConnectionPatch(xyA=(0.28, 0.45), xyB=(0.50, 0.45), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)
        
#%% add similarity vs time analysis
from plot_utils import fit_model
gs = fig.add_gridspec(1,2, left=l2, right=1.0, bottom=0.71, top=1.00, wspace = 0.5, hspace = 0.1)

data = pickle.load(open(dataname, 'rb'))

ax = fig.add_subplot(gs[0, :]) #fill the whole thing!

labels = ['DLS', 'MC']
for iname, name in enumerate(['DLS', 'MC']):
    dts, rec_times, sims, unums = [data[name][k] for k in ['dts', 'rec_times', 'sims', 'unums']]
    
    inds = (rec_times >= 14).nonzero()[0]
    bins = np.arange(0.5, 13.5+0.01, 1)
    long_sims = np.zeros((len(inds), len(bins)-1))
    for iind, ind in enumerate(inds):
        unum = unums[ind]
        dt, sim = dts[ind], sims[ind]
        long_sims[iind, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]

    nans = np.isnan(long_sims)
    no_nans = np.array([(not any(nans[i, :])) for i in range(nans.shape[0])])
    long_sims = long_sims[no_nans, :]
    inds = inds[no_nans]

    m, s = np.nanmean(long_sims, axis = 0), np.nanstd(long_sims, axis = 0)/np.sqrt(np.sum(1-np.isnan(long_sims), axis = 0))
    print(name+' number of units with >= 14 recording days:', long_sims.shape[0])
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
ax.set_ylim(-0.03, 1)
ax.set_yticks([0, 1])
ax.set_xticks([xs[0], xs[-1]])
ax.set_xlabel('time difference (days)', labelpad = -15)
ax.set_ylabel('correlation', labelpad = -10)
ax.legend(frameon = False)

#%% add stability index by fitting duration
gs = fig.add_gridspec(1,2, left=l2-0.01, right=1.0, bottom=0.58-0.23, top=0.58, wspace = 0.6, hspace = 0.1)

ax = fig.add_subplot(gs[0, 1])

data = pickle.load(open('../results/neural_similarity_analyses_twotap.p', 'rb'))
for name in ['MC', 'DLS']:
    dts, rec_times, sims, unums = [data[name][k] for k in ['dts', 'rec_times', 'sims', 'unums']]
    
    inds = (rec_times >= 14).nonzero()[0]
    bins = np.arange(0.5, 13.5+0.01, 1)
    xs = (bins[1:] + bins[:-1])/2
    long_sims = np.zeros((len(inds), len(bins)-1))
    for iind, ind in enumerate(inds):
        dt, sim = dts[ind], sims[ind]
        long_sims[iind, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]
        
    nans = np.isnan(long_sims)
    no_nans = np.array([(not any(nans[i, :])) for i in range(nans.shape[0])])
    long_sims = long_sims[no_nans, :]

    m, s = np.nanmean(long_sims, axis = 0), np.nanstd(long_sims, axis = 0)/np.sqrt(np.sum(1-np.isnan(long_sims), axis = 0))
    alphas = []
    max_dts = np.arange(len(xs)-2)+3
    for i in max_dts:
        fit = fit_model(xs[:i], m[:i], baseline = False)
        alphas.append(fit.x[1])

    ax.scatter(max_dts, alphas, color = get_col(name), s = 10)
    print(name+':', pearsonr(max_dts, alphas))
ax.axhline(0.00, ls = '--', color = 'k')
ax.set_xticks([2, 14])
ax.set_yticks([-0.04, 0.0])
ax.set_xlabel(r'max $\Delta$t', labelpad = -10)
ax.set_ylabel(r'$\alpha$', labelpad = -20)
ax.set_ylim(-0.042, 0.005)

#%% add stability indices for individual fits
ax = fig.add_subplot(gs[0, 0])
labels = ['DLS', 'MC']
v = 0.3
xvals = [[i-0.25+0.5/(2)*j for j in range(3)] for i in range(2)]
sigs = [['*', '*', '*'], ['*', '', '*']]

for igroup, group in enumerate([['Hindol', 'Dhanashri', 'Jaunpuri'],
                                ['Hamir', 'Gandhar', 'Gorakh']]):
    gname = ['DLS', 'MC'][igroup]
    
    meds, quants = [], []
    for iname, name in enumerate(group):
        alphas = data[name]['alphas']
        meds.append(np.nanmedian(alphas))
        quants.append(np.abs(np.nanquantile(alphas, [0.25, 0.75]) - meds[-1]))
        plt.errorbar([xvals[igroup][iname]],
                     [meds[iname]],
                     yerr=np.array(quants[iname]).reshape(2, 1),
                     capsize=5, linestyle='', marker='o',
                     color=get_col(name))
        
        perm_alphas = pickle.load(open('../results/'+name+'_shuffle_alphas_twotap.p', 'rb'))
        med_alpha = np.median(alphas)
        print(name+' n='+str(len(alphas))+', alpha_med='+str(np.round(med_alpha, 3))+', tau='+str(np.round(-1/med_alpha, 1))+', p='+str(np.mean(perm_alphas <= med_alpha)))
        
    for iname in range(3):
        plt.text(xvals[igroup][iname], 0.3/4/3, sigs[igroup][iname], verticalalignment='center',
                 horizontalalignment='center', fontsize=20)
        
    perm_alphas = pickle.load(open('../results/'+gname+'_shuffle_alphas_twotap.p', 'rb'))
    alphas = data[gname]['alphas']
    med_alpha = np.median(alphas)
    print(gname+' n='+str(len(alphas))+', alpha_med='+str(np.round(med_alpha, 3))+', tau='+str(np.round(-1/med_alpha, 1))+', p='+str(np.mean(perm_alphas <= med_alpha))+'\n')


ax.set_ylabel(r'$\alpha$', labelpad=-10)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.tick_params(axis="x", which="both", bottom=False, top=False)
ax.set_xlim(xvals[0][0]-0.2, xvals[-1][-1]+0.2)
ax.axhline(y=0.0, color='k', ls='-')

ax.set_ylim(-0.3, 0.3/4)
ax.set_yticks([0, -0.4])
ax.set_yticklabels([0, -0.4])
ax.set_ylim(-0.4, 0.3/4)
    
#%% add extrapolated data
gs = fig.add_gridspec(1,2, left=l2, right=1.0, bottom=0.0, top=0.24, wspace = 0.1, hspace = 0.1)

data = pickle.load(open(dataname, 'rb'))
for ireg, region in enumerate(['DLS', 'MC']):
    x, y, s = [data[region][k] for k in ['bin_x_dur', 'bin_y_dur', 'bin_s_dur']] #binned data
    res, yhat = data[region]['fit_dur'], data[region]['bin_yhat_dur']#model fit
    
    ax = fig.add_subplot(gs[0, ireg])
    boot_vals, boot_x = data[region]['fit_boot_vals_dur'], data[region]['boot_x']
    boot_xinds = np.where((boot_x >= x[0]-0.5) & (boot_x <= 35))[0]
    boot_vals, boot_x = boot_vals[..., boot_xinds], boot_x[boot_xinds]
    
    q1, q2, q3 = np.nanquantile(boot_vals, [0.25, 0.50, 0.75], axis = 0)
    boot_y = -np.abs(res[0]) - np.abs(res[1])*np.exp(-np.abs(res[2])*boot_x)
    ax.plot(boot_x, boot_y, color = get_col(region), ls = '--')
    ax.fill_between(boot_x, q1, q3, color = get_col(region), alpha = 0.2)
    
    ax.scatter(x, y, marker='x', color = 'k', s = 30)
    
    boot = data[region]['fit_boot_dur']
    print(region+' asymptotic:', 1/res[0])
    print(region+' bootstrapped quartiles:', np.nanquantile(1/boot[:,0], [0.25, 0.50, 0.75]))
    
    ax.axhline(0, ls = '--', color = 'k')
    ax.set_xlabel('rec. duration (d.)')
    if ireg == 0:
        ax.set_ylabel(r'$\alpha$', labelpad = -20)
        ax.set_yticks([-0.05, 0.0])
    else:
        ax.set_yticks([])
    ax.set_ylim(-0.06, 0.005)

    ### compute significance ###
    alphas, rec_times, alpha_inds = [np.array(data[region][k]) for k in ['alphas', 'rec_times', 'alpha_inds']]
    rec_times = rec_times[alpha_inds] #recording times for the neurons we have alphas for
    Nshuff = 10000
    rhos = np.zeros(Nshuff)
    for n in range(Nshuff):
        inds = np.random.choice(len(alphas), len(alphas), replace = True)
        rhos[n] = pearsonr(rec_times[inds], alphas[inds])[0]
    print(region, 'rho:', pearsonr(rec_times, alphas)[0], '  bootstrapped CI:', np.quantile(rhos, [0.025, 0.975]), '  p =', np.mean(rhos <= 0))


plt.text(-0.05, 1.07, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.05, 0.52, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.57, 1.07, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.57, 0.63, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.81, 0.63, 'F',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.57, 0.28, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    
#%% save fig
plt.savefig('../paper_figs/main_fig4.pdf', bbox_inches = 'tight')
plt.close()
