#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, ttest_ind, linregress, ttest_1samp
from plot_utils import get_col, col_blue
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

fig = plt.figure(figsize = (11*cm, 7.2*cm))
gs = fig.add_gridspec(1, 3, left=0, right=1, bottom=0.67, top=1., wspace = 0.5, hspace = 0.20, width_ratios = [1,1,0.8])

# raw similarity vs time
data = pickle.load(open('../results/decoding_and_cca/latent_similarity.p', 'rb'))
dts, sims_rho, sims_cca = data['dts'], data['sims_rho'], data['sims_cca']
u_dts = np.unique(dts) #unique time differences
#compute mean and std for each time difference
m_cca = np.array([np.mean(sims_cca[dts == d])for d in u_dts])
s_cca = np.array([np.std(sims_cca[dts == d])/np.sqrt(np.sum(dts == d)) for d in u_dts])
m_rho = np.array([np.mean(sims_rho[dts == d])for d in u_dts])
s_rho = np.array([np.std(sims_rho[dts == d])/np.sqrt(np.sum(dts == d)) for d in u_dts])

ax = fig.add_subplot(gs[0, 0])
ax.plot(u_dts, m_rho, "k-", label = 'single\nneuron')
ax.plot(u_dts, m_cca, color = col_blue, ls = '--', label = 'latent')
ax.fill_between(u_dts[:-1], (m_rho - s_rho)[:-1], (m_rho + s_rho)[:-1], color = 'k', alpha = 0.2)
ax.fill_between(u_dts[:-1], (m_cca - s_cca)[:-1], (m_cca + s_cca)[:-1], color = col_blue, alpha = 0.2)
ax.legend(frameon = False)
ax.set_xlabel('time difference (days)')
ax.set_ylabel('similarity')
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xticks([1,3,5])

## plot decoding
data = pickle.load(open('../results/decoding_and_cca/population_decoding.p', 'rb'))
dts, rhos, alphas = data['dts'], data['rhos_raw'], data['alphas_raw']
u_dts = np.unique(dts)
m_rho = np.array([np.mean([rhos[dts == dt]]) for dt in u_dts])
s_rho = np.array([np.std([rhos[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])

# plot similarity vs time
ax = fig.add_subplot(gs[0, 1])
ax.plot(u_dts, m_rho, "k-", label = 'single neuron')
ax.fill_between(u_dts[:-1], (m_rho - s_rho)[:-1], (m_rho + s_rho)[:-1], color = 'k', alpha = 0.2)
ax.set_xlabel('time difference (days)')
ax.set_ylabel('decoding perf.')
ax.set_xlim(0, 6)
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0, 1])
ax.set_xticks([1,3,5])

#plot histograms
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-0.03, 0.03, 51)
ax.hist(alphas, color = "k", alpha = 1, bins = bins)
ax.axvline(0, color = col_blue, ls = "--")
ax.set_xlabel('stability index')
ax.set_ylabel('frequency')
ax.set_yticks([])
ax.set_xticks([-0.03, 0.0, 0.03])
print('decoding stability: p='+str(np.mean(alphas >= 0)))

### compare with latent
gs = fig.add_gridspec(1, 2, left=0.1, right=0.9, bottom=0.0, top=0.36, wspace = 0.4, hspace = 0.20)
data = pickle.load(open('../results/decoding_and_cca/population_decoding.p', 'rb'))
dts, rhos, rhos_cca = data['dts'], data['rhos_raw'], data['rhos_cca']
u_dts = np.unique(dts)
m_rho = np.array([np.mean([rhos[dts == dt]]) for dt in u_dts])
s_rho = np.array([np.std([rhos[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])
m_cca = np.array([np.mean([rhos_cca[dts == dt]]) for dt in u_dts])
s_cca = np.array([np.std([rhos_cca[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])

# plot decoding similarity vs time
ax = fig.add_subplot(gs[0, 0])
ax.plot(u_dts, m_rho, "k-", label = 'population')
ax.plot(u_dts, m_cca, color = col_blue, ls = '--', label = 'latent')
ax.fill_between(u_dts[:-1], (m_rho - s_rho)[:-1], (m_rho + s_rho)[:-1], color = 'k', alpha = 0.2)
ax.fill_between(u_dts[:-1], (m_cca - s_cca)[:-1], (m_cca + s_cca)[:-1], color = col_blue, alpha = 0.2)
ax.legend(frameon = False)
ax.set_xlabel('time difference (days)')
ax.set_ylabel('decoding perf.')
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xticks([1,3,5])


### plot neural tuning stability ###
maxdt = 6
mincor = 0.1
p_dts = np.arange(maxdt+1)

ax = fig.add_subplot(gs[0, 1])
for name in ['DLS', 'MC']:
        data = pickle.load(open('../results/decoding_and_cca/glm_decoding_'+name+'.p', 'rb'))
        alphas, unums, dts, perfs, cor0s = [data[k] for k in ['alphas', 'unums', 'dts', 'perfs', 'cor0s']]

        data_cv = pickle.load(open('../results/decoding_and_cca/glm_decoding_crossval_'+name+'.p', 'rb'))
        unums_cv, sims_cv = data_cv['unums'], data_cv['sims']

        max_dts = np.array([np.amax(dt) for dt in dts])
        cor0s = np.array(cor0s)

        inds = ((max_dts >= maxdt) & (cor0s > mincor)).nonzero()[0]
        unums, dts, perfs, cor0s = [[arr[ind] for ind in inds] for arr in [unums, dts, perfs, cor0s]]

        p_sims = np.zeros((len(inds), maxdt+1)) + np.nan
        cv_sims = np.zeros((len(inds))) + np.nan
        for i in range(len(inds)):
                unum = unums[i]
                if unum in unums_cv:
                    cv_sims[i] = sims_cv[unums_cv.index(unum)]
                    for dt in p_dts:
                        sim = np.array(perfs[i])[np.array(dts[i]) == dt]
                        if len(sim) > 0:
                                p_sims[i,dt] = sim

        m = np.nanmean(p_sims, axis = 0)
        s = np.nanstd(p_sims, axis = 0)/np.sqrt(np.sum(~np.isnan(p_sims), axis = 0))
        plt.plot(p_dts[1:], m[1:], color = get_col(name), ls = '-', label = name)
        plt.fill_between(p_dts[1:], (m-s)[1:], (m+s)[1:], color = get_col(name), alpha = 0.2)

        mean_cv = np.nanmean(cv_sims)
        plt.plot([p_dts[1], p_dts[-1]], [mean_cv, mean_cv], ls = '--', color = get_col(name))

ax.set_xlim(0, p_dts[-1])
ax.set_ylim(0.0, 0.4)
ax.set_yticks([0, 0.4])
ax.set_xticks([1,3,5])
ax.set_xlabel('time difference (days)')
ax.set_ylabel('correlation', labelpad = -10)
ax.legend(frameon = False, ncol = 2, loc = 'upper center', columnspacing = 0.8, handlelength=1.5, handletextpad = 0.5, bbox_to_anchor = [0.5, 1.1])

##### add labels #####

plt.text(-0.1, 1.1, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.30, 1.1, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.75, 1.1, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.04, 0.46, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.45, 0.46, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    
plt.savefig('../paper_figs/ext_data_fig5.jpg', bbox_inches = 'tight', dpi = png_dpi) 
plt.close()
