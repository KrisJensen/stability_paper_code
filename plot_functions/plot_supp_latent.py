#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

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

fig = plt.figure(figsize = (16*cm, 3*cm))

gs = fig.add_gridspec(1, 4, left=0, right=1, bottom=0.0, top=1., wspace = 0.5, hspace = 0.20)


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
ax.plot(u_dts, m_cca, 'b--', label = 'latent')
ax.fill_between(u_dts[:-1], (m_rho - s_rho)[:-1], (m_rho + s_rho)[:-1], color = 'k', alpha = 0.2)
ax.fill_between(u_dts[:-1], (m_cca - s_cca)[:-1], (m_cca + s_cca)[:-1], color = 'b', alpha = 0.2)
ax.legend(frameon = False)
ax.set_xlabel('time difference (days)')
ax.set_ylabel('similarity')
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)

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


#plot histograms
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-0.03, 0.03, 51)
ax.hist(alphas, color = "k", alpha = 1, bins = bins)
ax.axvline(0, color = 'b', ls = "--")
ax.set_xlabel('stability index')
ax.set_ylabel('frequency')
ax.set_yticks([])
ax.set_xticks([-0.03, 0.0, 0.03])
print('decoding stability: p='+str(np.mean(alphas >= 0)))



### compare with latent

data = pickle.load(open('../results/decoding_and_cca/population_decoding.p', 'rb'))
dts, rhos, rhos_cca = data['dts'], data['rhos_raw'], data['rhos_cca']
u_dts = np.unique(dts)
m_rho = np.array([np.mean([rhos[dts == dt]]) for dt in u_dts])
s_rho = np.array([np.std([rhos[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])

m_cca = np.array([np.mean([rhos_cca[dts == dt]]) for dt in u_dts])
s_cca = np.array([np.std([rhos_cca[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])

# plot decoding similarity vs time
ax = fig.add_subplot(gs[0, 3])
ax.plot(u_dts, m_rho, "k-", label = 'population')
ax.plot(u_dts, m_cca, 'b--', label = 'latent')
ax.fill_between(u_dts[:-1], (m_rho - s_rho)[:-1], (m_rho + s_rho)[:-1], color = 'k', alpha = 0.2)
ax.fill_between(u_dts[:-1], (m_cca - s_cca)[:-1], (m_cca + s_cca)[:-1], color = 'b', alpha = 0.2)
ax.legend(frameon = False)
ax.set_xlabel('time difference (days)')
ax.set_ylabel('decoding perf.')
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)


plt.text(-0.08, 1.25, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.21, 1.25, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.51, 1.25, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.73, 1.25, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    
plt.savefig('../paper_figs/Sfig_latent.png', bbox_inches = 'tight', dpi = png_dpi) 
plt.savefig('../paper_figs/Sfig_latent.pdf', bbox_inches = 'tight')
#plt.show()
plt.close()