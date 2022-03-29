#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, linregress
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

fig = plt.figure(figsize = (16*cm, 6*cm))

gs = fig.add_gridspec(2, 8, left=0, right=1, bottom=0, top=1., wspace = 0.35, hspace = 0.60, width_ratios = [1,1,1,1,1,1,0.05,1.5])

### plot all behavior and a histogram of the CIs. Each row is an experiment ###


names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh']
cols = [[0, 0, 1], [0, 0, 0.8], [0, 0, 0.6], [1, 0, 0], [0.8, 0, 0], [0.6, 0, 0]]

for iwds, wds in enumerate([False, True]):
    
    if wds:
        data = pickle.load(open('../results/behav_similarity_analyses_wds.p', 'rb'))
    else:
        data = pickle.load(open('../results/behav_similarity_analyses_twotap.p', 'rb'))

    
    allcorrs = []
    rawcorrs = []
    bins = np.linspace(-1, 1, 1501)
    for iname, name in enumerate(names):
        if wds:
            name += '_wds'
        
        ax = fig.add_subplot(gs[iwds, iname])
        dts, sims = data[name]['dts'], data[name]['sims']
        ax.scatter(dts, sims, color = cols[iname], marker = 'o', s = 1, alpha = 0.05)
        
        s, i = linregress(dts, sims)[:2]
        
        xs = np.array([np.amin(dts), np.amax(dts)])
        ax.plot(xs, i+s*xs, ls = '-', color = 'k')#cols[iname])
        ax.set_ylim(0, 1)
        ax.set_xlim(xs[0], xs[1])
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize = 10)
        ax.set_yticks([])
        ax.set_xlabel(r'$\Delta$'+'time (d.)', labelpad = -1)
        if iname == 0:
            ax.set_yticks([0, 1])
            ax.set_ylabel('correlation', labelpad = -10)
            #if not wds:
            #    ax.text(-0.50, 1.25, 'A', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
            
        newcorrs = data[name]['corrs']
        h = np.histogram(newcorrs, bins=bins)[0].astype(float)
        c = gaussian_filter1d(h, 5, mode = 'nearest')
        #print(h, c)
        allcorrs.append(c)
        rawcorrs.append(newcorrs)
    
    ax = fig.add_subplot(gs[iwds, -1])
    print('\nwds:', wds)
    for iname in range(len(names)):
        print(names[iname]+': p='+str(np.nanmean(rawcorrs[iname] >= 0.))+' (n='+str(len(rawcorrs[iname]))+')')
        plt.plot(bins[1:], allcorrs[iname], color = cols[iname], ls = '-', lw = 1)
    ax.set_xlim(-0.7, 0.1)
    ax.axvline(0, ls = '-', color = 'k')
    ax.set_yticks([])
    xs = [-0.5, 0]
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, fontsize = 10)
    ax.set_xlabel('correlation', labelpad = -1)
    ax.set_ylabel('frequency')
    #if not wds:
    #    ax.text(-0.20, 1.25, 'B', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
        

plt.text(-0.05, 1.12, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.80, 1.12, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
        
#plt.savefig('../paper_figs/Sfig_behav.png', bbox_inches = 'tight')
plt.savefig('../paper_figs/Sfig_behav.png', bbox_inches = 'tight', dpi = png_dpi)  
plt.savefig('../paper_figs/Sfig_behav.pdf', bbox_inches = 'tight')
#plt.show()
plt.close()

