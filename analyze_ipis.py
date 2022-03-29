#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, sem
from scipy.ndimage import gaussian_filter1d
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
panel_font = 16
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54
np.random.seed(8490410)

ipi_data = {}

fig = plt.figure(figsize = (17*cm, 3*cm))

gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=1., wspace = 0.25, hspace = 0.20)

ax = fig.add_subplot(gs[0, 0])
ax.text(-0.20, 1.20, 'a', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')

names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh']
cols = [[0, 0, 1], [0, 0, 0.8], [0, 0, 0.6], [1, 0, 0], [0.8, 0, 0], [0.6, 0, 0]]

def rolling_median(ipis, wsize = 500, step = 100):
    b0s = np.arange(0, len(ipis)-wsize+step, step)
    b1s = b0s+wsize
    print(b0s[:5], b1s[:5])
    ipi_bins = [ipis[b0s[i]:b1s[i]] for i in range(len(b0s))]
    meds = np.array([np.nanmedian(ipi) for ipi in ipi_bins])
    return meds

ipi_lens = []
ipi_day_lens = []
all_ipis = []
for iname, name in enumerate(names):

    rat = pickle.load(open('./data/'+name+'_data_warped.p', 'rb'))
    ipis_day = [day['ipis'] for day in rat['trials'].values()]
        
    ipis = np.concatenate(ipis_day)*1e3 #ms
    
    ipis = ipis[~np.isnan(ipis)]
    ipis_c = gaussian_filter1d(ipis, 200)
    #ipis_c = rolling_median(ipis)
    ts = np.linspace(0, 1, len(ipis_c))
    ax.plot(ts, ipis_c, color = cols[iname], ls = '-')
    
    lens = [np.sum(1-np.isnan(v)) for v in ipis_day]
    ipi_lens.append(len(ipis))
    ipi_day_lens.append(np.mean(lens))
    print(name, len(ipis), '  trials:', np.mean(lens), np.std(lens))
    all_ipis.append(ipis)
    
print(np.mean(ipi_lens), '+-', np.std(ipi_lens))
print(np.mean(ipi_day_lens), '+-', np.std(ipi_day_lens))
    
ax.axhline(700, ls = '-', color = 'k')
ax.set_xlabel('time (normalized)')
ax.set_ylabel('IPI (ms)')
ax.set_xticks([])
ax.set_xlim(0, 1)

ax.set_yticks([600, 700, 800])

ipi_data['names'] = names
ipi_data['ipis'] = all_ipis
ipi_data['trials'] = ipi_day_lens
    
    
#%% now compute autocorrelation ###
step = 10

plt.figure()
gs = fig.add_gridspec(1, 2, left=1/6, right=1-1/6, bottom=0.0, top=1., wspace = 0.25, hspace = 0.20)

bins = np.arange(1/200, 10, 1/12)
all_autocors = [] 
binned_autocors = np.zeros((len(names), len(bins)-1))
all_inds = []
for iname, name in enumerate(names):
    ipis = all_ipis[iname]
    
    inds = (step*np.arange(int(np.floor(len(ipis)/step/2)))+1).astype(int)
    autocor = [pearsonr(ipis[ind:], ipis[:-ind])[0] for ind in inds]
    autocor_c = gaussian_filter1d(autocor, 1)
    xs = inds / ipi_day_lens[iname]
    plt.plot(xs, autocor_c, color = cols[iname], ls = '--', lw = 1, alpha = 0.5)
    
    all_autocors.append(autocor)
    all_inds.append(inds)
    binned_autocors[iname, :] = binned_statistic(xs, autocor, 'mean', bins = bins)[0]
    
ipi_data['autocors'] = all_autocors
ipi_data['autocor_inds'] = all_inds
    
m, s = [f(binned_autocors, axis = 0) for f in [np.nanmean, np.nanstd]]
#s /= np.sqrt(np.sum(1-np.isnan(all_autocors), axis = 0))

plt.plot(bins[:-1], m, color = 'k')
plt.fill_between(bins[:-1], m-s, m+s, color = 'k', alpha = 0.2)
#plt.axvline(1, ls = ':', color = 'k')
plt.axhline(0, color = 'k', ls = ':')
plt.xlim(0, 4)
plt.xlabel('autocorrelation')
plt.ylabel('time (pseudo days)')
    
plt.show()


pickle.dump(ipi_data, open('./results/ipi_data.p', 'wb'))

