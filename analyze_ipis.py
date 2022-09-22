#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, sem
from utils import get_modes
from scipy.ndimage import gaussian_filter1d
np.random.seed(8490410)

ipi_data = {}

names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh']
cols = [[0, 0, 1], [0, 0, 0.8], [0, 0, 0.6], [1, 0, 0], [0.8, 0, 0], [0.6, 0, 0]]

def rolling_median(ipis, wsize = 500, step = 100):
    b0s = np.arange(0, len(ipis)-wsize+step, step)
    b1s = b0s+wsize
    ipi_bins = [ipis[b0s[i]:b1s[i]] for i in range(len(b0s))]
    meds = np.array([np.nanmedian(ipi) for ipi in ipi_bins])
    return meds

ipi_lens = []
ipi_day_lens = []
raw = False
all_ipis = []
for iname, name in enumerate(names):
    
    if raw: #consider all trials
        rat = pickle.load(open('./data/'+name+'_data.p', 'rb'))
        mode = get_modes(name)[0]
        ipis_day = [day['ipis'][day['modes'] == mode] for day in rat['trials'].values()]
    
    else: #only those used in other analyses
        rat = pickle.load(open('./data/'+name+'_data_warped.p', 'rb'))
        ipis_day = [day['ipis'] for day in rat['trials'].values()]
        
    ipis = np.concatenate(ipis_day)*1e3 #ms
    
    ipis = ipis[~np.isnan(ipis)]
    ipis_c = gaussian_filter1d(ipis, 200)
    ts = np.linspace(0, 1, len(ipis_c))
    
    lens = [np.sum(1-np.isnan(v)) for v in ipis_day]
    ipi_lens.append(len(ipis))
    ipi_day_lens.append(np.mean(lens))
    all_ipis.append(ipis)

ipi_data['names'] = names
ipi_data['ipis'] = all_ipis
ipi_data['trials'] = ipi_day_lens
    
    
#%% now compute autocorrelation ###
step = 10

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
    
    all_autocors.append(autocor)
    all_inds.append(inds)
    binned_autocors[iname, :] = binned_statistic(xs, autocor, 'mean', bins = bins)[0]
    
ipi_data['autocors'] = all_autocors
ipi_data['autocor_inds'] = all_inds

pickle.dump(ipi_data, open('./results/ipi_data.p', 'wb'))

