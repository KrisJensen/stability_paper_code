#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:51:38 2021

@author: kris
"""
import numpy as np
import pickle
from compute_neural_sim import compute_neural_sim, calc_alphas, compute_shuffled_sims
from utils import load_rat, check_modulated
from compute_behav_sim import compute_behav_sim
from compute_neural_behav_sim import comp_neural_behav_sim
from scipy.stats import pearsonr
import sys
import os
os.nice(5)

np.random.seed(17161309)

if len(sys.argv) > 1: #0/1 for twotap/wds
    wds = bool(int(sys.argv[1]))
else:
    wds = False
print('wds:', wds)

permute = True #run permutation test on stability indices
names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh', 'DLS', 'MC']
data = {}
for name in names:
    print('')
    
    if wds:
        name += '_wds'
        same_sims = pickle.load(open('./results/sameday_sim/sims_wds_split.p', 'rb'))
    else:
        same_sims = pickle.load(open('./results/sameday_sim/sims_twotap_split.p', 'rb'))
        
    data[name] = {}
    
    rat = load_rat(name, trim = True)
    
    unums, dts, sims, day1s = compute_neural_sim(rat) #compute similarity vs time difference
    
    mod_inds, _ = check_modulated(unums, same_sims, name) #only consider significantly modulated
    unums, dts, sims, days1 = [[arr[ind] for ind in mod_inds] for arr in [unums, dts, sims, day1s]]

    data[name]['unums'], data[name]['dts'], data[name]['sims'] = unums, dts, sims
    rec_times = np.array([int(np.amax(dt)+1) for dt in dts])
    data[name]['rec_times'] = rec_times
    alphas, inds, fits, errors, mean_errors, all_fits = calc_alphas(unums, dts, sims, comp_error=True) #compute similarity indices
    betas = [fit.x[0] for fit in fits]
    data[name]['alphas'], data[name]['alpha_inds'] = alphas, inds
    data[name]['fits'], data[name]['errors'], data[name]['mean_errors'] = fits, errors, mean_errors
    data[name]['all_fits'] = all_fits

    ### lower bound!
    long_inds = (rec_times >= 14).nonzero()[0]
    long_us = np.array(unums)[long_inds]
    if len(long_us) > 1:
        long_shuffled_sims = compute_shuffled_sims(rat, long_us, nsamples = 1000)
    else:
        long_shuffled_sims = []
    data[name]['long_shuffled_sims'] = long_shuffled_sims
    
    ### also add whether this is projection or inter
    anal_types = {unum: rat['unittypes'][unum] for unum in unums}
    data[name]['types'] = anal_types

    if name in ['MC', 'DLS', 'MC_wds', 'DLS_wds']:

        ### compare neural and behavioral data ###
        print('\nneural behavioral correlation for', name)
        dts_b, sims_b, day1s_b = compute_behav_sim(rat)
        unums, dts_n, sims_n, day1s_n = compute_neural_sim(rat, synthetic = False, day0 = False)
        mod_inds, _ = check_modulated(unums, same_sims, name) #only consider significantly modulated
        unums, dts_n, sims_n, day1s_n = [[arr[ind] for ind in mod_inds] for arr in [unums, dts_n, sims_n, day1s_n]]
        all_sims_n, all_sims_b, all_corrs = comp_neural_behav_sim(unums, dts_n, sims_n, day1s_n, dts_b, sims_b, day1s_b)
        pickle.dump({'all_sims_n': all_sims_n, 'all_sims_b': all_sims_b},
                        open('./results/neural_behav_corr_trimmed_'+name+'.p', 'wb'))
          
        ### run permutation test ###
        means = []
        nreps_nb = 5000
        print('running permutation test')
        for i in range(nreps_nb):
            _, _, corrs = comp_neural_behav_sim(unums, dts_n, sims_n, day1s_n, dts_b, sims_b, day1s_b, permute = True)
            means.append(np.mean(corrs))
            if i % 1000 == 0: print('permute', i, np.mean(means), np.std(means))

        ### save correlation analysis ###
        pickle.dump({'means': means, 'means_syn': [], 'all_corrs': all_corrs},
                    open('./results/neural_behav_corr_syn_trimmed_'+name+'.p', 'wb'))

    else:
        ### behavior; not for aggregate data ###
        dts_b, sims_b, day1s_b = compute_behav_sim(rat, verbose = False)
        data[name]['dts_b'] = dts_b
        data[name]['sims_b'] = sims_b
        data[name]['day1s_b'] = day1s_b

        ### bootstrap confidence interval ###
        nreps = 10000
        corrs = np.zeros(nreps)
        dts_b, sims_b = np.array(dts_b), np.array(sims_b)
        for nrep in range(nreps):
            inds_b = np.random.choice(len(dts_b), len(dts_b), replace = True)
            corrs[nrep] = pearsonr(dts_b[inds_b], sims_b[inds_b])[0]
            
        qs = [0.005, 0.025, 0.25, 0.5, 0.75, 0.975, 0.995]
        data[name]['corrs_b'] = corrs
        
if wds:
    pickle.dump(data, open('./results/neural_similarity_analyses_trimmed_wds.p', 'wb'))
else:
    pickle.dump(data, open('./results/neural_similarity_analyses_trimmed_twotap.p', 'wb'))

