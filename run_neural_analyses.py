#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:51:38 2021

@author: kris
"""
import numpy as np
import pickle
from compute_neural_sim import compute_neural_sim, calc_alphas, binning, bootstrap_asymptotic, compute_shuffled_sims
from utils import load_rat, fit_asymptotic_model, asymptotic_model, check_modulated
from calc_stationary_sim import resample_rat
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
    print(name)
    if wds:
        name += '_wds'
        same_sims = pickle.load(open('./results/sameday_sim/sims_wds_split.p', 'rb'))
    else:
        same_sims = pickle.load(open('./results/sameday_sim/sims_twotap_split.p', 'rb'))
        
    data[name] = {}
    
    rat = load_rat(name)
    
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

    ### repeat for resampled rats!!! ###
    print('resampling')
    data_re = {unum: {'dts': [], 'sims': []} for unum in rat['units'].keys()}
    resample = 100
    for i in range(resample):
        if i % 20 == 0: print(i)
        resampled_rat = resample_rat(rat)
        ure, dre, sre, d1re = compute_neural_sim(resampled_rat)
        mod_re, _ = check_modulated(ure, same_sims, name) #only consider significantly modulated
        ure, dre, sre, d1re = [[arr[ind] for ind in mod_re] for arr in [ure, dre, sre, d1re]]
        for iu, unum in enumerate(ure):
            data_re[unum]['dts'].append(dre[iu])
            data_re[unum]['sims'].append(sre[iu])
    data[name]['resamples'] = data_re

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
    
    if permute:
        #### also compute shuffled alphas #####
        print('permutation test')
        nreps = 2000
        shuffle_alphas = np.zeros(nreps)
        for i in range(nreps):
            if i % 400 == 0 and i > 0:
                print(i, np.nanmean(shuffle_alphas[:i]), np.nanstd(shuffle_alphas[:i]))
            rand_inds = [np.random.choice(len(dt), len(dt), replace = False) for dt in dts]
            newdts = [np.array(dt)[rand_inds[i]] for i, dt in enumerate(dts)]
            newalphas, newinds, _, _, _, _ = calc_alphas(unums, newdts, sims, comp_error=True)

            #check which alphas are from significantly modulated units
            shuffle_alphas[i] = np.nanmedian(newalphas) #store new data
        if wds:
            pickle.dump(shuffle_alphas, open('./results/'+name+'_shuffle_alphas_wds.p', 'wb'))
        else:
            pickle.dump(shuffle_alphas, open('./results/'+name+'_shuffle_alphas_twotap.p', 'wb'))
        
    
    if name in ['MC', 'DLS', 'MC_wds', 'DLS_wds']:
        
        ### stratify by recording time ###
        alpha_dts = [np.amax(dts[ind])+1 for ind in inds]
        
        bins = [3.5, 6, 10, 15, 21] + [max(np.amax(alpha_dts)+0.5, 22)]
        if 'wds' in name:
            bins = [3.5, 10, 18] + [np.amax(alpha_dts)+0.5]
        
        #only modulated neurons
        x, y, s = binning(alpha_dts, alphas, bins1 = bins)
        res = fit_asymptotic_model(alpha_dts, alphas)
        yhat = asymptotic_model(res, x)
        data[name]['bin_x_dur'], data[name]['bin_y_dur'], data[name]['bin_s_dur'] = x, y, s
        data[name]['fit_dur'], data[name]['bin_yhat_dur'] = res, yhat
    
        boot = bootstrap_asymptotic(alpha_dts, alphas, N = 10000, prop = False)#True)
        vals = []
        boot_x = np.linspace(1, 50, 101)
        for res in boot:
            if not any(np.isnan(res)):
                vals.append(asymptotic_model(res, boot_x))
        data[name]['fit_boot_dur'] = boot
        data[name]['fit_boot_vals_dur'] = np.array(vals)
        data[name]['boot_x'] = boot_x

if wds:
    pickle.dump(data, open('./results/neural_similarity_analyses_wds.p', 'wb'))
else:
    pickle.dump(data, open('./results/neural_similarity_analyses_twotap.p', 'wb'))

