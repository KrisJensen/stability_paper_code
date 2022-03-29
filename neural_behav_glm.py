#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:25:56 2021

@author: kris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, ttest_1samp, binned_statistic
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import pickle
from utils import global_params
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def make_design_mat(times, peths, vels, tmin = 'default', tmax = 'default', half = 'both', type_ = 'twotap'):
    
    assert type_ in ['twotap', 'wds']
    if tmin == 'default':
        tmin = -0.2 if type_ == 'wds' else -0.1
    if tmax == 'default':
        tmax = 0.5 if type_ == 'wds' else 0.8
    binsize = global_params['binsize']
    
    X, Y = vels, peths
    #print(X.shape)
    X = np.concatenate([X[limb] for limb in X.keys()], axis = -1) #trials x 241 x 4 / trials x 210 x 3
    ty_bins = np.arange(tmin, tmax+binsize, binsize)
    ty = 0.5*(ty_bins[1:] + ty_bins[:-1])
    inds = np.arange(len(ty))
    
    nbuf = 5
    buf = nbuf*0.02
    tx_bins = np.arange(tmin-buf, tmax+binsize+buf, binsize)
    
    if type_ == 'wds':
        tx = times[(times <= tmax) & (times >= tmin)]
    else:
        tx = times
    
    #print(X.shape, tx.shape)
    # ntrial x nbins+2*buf x 4
    X = np.array([binned_statistic(tx, x_t.T, statistic = 'mean', bins = tx_bins)[0].T for x_t in X])
    X[np.isnan(X)] = 0
    
    if half == 'first':
        X = X[0::2, ...]
        Y = Y[0::2, ...]
    elif half == 'last':
        X = X[1::2, ...]
        Y = Y[1::2, ...]
    
    #ntrials x nbins x nfeatures (limb x coord x timebins)
    X = np.array([[X[t, ind:int(ind+2*nbuf+1), :].flatten() for ind in inds] for t in range(X.shape[0])])
    timecol = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))[..., None]
    #X = np.concatenate([X, timecol], axis = -1)
    
    X = np.array([X[..., f].flatten() for f in range(X.shape[-1])]).T #trials*timepoints x features
    X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
    Y = Y.flatten()[:, None]
    
    X = np.column_stack([np.ones_like(Y), X]) #add offset
    
    return X, Y

def neg_log_lik_lnp(theta, X, Y):
    '''X: T x F,  Y: T x 1, theta: F x 1'''
    # Compute the Poisson log likeliood
    rate = np.exp(X @ theta) # T x 1
    log_lik = Y.T @ np.log(rate) - rate.sum()
    return -log_lik

def sample_data(rat):
    for unum in np.sort(list(rat['units'].keys())):
        days = np.sort(list(rat['units'][unum].keys()))
        for day in days:
            if rat['units'][unum][day]['glm']:
                rate = rat['units'][unum][day]['glm_rate_w']
                peth_syn_t = np.random.poisson(rate)
                peth_syn = np.sum(peth_syn_t, axis = 0)
                
                if global_params['smooth_peth']:
                    conv_key = 'conv_wds' if ('wds' in rat['name']) else 'conv_2tap'
                    sig = global_params[conv_key]/global_params['binsize']
                    peth_syn = gaussian_filter1d(peth_syn, sig, mode = 'nearest') #3/4 binsize
                
                rat['units'][unum][day]['glm_peth_w'] = peth_syn
    return rat

def gen_synthetic_rat(rat):
    type_, kin = ('wds', 'kinematics_w')  if 'wds' in rat['name'] else ('twotap', 'vels_w')
    rat['glm_params'] = {}
    for unum in np.sort(list(rat['units'].keys())):
        
        days = np.sort(list(rat['units'][unum].keys()))
        
        if len(days) > 0:
            ### fit GLM model to first day of recording ###
            d0 = days[0]
            peths = rat['units'][unum][d0]['peth_w_t']
            vels = rat['trials'][d0][kin]
            times = rat['trials'][d0]['times']
            X, Y = make_design_mat(times, peths, vels, type_ = type_)
            
            F = X.shape[-1] #number of features
            
            if np.sum(Y) > F:
                w0 = np.random.normal(0, .0002, F)
                # Find parameters that minimize the negative log likelihood function
                res = minimize(neg_log_lik_lnp, w0, args=(X, Y), options = {'maxiter': 5000, 'maxfun': 100000}, method = 'L-BFGS-B')
                w = res['x']
                print('fitting glm:', unum, res['success'], np.sum(Y), X.shape, Y.shape)
                rat['glm_params'][unum] = res
            else:
                res = {'success': False}
        
        ### generate synthetic data for all trials ###
        for day in days:
            
            if res['success']:
                
                peths = rat['units'][unum][day]['peth_w_t']
                vels = rat['trials'][day][kin]
                times = rat['trials'][day]['times']
                X, Y = make_design_mat(times, peths, vels, type_ = type_)
                
                rate = (np.exp(X @ w)).reshape(len(peths), -1)
                peth_syn_t = np.random.poisson(rate)
                peth_syn = np.sum(peth_syn_t, axis = 0)
                
                if global_params['smooth_peth']:
                    conv_key = 'conv_wds' if ('wds' in rat['name']) else 'conv_2tap'
                    sig = global_params[conv_key]/global_params['binsize']
                    peth_syn = gaussian_filter1d(peth_syn, sig, mode = 'nearest')
                
                rat['units'][unum][day]['glm_rate_w'] = rate
                rat['units'][unum][day]['glm_peth_w'] = peth_syn
                
                rat['units'][unum][day]['glm'] = res['success']
            else:
                rat['units'][unum][day]['glm'] = False
                       
    return rat

