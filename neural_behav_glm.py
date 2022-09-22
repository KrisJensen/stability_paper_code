#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:25:56 2021

@author: kris
"""

import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import copy
from utils import global_params

def get_regressors(day, unum, rat):
    type_, kin = ('wds', 'kinematics_w')  if 'wds' in rat['name'] else ('twotap', 'vels_w')
    peths = rat['units'][unum][day]['peth_w_t']
    vels = copy.copy(rat['trials'][day][kin])
    times = rat['trials'][day]['times']
    keep = rat['units'][unum][day]['keep']
    keys = list(vels.keys())
    if type_ == 'wds':
        for key in keys:
            vels[key+'_abs'] = np.abs(vels[key])
    return peths, vels, times, keep, type_

def make_design_mat(times, peths, vels, tmin = 'default', tmax = 'default', half = 'both', type_ = 'twotap', keep = []):
    
    assert type_ in ['twotap', 'wds']
    if tmin == 'default':
        tmin = -0.2 if type_ == 'wds' else -0.1
    if tmax == 'default':
        tmax = 0.5 if type_ == 'wds' else 0.8
    binsize = global_params['binsize']
    
    X, Y = vels, peths
    X = np.concatenate([X[limb] for limb in X.keys()], axis = -1) #trials x 241 x 4 / trials x 210 x 3

    if len(keep) > 0:
        X, Y = X[keep, ...], Y[keep, ...] #only keep the data in 'keep'

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
    rat['glm_params'] = {}
    for unum in np.sort(list(rat['units'].keys())):
        
        days = np.sort(list(rat['units'][unum].keys()))
        
        if len(days) > 0:
            ### fit GLM model to first day of recording ###
            d0 = days[0]
            peths, vels, times, keep, type_ = get_regressors(d0, unum, rat)
            X, Y = make_design_mat(times, peths, vels, type_ = type_, keep = keep)
            
            F = X.shape[-1] #number of features
            
            if np.sum(Y) > F:
                w0 = np.random.normal(0, .0002, F)
                # Find parameters that minimize the negative log likelihood function
                res = minimize(neg_log_lik_lnp, w0, args=(X, Y), options = {'maxiter': 20000, 'maxfun': 200000}, method = 'L-BFGS-B')
                w = res['x']
                rat['glm_params'][unum] = res
            else:
                res = {'success': False}
        
        ### generate synthetic data for all trials ###
        for day in days:
            if res['success']:
                peths, vels, times, keep, type_ = get_regressors(day, unum, rat)
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
